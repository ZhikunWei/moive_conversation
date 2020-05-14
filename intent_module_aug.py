"""
Some techniques:
- use pretrained embedding, but adapt them
- use pretrained embedding, keep it unchanged
- use pretrained embedding, keep it unchanged, add noise
"""

MAX_LENGTH = 60
MAX_VOCAB = 20000

import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score, hamming_loss
from torch import optim
from common_module import get_word2vec
from nlu_module import tokenize
import json
import os

def macroF1(pred, ground_truth):
    confusion_matrix = np.zeros((3, 3))
    for p, g in zip(pred, ground_truth):
        confusion_matrix[p][g] += 1
    print('confuse mat:')
    print(confusion_matrix)
    precision = [None] * 3
    recall = [None] * 3
    F_1 = [None] * 3
    for i in range(3):
        precision[i] = confusion_matrix[i][i] / np.sum(confusion_matrix[i])
        recall[i] = confusion_matrix[i][i] / np.sum(confusion_matrix[:,i])
        F_1[i] = (precision[i] * recall[i] * 2) / (precision[i] + recall[i])
    print(precision)
    print(recall)
    print(F_1)
    met = np.mean(F_1)
    print('macro F1', met)
    return met

def padding(ele, max_length=128, padIdx=0):
    if len(ele) >= max_length:
        return ele[:max_length]
    if isinstance(ele, list):
        ele += [padIdx] * (max_length - len(ele))
        return ele
    else:
        ele = ele.tolist()
        ele += [padIdx] * (max_length - len(ele))
        return ele

class cnn(nn.Module):
    def __init__(self, pretrained_emb, num_class, freeze=False, noise=0.0):
        super(cnn, self).__init__()
        self.kernel = [2,3,4,5]
        self.num_kernel = [64,64,64,64]
        self.noise = noise
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(pretrained_emb, dtype=torch.float32), freeze=freeze, padding_idx=0)
        self.filters = nn.ModuleList()
        self.num_class = num_class
        for i in range(len(self.kernel)):
            self.filters.add_module("kernel_%d" % (i), nn.Conv2d(1, self.num_kernel[i], (self.kernel[i], 300), padding=(self.kernel[i] // 2, 0)))
        self.classifier = nn.Linear(sum(self.num_kernel), self.num_class)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.embedding(x)  # (N, W, D)
        x = x + torch.randn_like(x) * self.noise
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.leaky_relu(conv(x)).squeeze(3) for conv in self.filters]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.classifier(x)  # (N, C)
        return logit
    
    def get_prediction(self, x):
        x = self.embedding(x)  # (N, W, D)
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.leaky_relu(conv(x)).squeeze(3) for conv in self.filters]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        logit = self.classifier(x)  # (N, C)
        return F.softmax(logit, dim=1)

class intent_detection_module(nn.Module):
    def  __init__(self, train_data=None, valid_data=None, device=torch.device('cuda'), mod=0):
        super(intent_detection_module, self).__init__()
        # we do not use learned tokenizer
        print('begin loading word2vec ...')
        self.word2vec = get_word2vec()
        self.word2idx = {'<pad>': 0, '<unk>': 1}
        self.idx2word = ['<pad>', '<unk>']
        pretrained_embedding = [
            np.zeros(300),
            np.random.randn(300) * np.std(self.word2vec[','])
        ]

        idx = 2
        for w in self.word2vec:
            w_lower = w.lower()
            if w_lower in self.word2idx:
                continue
            self.word2idx[w_lower] = idx
            self.idx2word.append(w_lower)
            idx += 1
            if w_lower in self.word2vec:
                pretrained_embedding.append(self.word2vec[w_lower])
            else:
                pretrained_embedding.append(self.word2vec[w])
        pretrained_embedding = np.array(pretrained_embedding)

        self.mod = mod
        if mod == 0:
            self.model = cnn(pretrained_embedding, 3, freeze=False, noise=0)
        elif mod == 1:
            self.model = cnn(pretrained_embedding, 3, freeze=True, noise=0)
        elif mod == 2:
            self.model = cnn(pretrained_embedding, 3, freeze=True, noise=0.001)
        elif mod == 3:
            self.model = cnn(pretrained_embedding, 3, freeze=False, noise=0.001)
        self.model.to(device)

        if not train_data is None:
            self.train_dataset = [self.sentence2idx(x['data']) for x in train_data]
            self.train_dataset = [padding(x, MAX_LENGTH, 0) for x in self.train_dataset]
            self.train_dataset = torch.tensor(self.train_dataset, dtype=torch.long)
            train_label, train_label_id = self.process_label(train_data)
            self.train_label = train_label
            self.train_label_id = train_label_id
        
        if not valid_data is None:
            self.valid_dataset = [self.sentence2idx(x['data']) for x in valid_data]
            self.valid_dataset = [padding(x, MAX_LENGTH, 0) for x in self.valid_dataset]
            self.valid_dataset = torch.tensor(self.valid_dataset, dtype=torch.long)

            valid_label, valid_label_id = self.process_label(valid_data)
            self.valid_label = valid_label
            self.valid_label_id = valid_label_id

            self.valid_dataset = TensorDataset(self.valid_dataset, self.valid_label)
            self.valid_sampler = SequentialSampler(self.valid_dataset)
            self.valid_loader = DataLoader(self.valid_dataset, sampler=self.valid_sampler, batch_size=1024)

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        print('model init done')

    def sentence2idx(self, sent):
        tokenized = tokenize(sent)
        return [self.word2idx.get(x.lower(), 1) for x in tokenized]

    def generate_train_batch(self, batch_size):
        idx = []
        idx.extend(np.random.choice(self.train_label_id[0], size=(batch_size // 3)))
        idx.extend(np.random.choice(self.train_label_id[1], size=(batch_size // 3)))
        idx.extend(np.random.choice(self.train_label_id[2], size=(batch_size // 3)))
        return self.train_dataset[idx], self.train_label[idx]

    def process_label(self, data):
        labels = [[], [], []]
        all_label = []
        for i, x in enumerate(data):
            if x['intent'] == 'preference':
                labels[0].append(i)
                all_label.append(0)
            elif x['intent'] == 'factoid':
                labels[1].append(i)
                all_label.append(1)
            else:
                labels[2].append(i)
                all_label.append(2)
        return torch.tensor(all_label, dtype=torch.long), labels

    def save_module(self):
        if not os.path.exists('model/intent'):
            os.makedirs('model/intent')

        self.model.cpu()
        
        if self.mod == 0:
            torch.save(self.model.state_dict(), 'model/intent/model_param_aug.pt')
        elif self.mod == 1:
            torch.save(self.model.state_dict(), 'model/intent/model_param_aug_freeze.pt')
        elif self.mod == 2:
            torch.save(self.model.state_dict(), 'model/intent/model_param_aug_freeze_0.001.pt')
        elif self.mod == 3:
            torch.save(self.model.state_dict(), 'model/intent/model_param_aug_0.001.pt')

        self.model.cuda()

    @classmethod
    def load_module(cls, mod=0):
        if mod == 0:
            state_dict = torch.load('model/intent/model_param_aug.pt')
        elif mod == 1:
            state_dict = torch.load('model/intent/model_param_aug_freeze.pt')
        elif mod == 2:
            state_dict = torch.load('model/intent/model_param_aug_freeze_0.001.pt')
        elif mod == 3:
            state_dict = torch.load('model/intent/model_param_aug_0.001.pt')
        mod = cls(None, None, device=torch.device('cpu'), mod=mod)
        mod.model.load_state_dict(state_dict)
        return mod

    def predict_instances(self, sentences):
        texts = [self.sentence2idx(s) for s in sentences]
        texts = [padding(x, MAX_LENGTH, 0) for x in texts]
        texts = torch.tensor(texts, dtype=torch.long)
        self.model.eval()
        predicted = self.model.get_prediction(texts)
        return predicted.detach().cpu().numpy()

    def test_module(self, test_dataset, batch_size=1024):
        self.model.eval()
        test_label, _ = self.process_label(test_dataset)
        test_dataset = [self.sentence2idx(x['data']) for x in test_dataset]
        test_dataset = [padding(x, MAX_LENGTH, 0) for x in test_dataset]
        test_dataset = torch.tensor(test_dataset, dtype=torch.long)

        test_dataset = TensorDataset(test_dataset, test_label)

        test_sampler = SequentialSampler(test_dataset)
        test_loader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size)

        self.model.cuda()

        preds = []
        ground_truth = []

        with torch.no_grad():
            for data in test_loader:
                logit = self.model(data[0].cuda())
                labels = logit.argmax(dim = 1)
                ground_truth.extend(data[1].numpy().tolist())
                preds.extend(labels.detach().cpu().numpy().tolist())

        confusion_matrix = np.zeros((3, 3))
        for p, g in zip(preds, ground_truth):
            confusion_matrix[p][g] += 1
        confusion_matrix = np.array(confusion_matrix)
        print('confuse_mat:')
        print(confusion_matrix)
        precision = [None] * 3
        recall = [None] * 3
        F_1 = [None] * 3
        for i in range(3):
            precision[i] = confusion_matrix[i][i] / np.sum(confusion_matrix[i])
            recall[i] = confusion_matrix[i][i] / np.sum(confusion_matrix[:,i])
            F_1[i] = (2 * precision[i] * recall[i]) / (precision[i] + recall[i])
        print(precision)
        print(recall)
        print(F_1[2], F_1[0], F_1[1])
        met = np.mean(F_1)
        #acc = sum(preds) / len(preds)
        print('macro-F1: %f' % (met))

        label = test_label.numpy()

        preds = np.array(preds)
        ground_truth = np.array(ground_truth)
        preds = (preds == ground_truth)

        acc = sum(preds) / len(preds)
        acc_p = sum(preds[label == 0]) / len(preds[label == 0])
        acc_f = sum(preds[label == 1]) / len(preds[label == 1])
        acc_q = sum(preds[label == 2]) / len(preds[label == 2])

        print('total acc: ', acc)
        print('preference acc: ', acc_p)
        print('factoid acc: ', acc_f)
        print('query acc: ', acc_q)
                
    def train_module(self, max_epoch = 5 * 1000, batch_size = 128 * 3):
        self.model.train()
        max_acc = 0
        patience = 0
        for e in range(max_epoch):
            train, label = self.generate_train_batch(batch_size)
            train = train.cuda()
            label = label.cuda()
            self.optimizer.zero_grad()
            logit = self.model(train)
            loss = F.cross_entropy(logit, label)
            loss.backward()
            self.optimizer.step()

            print('\repoch: %d, loss: %f' % (e, loss.item()), end='')

            if e % 100 == 0 and e > 0:
                print('')
                self.model.eval()
                preds = []
                ground_truth = []
                for data in self.valid_loader:
                    logit = self.model(data[0].cuda())
                    labels = logit.argmax(dim = 1)
                    preds.extend(labels.detach().cpu().numpy().tolist())
                    ground_truth.extend(data[1].numpy().tolist())

                    #correct_list = (labels.detach().cpu().numpy() == data[1].numpy())
                    #preds.extend(correct_list.tolist())
            
                #met = cohen_kappa_score(preds, ground_truth)
                confusion_matrix = np.zeros((3, 3))
                for p, g in zip(preds, ground_truth):
                    confusion_matrix[p][g] += 1
                confusion_matrix = np.array(confusion_matrix)
                print('confuse_mat:')
                print(confusion_matrix)
                precision = [None] * 3
                recall = [None] * 3
                F_1 = [None] * 3
                for i in range(3):
                    precision[i] = confusion_matrix[i][i] / np.sum(confusion_matrix[i])
                    recall[i] = confusion_matrix[i][i] / np.sum(confusion_matrix[:,i])
                    F_1[i] = (2 * precision[i] * recall[i]) / (precision[i] + recall[i])
                print(precision)
                print(recall)
                print(F_1)
                met = np.mean(F_1)
                #acc = sum(preds) / len(preds)
                print('epoch %d: kappa: %f patience: %d' % (e, met, patience))

                if met > max_acc:
                    max_acc = met
                    self.save_module()
                    patience = 0
                
                patience += 1
                if patience > 5:
                    break
                self.model.train()

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='test')
    parser.add_argument('--mod', default=0, type=int)
    args = parser.parse_args()
    mode = args.mode
    mod = args.mod


    # train the module

    if mode == 'train':
        train = json.load(open('data/intent/train.M.json', 'r'))
        valid = json.load(open('data/intent/valid.M.json', 'r'))
        module = intent_detection_module(train, valid, mod=mod)
        module.train_module()

    # test the module
    elif mode == 'test':
        test = json.load(open('data/intent/test.M.json', 'r'))

        module = intent_detection_module.load_module(mod=mod)
        module.test_module(test)

    # test for interactive
    elif  mode == 'interactive':
        module = intent_detection_module.load_module(mod=mod)

        while 1:
            try:
                text = input('> enter your sentences <')
                prob = module.predict_instances([text])[0]
                print('preference: %f, factoid: %f, query: %f' % (prob[0], prob[1], prob[2]))
            except:
                print('exit')
                break
        #'''