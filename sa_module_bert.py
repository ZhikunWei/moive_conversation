# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn.functional as F
from models.lcf_bert import LCF_BERT
#from models.aen import AEN_BERT
#from models.bert_spc import BERT_SPC
from pytorch_transformers import BertModel
from data_utils import Tokenizer4Bert
import argparse

def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x

def prepare_data(text_left, aspect, text_right, tokenizer):
    text_left = text_left.lower().strip()
    text_right = text_right.lower().strip()
    aspect = aspect.lower().strip()
    
    text_raw_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)            
    aspect_indices = tokenizer.text_to_sequence(aspect)
    aspect_len = np.sum(aspect_indices != 0)
    text_bert_indices = tokenizer.text_to_sequence('[CLS] ' + text_left + " " + aspect + " " + text_right + ' [SEP] ' + aspect + " [SEP]")
    text_raw_bert_indices = tokenizer.text_to_sequence(
        "[CLS] " + text_left + " " + aspect + " " + text_right + " [SEP]")
    bert_segments_ids = np.asarray([0] * (np.sum(text_raw_indices != 0) + 2) + [1] * (aspect_len + 1))
    bert_segments_ids = pad_and_truncate(bert_segments_ids, tokenizer.max_seq_len)
    aspect_bert_indices = tokenizer.text_to_sequence("[CLS] " + aspect + " [SEP]")

    return text_bert_indices, bert_segments_ids, text_raw_bert_indices, aspect_bert_indices

class module_opt:
    def __init__(self):
        super().__init__()
        self.bert_dim = 768
        self.hidden_dim = 300
        self.dropout = 0.1
        self.polarities_dim = 3
        self.device = torch.device('cpu')
        self.max_seq_len = 80
        self.SRD = 3
        self.local_context_focus = 'cdm'

class sa_module_bert:
    def __init__(self):
        opt = module_opt()
        bert = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = Tokenizer4Bert(80, 'bert-base-uncased')
        model = LCF_BERT(bert, opt).to(opt.device )
    
        print('loading sa module ...')
        model.load_state_dict(torch.load('state_dict/lcf_bert_movie_val_acc0.8203',map_location=torch.device('cpu')))
        model.eval()
        torch.autograd.set_grad_enabled(False)
        self.model = model
        self.opt = opt
    
    def get_result(self, tokenized, idx):
        opt = self.opt
        t = [x for x in tokenized]
        for ii in range(len(t)):
            if t[ii] == 'MOV':
                t[ii] = '<The Will Huntings>'
            elif t[ii] == 'PERSON':
                t[ii] = 'John Smith'
            elif t[ii] == 'RATE':
                t[ii] = '5.0'
            elif t[ii] == 'GENRE':
                t[ii] = 'scary'
            elif t[ii] == 'TIME':
                t[ii] = '1 hour'
            elif t[ii] == 'YEAR':
                t[ii] = '2012'
        left_part = ' '.join(t[:idx])
        right_part = ' '.join(t[idx + 1:])
        middle = t[idx]
        text_bert_indices, bert_segments_ids, text_raw_bert_indices, aspect_bert_indices = \
        prepare_data(left_part, middle, right_part, self.tokenizer)
        
        text_bert_indices = torch.tensor([text_bert_indices], dtype=torch.int64).to(opt.device)
        bert_segments_ids = torch.tensor([bert_segments_ids], dtype=torch.int64).to(opt.device)
        text_raw_bert_indices = torch.tensor([text_raw_bert_indices], dtype=torch.int64).to(opt.device)
        aspect_bert_indices = torch.tensor([aspect_bert_indices], dtype=torch.int64).to(opt.device)
        inputs = [text_bert_indices, bert_segments_ids, text_raw_bert_indices, aspect_bert_indices]
        outputs = self.model(inputs)
        t_probs = F.softmax(outputs, dim=-1).cpu().numpy()
        return t_probs.argmax(axis=-1) - 1

    def parse_sentence(self, tokenized, info):
        results = {
            'like': {
                'person': [],
                'year': [],
                'genre': [],
                'ratings': [],
                'time': [],
                'movie': []
            },
            'dislike': {
                'person': [],
                'year': [],
                'genre': [],
                'ratings': [],
                'time': [],
                'movie': []
            }
        }
        idx = [0] * 6
        for i in range(len(tokenized)):
            if tokenized[i] == 'MOV':
                sa = self.get_result(tokenized, i)
                if sa == -1:
                    results['dislike']['movie'].append(info['movie'][idx[0]])
                else:
                    results['like']['movie'].append(info['movie'][idx[0]])
                idx[0] += 1
            elif tokenized[i] == 'PERSON': 
                sa = self.get_result(tokenized, i)
                if sa == -1:
                    results['dislike']['person'].append(info['person'][idx[1]])
                else:
                    results['like']['person'].append(info['person'][idx[1]])
                idx[1] += 1
            elif tokenized[i] == 'RATE':
                sa = self.get_result(tokenized, i)
                if sa == -1:
                    results['dislike']['ratings'].append(info['rate'][idx[2]])
                else:
                    results['like']['ratings'].append(info['rate'][idx[2]])
                idx[2] += 1
            elif tokenized[i] == 'GENRE':
                sa = self.get_result(tokenized, i)
                if sa == -1:
                    results['dislike']['genre'].append(info['genre'][idx[3]])
                else:
                    results['like']['genre'].append(info['genre'][idx[3]])
                idx[3] += 1
            elif tokenized[i] == 'TIME':
                sa = self.get_result(tokenized, i)
                if sa == -1:
                    results['dislike']['time'].append(info['time'][idx[4]])
                else:
                    results['like']['time'].append(info['time'][idx[4]])
                idx[4] += 1
            elif tokenized[i] == 'YEAR':
                sa = self.get_result(tokenized, i)
                if sa == -1:
                    results['dislike']['year'].append(info['year'][idx[5]])
                else:
                    results['like']['year'].append(info['year'][idx[5]])
                idx[5] += 1
        return results
