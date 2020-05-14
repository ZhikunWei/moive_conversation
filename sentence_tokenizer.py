from nlu_module import *
import numpy as np

class nlu_tokenizer:
    def __init__(self, lower=True):
        super().__init__()
        self.info_extracter = info_module()
        self.word2vec_raw = get_word2vec(False)
        self.lower = lower
        self.word2idx = {
            '<unk>': 1,
            '<pad>': 0,
            'MOV': 2,
            'YEAR': 3,
            'TIME': 4,
            'GENRE': 5,
            'RATE': 6,
            'PERSON': 7
        }
        # currently we just use words in word2vec
        idx = 8
        for w in self.info_extracter.word2vec.keys():
            if w.lower() in self.word2idx:
                # reduce duplicated word
                continue
            if w.lower() not in self.word2vec_raw:
                self.word2vec_raw[w.lower()] = self.word2vec_raw[w]
            self.word2idx[w.lower()] = idx
            idx += 1

        self.idx2word = [None] * idx
        for word in self.word2idx:
            self.idx2word[self.word2idx[word]] = word
    
    def parse_sentence(self, sentence):
        tokenized, info = self.info_extracter.parse_sentence(sentence)
        if self.lower:
            for i in range(len(tokenized)):
                if tokenized[i] in ['MOV', 'YEAR', 'TIME', 'GENRE', 'RATE', 'PERSON']:
                    continue
                tokenized[i] = tokenized[i].lower()
        return tokenized, info
    
    def token_to_idx(self, tokenized):
        return [self.word2idx.get(x, self.word2idx.get(x.lower(), 1)) for x in tokenized]
    
    def get_pretrained_embedding(self):
        emb = []
        for word in self.idx2word:
            if word == '<pad>':
                emb.append(np.zeros(300))
            elif word in ['<unk>', 'MOV', 'YEAR', 'TIME', 'GENRE', 'RATE', 'PERSON']:
                e = np.random.randn(300)
                e = e / np.sqrt(np.sum(e ** 2))
                emb.append(e)
            else:
                emb.append(self.word2vec_raw.get(word))
        return np.array(emb)