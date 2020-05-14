from sentence_tokenizer import nlu_tokenizer
from intent_module_aug import intent_detection_module
from sa_module_bert import sa_module_bert
from query_module import query_extract_module
from common_module import get_stop_words, get_word2vec
from nlu_module import tokenize
import numpy as np

class nlu_subsystem():
    def __init__(self):
        super().__init__()
        self.intent_detection = intent_detection_module.load_module()
        self.tokenizer = nlu_tokenizer()
        self.sa = sa_module_bert()
        self.query = query_extract_module(self.tokenizer.info_extracter.word2vec, self.tokenizer.info_extracter.stopword)

    def _process_genres(self, data):
        m = set()
        for g in data['like']['genre']:
            m.update(g)
        data['like']['genre'] = list(m)
        m = set()
        for g in data['dislike']['genre']:
            m.update(g)
        data['dislike']['genre'] = list(m)
        
    def process_sentence(self, sentence, log=False):
        """
        Param:
        ------
        sentence: the raw str that represents the input sentence

        Return:
        ------
        A dict. Refer to protocol.md

        """
        prin = (lambda *x:None) if not log else print
        res = self.intent_detection.predict_instances([sentence])[0]
        res = np.argmax(res)
        if res == 0:
            # preference
            prin('> detect intent: preference')
            tokenized, info = self.tokenizer.parse_sentence(sentence)
            prin('> tokenized: ' + str(tokenized))
            prin('> info:', info)
            info['ratings']=info['rate']
            data = self.sa.parse_sentence(tokenized, info)
            self._process_genres(data)
            prin('> data:', data)
            return {'intent': 'recommendation', 'data': data}
        elif res == 1:
            # factoid
            prin('> detect intent: factoid')
            data = self.query.parse(tokenize(sentence))
            prin('> data:', data)
            return {'intent': 'factoid', 'data': data}
        else:
            # query
            prin('> detect intent: query')
            tokenized, info = self.tokenizer.parse_sentence(sentence)
            prin('> tokenized: ', tokenized)
            prin('> info:', info)
            info['ratings']=info['rate']
            data = {
                'like': info,
                'dislike': {'person': [], 'year': [], 'genre': [], 'ratings': [], 'time': [], 'movie': []}
            }
            self._process_genres(data)
            return {'intent': 'recommendation', 'data': data}

if __name__ == '__main__':
    mod = nlu_subsystem()
    q = input('> Input: ')
    while q != 'q':
        mod.process_sentence(q, log=True)
        q = input('> Input: ')