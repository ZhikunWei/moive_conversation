# contains some modules that will be used in common
import numpy as np
import os
import pickle
from tqdm import tqdm
import pickle

def words_matching(sentence, target_word_list, replace_words, exact_match=False, word_to_vec=None, ratio=0.7, ignore_words=['GPE', 'PER', 'LOC'], debug=False):
    # matching words
    if isinstance(replace_words, str):
        replace_words = [replace_words] * len(target_word_list)
    
    ignore_words = [x for x in ignore_words if x not in target_word_list] + ['TIME', 'GPE', 'PERSON', 'YEAR', 'RATE']
    if debug:
        print('him' in ignore_words)

    info_record = []

    if exact_match:
        for i, w in enumerate(target_word_list):
            for j in range(len(sentence)):
                if sentence[j] in ignore_words:
                    continue
                if sentence[j] == w:
                    sentence[j] = replace_words[i]
                    info_record.append([j, w])
        info_record = sorted(info_record, key=lambda x:x[0])
        info_record = [x[1] for x in info_record]
        return sentence, info_record
    
    assert word_to_vec is not None
    for i, w in enumerate(target_word_list):
        if not w in word_to_vec:
            # perform exact match
            if w in ignore_words:
                continue
            for j in range(len(sentence)):
                #if sentence[j] in ignore_words:
                #    continue
                if sentence[j] == w:
                    sentence[j] = replace_words[j]
                    info_record.append([j, w])
                    
        for j in range(len(sentence)):
            if sentence[j] in ignore_words:
                continue
            if sentence[j] not in word_to_vec:
                continue
            if sum(word_to_vec[sentence[j]] * word_to_vec[w]) >= ratio:
                # we have a match
                sentence[j] = replace_words[i]
                info_record.append([j, w])

    info_record = sorted(info_record, key=lambda x:x[0])
    info_record = [x[1] for x in info_record]
    return sentence, info_record

def get_stop_words():
    return [x.strip() for x in open('model/nlu/stopwords.txt', 'r').readlines() if not x.startswith('#')]

def get_word2vec(normalized=True):
    if os.path.exists('model/nlu/word2vec.pkl'):
        word2vec = pickle.load(open('model/nlu/word2vec.pkl', 'rb'))
    else:
        all_word = open('model/nlu/word2vec', 'r').readlines()[1:]
        word2vec = {}
        for line in tqdm(all_word):
            sp = line.split()
            word2vec[sp[0]] = np.asarray(sp[1:], dtype='float32')
        pickle.dump(word2vec, open('model/nlu/word2vec.pkl', 'wb'))
    if normalized:
        for k in word2vec:
            word2vec[k] /= np.sqrt(np.sum(word2vec[k] ** 2))
    return word2vec

def pickle_save(obj, path):
    if os.path.exists(path):
        os.remove(path)
    pickle.dump(obj, open(path, 'wb'))

def pickle_load(path):
    return pickle.load(open(path, 'rb'))