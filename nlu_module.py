import nltk
import traceback
import numpy as np
from tqdm import tqdm
import os
import pickle
from common_module import *

MAX_RANGE = 8

def tokenize(sentence):
    return nltk.tokenize.word_tokenize(sentence)

def personlize(tokenized):
    for i in range(len(tokenized)):
        if tokenized[i] == 'PERSON':
            tokenized[i] = 'person'
        if tokenized[i] == 'MOV':
            tokenized[i] = 'xxxmov'
    parsed = nltk.ne_chunk(nltk.pos_tag(tokenized))
    persons = []
    processed = []
    for p in parsed:
        if type(p) == nltk.tree.Tree:
            processed.append(p.label())
            if p.label() == 'PERSON':
                persons.append(' '.join([x[0] for x in p.leaves()]))
        else:
            processed.append(p[0])
    for i in range(len(processed)):
        if processed[i] == 'xxxmov':
            processed[i] = 'MOV'
    return processed, persons

def is_year(strs):
    try:
        year = int(strs)
        if 1800 <= year <= 2020:
            return True
    except:
        return False

def detect_year(tokenized):
    for i in range(len(tokenized)):
        if tokenized[i] == 'YEAR':
            tokenized[i] = 'xxxxyear'
    processed = []
    years = []
    for idx, token in enumerate(tokenized):
        min_year = -1
        max_year = -1
        if is_year(token):
            processed.append('YEAR')
            min_year = max_year = int(token)
        elif token.endswith('0s') and is_year(token[:-1]):
            processed.append('YEAR')
            min_year = int(token[:-1])
            max_year = int(token[:-1]) + 10
        else:
            processed.append(token)
            continue
        ranges = MAX_RANGE
        context = (tokenized[max(0, idx-ranges): idx+ranges])
        if 'in' in processed[-ranges:] or 'during' in processed[-ranges:]:
            if 'late' in processed[-ranges:]:
                min_year = (min_year + max_year) / 2
            elif 'early' in processed[-ranges:]:
                max_year = (min_year + max_year) / 2
        elif 'around' in processed[-ranges:] or 'about' in processed[-ranges:]:
            min_year -= 5
            max_year += 5
        elif 'before' in processed[-ranges:]:
            min_year = 1000
        elif 'after' in context:
            max_year = 2020
        elif 'later' in context:
            max_year = 2020
        elif 'older' in context:
            min_year = 1000
        elif 'newer' in context:
            max_year = 2020
        years.append((min_year, max_year))
    return processed, years

def parse_int(token):
    int_eng = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']
    if token in int_eng:
        return int_eng.index(token)
    try:
        return float(token)
    except:
        return -1

def detect_time(tokenized):
    for i in range(len(tokenized)):
        if tokenized[i] == 'TIME':
            tokenized[i] = 'xxxxtime'
    processed = []
    time = []
    idx = 0
    while idx < len(tokenized):
        token = tokenized[idx]
        added = 1
        if parse_int(token) != -1 and len(tokenized) > idx + 1 and tokenized[idx + 1] in ['hour', 'hours']:
            processed.append('TIME')
            time.append(parse_int(token) * 60)
            added = 2
        elif parse_int(token) != -1 and len(tokenized) > idx + 1 and tokenized[idx + 1] in ['min', 'minute', 'minutes']:
            processed.append('TIME')
            time.append(parse_int(token))
            added = 2
        elif parse_int(token[:-1]) != -1 and token[-1] == 'h':
            processed.append('TIME')
            time.append(parse_int(token[:-1]) * 60)
        elif parse_int(token[:-1]) != -1 and token[-1] == 'm':
            processed.append('TIME')
            time.append(parse_int(token[:-1]))
        elif parse_int(token[:-3]) != -1 and token[-3:] == 'min':
            processed.append('TIME')
            time.append(parse_int(token[:-3]))
        elif parse_int(token) != -1 and len(tokenized) > idx + 4 and tokenized[idx + 1: idx + 4] == ['and', 'a', 'half']:
            # and a half hours
            if tokenized[idx + 4] in ['hour', 'hours']:
                processed.append('TIME')
                time.append(parse_int(token) * 60 + 30)
                added = 5
            if tokenized[idx + 4] in ['minute', 'minutes']:
                processed.append('TIME')
                time.append(parse_int(token) + 0.5)
                added = 5
        elif tokenized[idx:idx + 3] == ['and', 'a', 'half']:
            # xxx hours and a half
            if tokenized[idx - 1] in ['hour', 'hours'] and processed[-1] == 'TIME':
                time[-1] += 30
                added = 3
            # xxx minutes and a half
            elif tokenized[idx - 1] in ['minute', 'minutes'] and processed[-1] == 'TIME':
                time[-1] += 0.5
                added = 3
            else:
                idx += 1
                processed += ['and']
                continue
        elif tokenized[idx:idx + 2] == ['a', 'half']:
            # it is probabily a conjunction
            tokenized[idx] = '0.5'
            del tokenized[idx + 1]
            continue
        else:
            processed.append(token)
            idx += 1
            continue
        idx += added

    idx = 0
    time_idx = 0
    while idx < len(processed):
        token = processed[idx]
        if processed[idx:idx+3] == ['TIME','and','TIME']:
            time[time_idx] += time[time_idx + 1]
            del time[time_idx + 1]
            del processed[idx + 1]
            del processed[idx + 1]
            continue
        if processed[idx] == 'TIME':
            # we need to convert them to range
            # less than, at most xxx
            context = processed[max(idx - MAX_RANGE, 0): idx + 3]
            if 'less' in context or 'most' in context or 'only' in context or 'shorter' in context:
                time[time_idx] = (0, time[time_idx])
            # around, about ...
            elif 'around' in context or 'about' in context:
                time[time_idx] = (time[time_idx] - 30, time[time_idx] + 30)
            # more than, at least
            elif 'more' in context or 'least' in context or 'longer' in context:
                time[time_idx] = (time[time_idx], 540)
            else:
                if time_idx == len(time):
                    print(processed)
                    print(time)
                    exit(0)
                time[time_idx] = (time[time_idx] - 10, time[time_idx] + 10)
            time_idx += 1
        idx += 1

    return processed, time

def detect_genre(tokenized, stopword, word2vec):
    for i in range(len(tokenized)):
        if tokenized[i] == 'GENRE':
            tokenized[i] = 'xxxxgenre'

    genre_ret = []
    
    genre_expand = {
        'News': ['news', 'report', 'information', 'telling'],
        'Reality-TV': ['tv', 'television', 'soap', 'opera', 'serial', 'series'],
        'Music': ['music', 'musical', 'musician', 'song'],
        'Documentary': ['document', 'documentary', 'documentation', 'record', 'recordings'],
        'Romance': ['romantic', 'romance', 'lovely', 'sweet', 'nice'],
        'Talk-Show': ['talk', 'speech', 'speechful', 'talking', 'lecture', 'course'],
        'Biography': ['biography', 'bio', 'memoir', 'memory', 'memorial'],
        'Game-Show': ['game', 'gaming', 'playing'],
        'War': ['war', 'battle', 'battling', 'fire', 'gun', 'guns', 'fight', 'fighting', 'passion', 'passionate'],
        'Family': ['family', 'city', 'civil', 'day', 'daily'],
        'Fantasy': ['fantasy', 'fantistic', 'strange', 'legend', 'legendary', 'mystery', 'mysterious', 'wired', 'tale', 'fiction', 'novel', 'myth', 'figment', 'untruth','fable','fabrication', 'sci-fi', 'science'],
        'Adventure': ['adventurous', 'adventure', 'travel', 'traveling', 'venture', 'chance', 'wager', 'peril', 'hazard', 'jeopardy', 'danger'],
        'Drama': ['drama', 'dramatic', 'acting', 'farce', 'showmanship', 'act', 'show', 'play', 'footlights', 'stage', 'theatre'],
        'Horror': ['horror', 'horrorible', 'scary', 'scare', 'fear', 'fearful', 'terror', 'terrible', 'hatred', 'dread', 'trepid', 'trepidation'],
        'Sport': ['sport', 'sports', 'comeptition', 'competing', 'compete', 'athletic', 'athlete'],
        'History': ['old', 'historical', 'past', 'annal', 'annals', 'history'],
        'Comedy': ['happy', 'smile', 'comdies', 'parody', 'pleasure', 'release', 'comedy', 'fun', 'funny'],
        'Animation': ['anime', 'animation', 'comic', 'picture', 'comics'],
        'Action': ['action', 'act', 'passion', 'passionate'],
        'Crime': ['criminal', 'cirme', 'violation', 'police'],
    }

    genre_expand_reverse = {}

    for k in genre_expand:
        for value in genre_expand[k]:
            if value in genre_expand_reverse:
                genre_expand_reverse[value].append(k)
            else:
                genre_expand_reverse[value] = [k]

    genres = [x.lower() for x in genre_expand_reverse.keys() if x.lower() in word2vec]
    genre_matrix = np.array([word2vec.get(x) for x in genres])
    for idx, token in enumerate(tokenized):
        if token in stopword or token in set('!~@#$%^&*()_+-=[];\',./{}|:"<>?\\"~`') or token in ['TIME', 'GPE', 'PERSON', 'YEAR', 'RATE', 'like', 'love', 'enjoy', 'later', 'newer', 'older']:
            continue
        if token in genre_expand_reverse:
            current = set()
            current.update(genre_expand_reverse[token])
            genre_ret.append(current)
            tokenized[idx] = 'GENRE'
        elif token in word2vec:
            vec = word2vec.get(token)[None,:]
            similar = (vec @ genre_matrix.T)[0]
            current = set()
            for idx_, value in enumerate(similar):
                if value >= 0.8:
                    current.update(genre_expand_reverse[genres[idx_]])
            if len(current) == 0:
                continue
            genre_ret.append(current)
            tokenized[idx] = 'GENRE'
    return tokenized, genre_ret

def detect_rate(tokenized):
    for i in range(len(tokenized)):
        if tokenized[i] == 'RATE':
            tokenized[i] = 'xxxxrate'
    idx = 0
    rate = []
    while idx < len(tokenized):
        if tokenized[idx] in ['rate', 'rating', 'rates', 'ratings', 'score', 'scoring', 'scores']:
            # we just change all the possible numbers into RATE around idx
            for i in range(max(0, idx - MAX_RANGE), min(len(tokenized), idx + MAX_RANGE)):
                if parse_int(tokenized[i]) != -1:
                    rate.append(parse_int(tokenized[i]))
                    tokenized[i] = 'RATE'
        idx += 1
    
    idx = 0
    rate_idx = 0
    while idx < len(tokenized):
        if tokenized[idx] == 'RATE':
            context = tokenized[max(idx - MAX_RANGE, 0): idx + 3]
            if 'more' in context or 'higher' in context or 'at least' in context:
                rate[rate_idx] = (rate[rate_idx], 10)
            elif 'less' in context or 'lower' in context or 'at most' in context:
                rate[rate_idx] = (0, rate[rate_idx])
            else:
                rate[rate_idx] = (rate[rate_idx] - 0.5, rate[rate_idx] + 0.5)
            rate_idx += 1
        idx += 1
    return tokenized, rate

def detect_movie(tokenized):
    tokenized, movie = words_matching(tokenized, ['MOV'], 'MOV', True)
    movies = []
    movies_real = []
    idx = 0
    i = 0

    # detect something like < ... >
    in_a_movie = False
    start_idx = -1
    end_idx = -1
    while i < len(tokenized):
        if tokenized[i] == '<':
            in_a_movie = True
            start_idx = i
        elif tokenized[i] == '>':
            if in_a_movie:
                in_a_movie = False
                end_idx = i
                movies_real.append(' '.join(tokenized[start_idx + 1: end_idx]))
                tokenized[start_idx] = 'mmm'
                del tokenized[start_idx + 1: end_idx + 1]
                i = start_idx
        i += 1

    i_real = 0
    i = 0
    while i < len(tokenized):
        token = tokenized[i]
        if token == 'mmm':
            movies.append(movies_real[i_real])
            i_real += 1
            tokenized[i] = 'MOV'
        elif token == 'MOV':
            movies.append(movie[idx])
            idx += 1
        elif token.startswith('@'):
            movies.append(token)
            tokenized[i] = 'MOV'
            del tokenized[i + 1]
        i += 1
    return tokenized, movies

class info_module:
    def __init__(self):
        self.word2vec = get_word2vec()
        self.stopword = get_stop_words()
    
    def parse_sentence(self, sentence):
        # sentence: the raw str
        tokenized = tokenize(sentence)
        tokenized, movie_list = detect_movie(tokenized)
        tokenized, person_list = personlize(tokenized)
        tokenized, year = detect_year(tokenized)
        tokenized, time = detect_time(tokenized)
        tokenized, rate = detect_rate(tokenized)
        tokenized, genre = detect_genre(tokenized, self.stopword, self.word2vec)
        return tokenized, {
            'movie': movie_list,
            'person': person_list,
            'year': year,
            'time': time,
            'rate': rate,
            'genre': genre
        }

if __name__ == '__main__':
    while 1:
        try:
            inputs = input('Your Turn > ')
            info_parser = info_module()
            tokenized, info = info_parser.parse_sentence(inputs)
            print('info      >', str(info))
        except:
            traceback.print_exc()
            break