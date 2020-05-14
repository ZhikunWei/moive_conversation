# query analysis module
# used for extracting the 3 parts of query

# the questions are mainly:
# (NAME) director_of (MOVIE)
# (NAME) act_in (MOVIE)
# (NAME) writter_of (MOVIE)
# (number) rate_of (MOVIE)
# (year) year_of (MOVIE)
# (type) genre_of (MOVIE)

DIRECTOR_OF = 'director_of'
ACT_IN = 'act_in'
WRITTER_OF = 'writter_of'
RATE_OF = 'rate_of'
YEAR_OF = 'year_of'
GENRE_OF = 'genre_of'
DEFAULT = 'default'

from common_module import words_matching, get_stop_words, get_word2vec
from nlu_module import *

class query_extract_module:
    def __init__(self, word_to_vec, stop_words=[]):
        self.word_to_vec = word_to_vec
        self.stop_words = stop_words

    def _maybe_director(self, sent):
        s = [x for x in sent]
        _, info = words_matching(s, ['director', 'directing', 'directed', 'direct', 'directs'], '_', exact_match=False, word_to_vec=self.word_to_vec, ignore_words=self.stop_words)
        if len(info) > 0:
            return True
        return False

    def _maybe_actor(self, sent):
        s = [x for x in sent]
        _, info = words_matching(s, ['acts', 'stars', 'star', 'act', 'actor', 'actress', 'actors'], '_', exact_match=False, word_to_vec=self.word_to_vec, ignore_words=self.stop_words)
        if len(info) > 0:
            return True
        return False

    def _maybe_writter(self, sent):
        s = [x for x in sent]
        _, info = words_matching(s, ['writter', 'write', 'wrote', 'writes'], '_', exact_match=False, word_to_vec=self.word_to_vec, ignore_words=self.stop_words)
        if len(info) > 0:
            return True
        return False

    def _maybe_rate(self, sent):
        s = [x for x in sent]
        _, info = words_matching(s, ['rate', 'rates', 'ratings', 'rating'], '_', exact_match=False, word_to_vec=self.word_to_vec, ignore_words=self.stop_words)
        if len(info) > 0:
            return True
        return False

    def _maybe_year(self, sent):
        s = [x for x in sent]
        _, info = words_matching(s, ['shown', 'starts', 'start', 'starting', 'open', 'opening', 'year'], '_', exact_match=False, word_to_vec=self.word_to_vec, ignore_words=self.stop_words)
        if len(info) > 0:
            return True
        return False

    def _maybe_genre(self, sent):
        s = [x for x in sent]
        _, info = words_matching(s, ['kinds', 'kind', 'type', 'types', 'genre', 'genres'], '_', exact_match=False, word_to_vec=self.word_to_vec, ignore_words=self.stop_words)
        if len(info) > 0:
            return True
        return False

    def parse(self, sentence):
        if 'Who' in sentence or 'who' in sentence:
            # director_of, act_in, writter_of

            # director of
            typ = ''
            if self._maybe_director(sentence):
                typ = 'director_of'
            
            elif self._maybe_actor(sentence):
                typ = 'act_in'
            
            elif self._maybe_writter(sentence):
                typ = 'writter_of'

            else:
                return None

            _, mov = detect_movie(sentence)
            if len(mov) == 0:
                return [(None, typ, 'LAST')]
            elif len(mov) > 0:
                return [(None, typ, x) for x in mov]
            
        elif 'When' in sentence or 'when' in sentence:

            _, mov = detect_movie(sentence)
            if len(mov) == 0:
                return [(None, 'year_of', 'LAST')]
            elif len(mov) > 0:
                return [(None, 'year_of', x) for x in mov]

        elif 'What' in sentence or 'what' in sentence:
            # movie or year or type or rate(number)
            typ = None
            if self._maybe_actor(sentence):
                typ = 'act_in'
            elif self._maybe_director(sentence):
                typ = 'director_of'
            elif self._maybe_writter(sentence):
                typ = 'writter_of'
            
            _, ref = words_matching(sentence, ['it', 'them', 'that', 'this', 'he', 'his', 'him', 'her', 'hers', 'they'], 'REF', word_to_vec=self.word_to_vec, ignore_words=self.stop_words)

            if not typ is None:
                _, person = personlize(sentence)
                if len(person) == 0 and len(ref) != 0:
                    return [('LAST', typ, None)]
                elif len(person) > 0:
                    return [(x, typ, None) for x in person]
                return None

            # TODO: add rate, year, genre
            _, movie = detect_movie(sentence)

            if self._maybe_rate(sentence):
                _, rate = detect_rate(sentence)
                if len(rate) == 0 and len(movie) == 0:
                    return [('LAST?', 'rate_of', 'LAST?')]
                elif len(rate) == 0:
                    return [(None, 'rate_of', movie[0])]
                elif len(movie) == 0:
                    return [(rate[0], 'rate_of', None)]
                else:
                    return None
            
            elif self._maybe_genre(sentence):
                _, gen = detect_genre(sentence, stopword=self.stop_words, word2vec=self.word_to_vec)
                if len(gen) == 0 and len(movie) == 0:
                    return [('LAST?', 'genre_of', 'LAST?')]
                elif len(gen) == 0:
                    return [(None, 'genre_of', movie[0])]
                elif len(movie) == 0:
                    return [(gen[0], 'genre_of', None)]
                else:
                    return None
            
            elif self._maybe_year(sentence):
                _, year = detect_year(sentence)
                if len(year) == 0 and len(movie) == 0:
                    return [('LAST?', 'year_of', 'LAST?')]
                elif len(year) == 0:
                    return [(None, 'year_of', movie[0])]
                elif len(movie) == 0:
                    return [(year[0], 'year_of', None)]
                else:
                    return None
            
            elif len(set(['him', 'his', 'them', 'they', 'hers', 'her', 'she']).intersection(set(ref))) > 0:
                # means person of
                return [(None, 'person_of', 'LAST')]

        else:
            # true/false question
            _, per = personlize(sentence)
            _, mov = detect_movie(sentence)
            if self._maybe_actor(sentence) and not (len(mov) == 0 and len(per) == 0):
                if len(mov) == 0:
                    return [(x, ACT_IN + '?', 'LAST') for x in per]
                elif len(per) == 0:
                    return [('LAST', ACT_IN + '?', mov[0])]
                else:
                    return [(per[0], ACT_IN + '?', mov[0])]
            elif self._maybe_writter(sentence) and not (len(mov) == 0 and len(per) == 0):
                if len(mov) == 0:
                    return [(x, WRITTER_OF + '?', 'LAST') for x in per]
                elif len(per) == 0:
                    return [('LAST', WRITTER_OF + '?', mov[0])]
                else:
                    return [(per[0], WRITTER_OF + '?', mov[0])]
            elif self._maybe_director(sentence)  and not (len(mov) == 0 and len(per) == 0):
                if len(mov) == 0:
                    return [(x, DIRECTOR_OF + '?', 'LAST') for x in per]
                elif len(per) == 0:
                    return [('LAST', DIRECTOR_OF + '?', mov[0])]
                else:
                    return [(per[0], DIRECTOR_OF + '?', mov[0])]

            _, yea = detect_year(sentence)
            _, gen = detect_genre(sentence, stopword=self.stop_words, word2vec=self.word_to_vec)
            _, rat = detect_rate(sentence)

            if len(yea) > 0:
                if len(mov) != 0:
                    return [(yea[0], YEAR_OF + '?', mov[0])]
                else:
                    return [(yea[0], YEAR_OF + '?', 'LAST')]
            
            elif len(gen) > 0:
                if len(mov) != 0:
                    return [(gen[0], GENRE_OF + "?", mov[0])]
                return [(gen[0], GENRE_OF + '?', 'LAST')]
            
            elif len(rat) > 0:
                if len(mov) != 0:
                    return [(rat[0], RATE_OF + '?', mov[0])]
                return [(rat[0], RATE_OF + '?', 'LAST')]
            
            return None

if __name__ == '__main__':
    qp = query_extract_module(get_word2vec(), get_stop_words())
    x = input('input > ')
    while x != 'q':
        token = tokenize(x)
        print('tokenized: ', token)
        print('extracted: ', qp.parse(token))
        x = input('input > ')