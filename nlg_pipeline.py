import random

class nlg_subsystem:
    def __init__(self):
        super().__init__()
        self.template = [
            ['I think you would like', '.'],
            ['Have you tried', '? I bet you will like them!'],
            ['Maybe you can try these:','.'],
            ['I find some movies that suits you most:','.'],
        ]
    
    def _wrap_movie_name(self, movie):
        return "<" + movie + ">"
    
    def _genre_of(self, genre, movie):
        if isinstance(genre, str):
            genre = [genre]
        return self._wrap_movie_name(movie) + ' is a movie of type ' + ' '.join(genre)
    
    def _year_of(self, year, movie):
        if isinstance(year, int):
            year = [year]
        return self._wrap_movie_name(movie) + ' was first shown on screen in year ' + str(year[0])
    
    def _duration_of(self, duration, movie):
        if isinstance(duration, list):
            duration = duration[0]
        duration = int(duration)
        hour = duration // 60
        minute = duration % 60
        if hour == 0:
            return self._wrap_movie_name(movie) + ' will cost you about ' + str(minute) + ' mins'
        if minute == 0:
            return self._wrap_movie_name(movie) + " will cost you about " + str(hour) + ' hour(s)'
        return self._wrap_movie_name(movie) + "will cost you about %d hour(s) and %d mins" % (hour, minute)
    
    def _wrap_multi_movies(self, movies):
        if len(movies) == 1:
            return self._wrap_movie_name(movies[0])
        word_list = [self._wrap_movie_name(x) for x in movies]
        word_list.insert(len(word_list) - 1, 'and')
        return ' '.join(word_list)

    def _director_of(self, director, movie):
        if isinstance(movie, str):
            movie = [movie]
        if len(director) == 1:
            return director[0] + ' is the director of movie ' + self._wrap_multi_movies(movie)
        return ''.join(director) + ' is the directors of movie ' + self._wrap_multi_movies(movie)

    def _writter_of(self, writter, movie):
        if isinstance(movie, str):
            movie = [movie]
        if len(writter) == 1:
            return writter[0] + ' is the writter of movie ' + self._wrap_multi_movies(movie)
        return ''.join(writter) + ' is the writters of movie ' + self._wrap_multi_movies(movie)

    def _actor_of(self, actor, movie):
        if isinstance(movie, str):
            movie = [movie]
        if len(actor) == 1:
            return actor[0] + ' is the actor/actress of movie ' + self._wrap_multi_movies(movie)
        return ''.join(actor) + ' is the actors/actresses of movie ' + self._wrap_multi_movies(movie)

    def _rate_of(self, rate, movie):
        if isinstance(rate, list):
            rate = rate[0]
        return self._wrap_movie_name(movie) + ' has a rating of ' + str(rate)

    def generate_answers(self, data):
        """
        data: a dict, refer to protocol.md
        """
        if data['intent'] == 'recommendation':
            # we need to recommend movies
            tmp = random.choice(self.template)
            return tmp[0] + ' ' + ' '.join([self._wrap_movie_name(x) for x in data['data']]) + tmp[1]
        # this is factoid
        question_list = data['origin']['data']
        answer_list = data['data']
        answer_list_str = []
        for i in range(len(answer_list)):
            q = question_list[i]
            a = answer_list[i]
            if q[1].endswith('?'):
                # this is a right/wrong question
                if isinstance(a[0], list):
                    answer = 'No, '
                else:
                    answer = 'Yes, you are right! '
                if a[1] == 'genre_of':
                    answer += self._genre_of(a[0], a[2])
                if a[1] == 'year_of':
                    answer += self._year_of(a[0], a[2])
                if a[1] == 'time_of':
                    answer += self._duration_of(a[0], a[2])
                if a[1] == 'director_of':
                    answer += self._director_of(a[0], a[2])
                if a[1] == 'act_in':
                    answer += self._actor_of(a[0], a[2])
                if a[1] == 'rate_of':
                    answer += self._rate_of(a[0], a[2])
                if a[1] == 'writter_of':
                    answer += self._writter_of(a[0], a[2])
                answer_list_str.append(answer)
            else:
                # answer question
                if a[1] == 'genre_of':
                    if q[0] == None or (isinstance(a[0], list) and q[0] == 'LAST?'):
                        answer = self._genre_of(a[0], a[2])
                    else:
                        if isinstance(a[2], str):
                            x = [a[2]]
                        answer = 'Movies ' + self._wrap_multi_movies(x) + ' is of type ' + a[0]
                if a[1] == 'writter_of':
                    answer = self._writter_of(a[0], a[2])
                elif a[1] == 'director_of':
                    answer = self._director_of(a[0], a[2])
                elif a[1] == 'act_in':
                    answer = self._actor_of(a[0], a[2])
                elif a[1] == 'rate_of':
                    if q[0] == None or (isinstance(a[0], list) and q[0] == 'LAST?'):
                        answer = self._rate_of(a[0], a[2])
                    else:
                        answer = 'Movies ' + self._wrap_multi_movies(a[2]) + ' is of rate ' + a[0]
                elif a[1] == 'year_of':
                    if q[0] == None or (isinstance(a[0], list) and q[0] == 'LAST?'):
                        answer = self._year_of(a[0], a[2])
                    else:
                        if isinstance(a[2], str):
                            x = [a[2]]
                        answer = 'Movies ' + self._wrap_multi_movies(x) + ' is of year ' + str(a[0])
                answer_list_str.append(answer)
        if len(answer_list_str) == 0:
            return 'Sorry, no suitable answer is found. Please try another question.'
        return '\n'.join(answer_list_str)

if __name__ == '__main__':
    nlg = nlg_subsystem()
    data = {
        'intent': 'recommendation',
        'data': ['Interstella', 'The Greate Wall', 'Star War II']
    }
    print(nlg.generate_answers(data))
    print(nlg.generate_answers(data))
    print(nlg.generate_answers(data))
    data = {
        'intent': 'factoid',
        'data': [(['p1', 'p2'], 'act_in', 'The Great Wall'), (['p2'], 'year_of', 'm1'), (['g1'], 'genre_of', 'm1'), ('g1', 'genre_of', ['m1'])],
        'origin': {
            'data': [(None, 'act_in', 'LAST'), ('p1', 'year_of?', 'm1'), ('LAST?', 'genre_of', 'LAST?'), ('LAST?', 'genre_of', 'LAST?')],
        }
    }
    print(nlg.generate_answers(data))