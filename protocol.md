## 框架
自然语言理解 --> 对话管理 --> 自然语言生成


## 数据格式定义
### 自然语言理解 -> 对话管理
传递数据格式：
```
{
    'intent': xxx,     // choosen from ['recommendation', 'factoid']
    'data': xxx         // given below
}
```

如果 'intent' 是 'recommendation'，则数据长这样：
```
{
    'intent': 'recommendation',
    'data': {
        'like': {
            'person': ['blabla', 'blabla', …],
            'year': [(min_year, max_year), (min_year, max_year), …],
            'genre': ['g2', 'g1', …],
            'ratings': [(min_rating, max_rating), (min_rating, max_rating), …],
            'time': [(min_time, max_time), (min_time, max_time), …]
        },
        'dislike': {
            'person': ['blabla', 'blabla', …],
            'year': [(min_year, max_year), (min_year, max_year), …],
            'genre': ['g2', 'g1', …],
            'ratings': [(min_rating, max_rating), (min_rating, max_rating), …],
            'time': [(min_time, max_time), (min_time, max_time), …]
        }
    }
}
```
查询规则暂定为 取满足所有条件的交集

如果'intent'是'factoid'，则数据长这样：
```
{
    'intent': 'factoid',
    'data': [(q1, r1, t1), (q2, r2, t2), …],
}
```

其中，q，t为头实体和尾实体，可以为人名、时间范围、年份范围、电影种类、评分范围等等（即上面intent为推荐时，like和dislike里面的那些元素）。r为关系对，目前三元组的形式为:
```
(NAME) director_of (MOVIE)
(NAME) act_in (MOVIE)
(NAME) writter_of (MOVIE)
(number) rate_of (MOVIE)
(year) year_of (MOVIE)
(type) genre_of (MOVIE)
```
查询问题分为两类，一类是查询问题，一类是正误问题。

- 查询问题的三元组中，未知的项被定义为None，如
`('John Walker', 'writter_of', None)`表示要查询John Walker的作品。None现在只会出现在头实体和尾实体中，并不会出现在关系上。

- 正误问题的三元组中，三元组各项都会给出，但是会在关系里缀上`?`，比如`('John Walker', 'writter_of?', 'Star War')`。

注：暂定用`'LAST'`来定义上文所指内容（一般指上次我们给出的回答）。比如`('LAST', 'writter_of', None)`表示查询上次回答时给出的作者的其他作品。此外，有时需要根据上次回答的具体内容来判断是头实体还是尾实体，此时头尾实体均为`'LAST?'`，比如`('LAST?', 'year_of', 'LAST?')`，根据上次的回答种类，如果是年份，则为查询电影名称，若为电影名，则查询为上映年份。

### 对话管理 -> 自然语言生成
数据格式
```
{
    'intent': xxx, // from ['recommendation', 'factoid']
    'data': xxx,
    'origin': xxx // this is what NLU passes to Daliog Manager
}
```
其中，为了debug方便以及其他用途，请将自然语言处理发给对话管理的数据放在对话管理返回数据的origin部分。

如果intent为recommendation，则如下：
```
{
    'intent': 'recommendation',
    'data': ['movie1', 'movie2', 'movie3', ...],
    'origin': xxx
}
```

如果intent为factoid，则如下：
```
{
    'intent': 'factoid',
    'data': [(q1, r1, t1), (q2, r2, t2), ...],
    'origin': xxx,
}
```
这里data里的三元组应该与'origin'中的'data'一一对应。对于查询三元组，给出结果即可（结果均用list表示，这是为了应对多结果的情况）。对于正误三元组，保持关系种类和电影部分不变，给出正确的头实体，若本身三元组正确。

注：凡涉及`'LAST'`或者`'LAST?'`的地方需要变成对应的实体。