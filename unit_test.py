from sa_module_bert import sa_module_bert
from sentence_tokenizer import nlu_tokenizer

def test_sa_module():
    sa_module = sa_module_bert()
    tokenized = ['I', 'like', 'to', 'watch', 'GENRE', 'movies', ',', 'especially', 'those', 'TIME']
    info = {
        'movie': [],
        'genre': ['scary'],
        'time': [(300, 500)],
        'person': [],
        'rate': [],
        'year': []
    }
    print(sa_module.parse_sentence(tokenized, info))

    tokenized = ['I', 'do', 'n\'t', 'like', 'to', 'watch', 'MOV']#, ',', 'I', 'like', 'to', 'watch', 'movies', 'that', 'are', 'older', 'than', 'YEAR']
    info = {
        'movie': ['The Great Wall'],
        'genre': [],
        'time': [],
        'person': [],
        'rate': [],
        'year': []#[(1000, 2000)]
    }
    print(sa_module.parse_sentence(tokenized, info))


if __name__ == '__main__':
    test_sa_module()