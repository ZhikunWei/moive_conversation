import rdflib
from rdflib import Graph, Literal, BNode, Namespace, URIRef
from rdflib import RDF, RDFS

def load():
    g = Graph()
    g.parse('imdb_data/moiveDatabase.ttl', format='xml')
    print('finish load data')

    q = "select ?moive where {?moive ns:involve ?actor. ?actor rdfs:label 'Jane Asher'}"
    q2 =  "select ?titleLabel where { ?moiveTitle rdfs:label ?titleLabel. ?moive ns:hasTitle ?moiveTitle. ?moive ns:involve ?person. ?person rdfs:label 'Jane Asher'} "
    x = g.query(q2)
    for res in list(x):
        print(res)

    for r in list(x):
        r = r[0]
        print(r)

    # print(list(x))
    while True:
        try:
            q = input()
            x = g.query(q)
            for res in list(x):
                print(res)
        except Exception as err:
            print(err)

def read():
    import pickle
    with open('imdb_data/test_res.pkl', 'rb') as f:
        x = pickle.load(f)


if __name__ == '__main__':
    load()

    q= "select ?titleLabel ?year where {?moiveTitle rdfs:label ?titleLabel.?moive ns:hasTitle ?moiveTitle. ?moive ns:showInYear ?ins_year. ?ins_year rdfs:label ?year. FILTER (?year >= 1950). FILTER(?year <= 2000)}  "
    q="select ?titleLabel where {?moiveTitle rdfs:label ?titleLabel. ?moive ns:hasTitle ?moiveTitle. ?moive ns:belongsToGenre ?ins_genre. ?ins_genre rdfs:label 'Drama'}"
    q="select ?titleLabel where {?moiveTitle rdfs:label ?titleLabel. ?moive ns:hasTitle ?moiveTitle.  ?moive ns:ratedBy ?ins_rating. ?ins_rating rdfs:label ?rating. FILTER(?rating >= 9.5). FILTER(?rating <= 9.6)}   "
    q="select ?moiveID where {?moive rdfs:label ?moiveID. ?moive ns:belongsToGenre ?ins_genre. ?ins_genre rdfs:label 'Drama'} "
    q="select ?directorName where {?insTitle rdfs:label 'Star War'. ?moive ns:hasTitle ?insTitle. ?moive ns:directedBy ?person. ?person rdfs:label ?directorName}"