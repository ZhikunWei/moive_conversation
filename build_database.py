import rdflib
from rdflib import Graph, Literal, BNode, Namespace, URIRef
from rdflib import RDF, RDFS
import pickle

def buildGraph():
    g = Graph()
    url_prefix = "http://example.org/"
    my_namespace = Namespace("http://example.org/")
    uriDict = {}
    entityNum = 0
    relationNum = 0

    # concepts
    moiveID = URIRef(url_prefix+'moiveID')
    rating = URIRef(url_prefix+'rating')
    year = URIRef(url_prefix+'year')
    title = URIRef(url_prefix+'title')
    director = URIRef(url_prefix+'director')
    actor = URIRef(url_prefix+'actor')
    writer = URIRef(url_prefix+'writer')
    person = URIRef(url_prefix+'person')
    genres = URIRef(url_prefix+'genres')

    uriDict['moiveID'] = moiveID
    uriDict['rating'] = rating
    uriDict['year'] = year
    uriDict['title'] = title
    uriDict['director'] = director
    uriDict['actor'] = actor
    uriDict['writer'] = writer
    uriDict['person'] = person
    uriDict['genres'] = genres

    # each concept is a RDFS.Class
    g.add((moiveID, RDF.type, RDFS.Class))
    g.add((rating, RDF.type, RDFS.Class))
    g.add((year, RDF.type, RDFS.Class))
    g.add((title, RDF.type, RDFS.Class))
    g.add((director, RDF.type, RDFS.Class))
    g.add((actor, RDF.type, RDFS.Class))
    g.add((writer, RDF.type, RDFS.Class))
    g.add((person, RDF.type, RDFS.Class))
    g.add((genres, RDF.type, RDFS.Class))
    # concept label
    g.add((moiveID, RDFS.label, Literal('moiveID')))
    g.add((rating, RDFS.label, Literal('rating')))
    g.add((year, RDFS.label, Literal('year')))
    g.add((title, RDFS.label, Literal('title')))
    g.add((director, RDFS.label, Literal('director')))
    g.add((actor, RDFS.label, Literal('actor')))
    g.add((writer, RDFS.label, Literal('writer')))
    g.add((person, RDFS.label, Literal('person')))
    g.add((genres, RDFS.label, Literal('genres')))

    # property
    ratedBy = URIRef(url_prefix+'ratedBy')
    hasTitle = URIRef(url_prefix+'hasTitle')
    showInYear = URIRef(url_prefix+'showInYear')
    belongsToGenre = URIRef(url_prefix+'belongsToGenre')
    directedBy = URIRef(url_prefix+'directedBy')
    writedBy = URIRef(url_prefix+'writedBy')
    starredBy = URIRef(url_prefix+'starredBy')
    getId = URIRef(url_prefix+'getId')
    involve = URIRef(url_prefix+'involve')
    uriDict['ratedBy'] = ratedBy
    uriDict['hasTitle'] = hasTitle
    uriDict['showInYear'] = showInYear
    uriDict['belongsToGenre'] = belongsToGenre
    uriDict['directedBy'] = directedBy
    uriDict['writedBy'] = writedBy
    uriDict['starredBy'] = starredBy
    uriDict['getId'] = getId
    uriDict['involve'] = involve
    # each property is a RDFS.Property
    g.add((ratedBy, RDF.type, RDF.Property))
    g.add((hasTitle, RDF.type, RDF.Property))
    g.add((showInYear, RDF.type, RDF.Property))
    g.add((belongsToGenre, RDF.type, RDF.Property))
    g.add((directedBy, RDF.type, RDF.Property))
    g.add((writedBy, RDF.type, RDF.Property))
    g.add((starredBy, RDF.type, RDF.Property))
    g.add((getId, RDF.type, RDF.Property))
    g.add((involve, RDF.type, RDF.Property))
    # property lable
    g.add((ratedBy, RDFS.label, Literal('ratedBy')))
    g.add((hasTitle, RDFS.label, Literal('hasTitle')))
    g.add((showInYear, RDFS.label, Literal('showInYear')))
    g.add((belongsToGenre, RDFS.label, Literal('belongsToGenre')))
    g.add((directedBy, RDFS.label, Literal('directedBy')))
    g.add((writedBy, RDFS.label, Literal('writedBy')))
    g.add((starredBy, RDFS.label, Literal('starredBy')))
    g.add((getId, RDFS.label, Literal('getId')))
    g.add((involve, RDFS.label, Literal('involve')))
    
    # subclass
    g.add((director, RDFS.subClassOf, person))
    g.add((actor, RDFS.subClassOf, person))
    g.add((writer, RDFS.subClassOf, person))
    g.add((directedBy, RDFS.subClassOf, involve))
    g.add((starredBy, RDFS.subClassOf, involve))
    g.add((writedBy, RDFS.subClassOf, involve))

    #domain & range
    g.add((hasTitle, RDFS.domain, moiveID))
    g.add((ratedBy, RDFS.domain, title))
    g.add((showInYear, RDFS.domain, title))
    g.add((belongsToGenre, RDFS.domain, title))
    g.add((directedBy, RDFS.domain, title))
    g.add((writedBy, RDFS.domain, title))
    g.add((starredBy, RDFS.domain, title))
    g.add((getId, RDFS.domain, title))
    g.add((hasTitle, RDFS.range, title))
    g.add((ratedBy, RDFS.range, rating))
    g.add((showInYear, RDFS.range, year))
    g.add((belongsToGenre, RDFS.range, genres))
    g.add((directedBy, RDFS.range, director))
    g.add((writedBy, RDFS.range, writer))
    g.add((starredBy, RDFS.range, actor))
    g.add((getId, RDFS.range, moiveID))
    

    ins_genres = {}
    ins_directors = {}
    ins_writers = {}
    ins_actors = {}
    with open('./imdb_data/moive_dict.pkl', 'rb') as f:
        moive_dict = pickle.load(f)
        for k in moive_dict:
            if 'Episode' in moive_dict[k]['primaryTitle']:
                continue
            ins_moiveID = URIRef(url_prefix+'_'+k+'_moiveID')
            g.add((ins_moiveID, RDFS.label, Literal(k)))
            g.add((ins_moiveID, RDF.type, moiveID))

            ins_rating = URIRef(url_prefix+'_'+k+'_rating')
            g.add((ins_rating, RDFS.label, Literal(moive_dict[k]['rating'])))
            g.add((ins_rating, RDF.type, rating))
            g.add((ins_moiveID, ratedBy, ins_rating))

            
            ins_title = URIRef(url_prefix+'_'+k+'_title')
            g.add((ins_title, RDFS.label, Literal(moive_dict[k]['primaryTitle'])))
            g.add((ins_title, RDF.type, title))
            g.add((ins_moiveID, hasTitle, ins_title))

            ins_year = URIRef(url_prefix+'_'+k+'_year')
            if moive_dict[k]['startYear'] != '\\N':
                g.add((ins_year, RDFS.label, Literal(int(moive_dict[k]['startYear']))))
                g.add((ins_year, RDF.type, year))
                g.add((ins_moiveID, showInYear,ins_year))
            

            entityNum += 4
            relationNum += 11
            for gener_name in moive_dict[k]['genres'].split(','):
                if gener_name == r'\N':
                    continue
                if gener_name not in ins_genres:
                    ins_genres[gener_name] = URIRef(url_prefix+'_genres_'+gener_name.replace(' ', '_'))
                    g.add((ins_genres[gener_name], RDF.type, genres))
                    g.add((ins_genres[gener_name], RDFS.label, Literal(gener_name)))
                    entityNum += 1
                    relationNum += 2
                g.add((ins_moiveID, belongsToGenre, ins_genres[gener_name]))
                relationNum += 1


            for d in moive_dict[k]['directors']:
                if d not in ins_directors:
                    ins_directors[d] = URIRef(url_prefix+'_director_'+d.replace(' ', '_'))
                    g.add((ins_directors[d], RDF.type, director))
                    g.add((ins_directors[d], RDFS.label, Literal(d)))
                    entityNum += 1
                    relationNum += 2
                g.add((ins_moiveID, directedBy, ins_directors[d]))
                g.add((ins_moiveID, involve, ins_directors[d]))
                relationNum += 2
            
            for w in moive_dict[k]['writers']:
                if w not in ins_writers:
                    ins_writers[w] = URIRef(url_prefix+'_writer_'+w.replace(' ', '_'))
                    g.add((ins_writers[w], RDF.type, writer))
                    g.add((ins_writers[w], RDFS.label, Literal(w)))
                    entityNum += 1
                    relationNum += 2
                g.add((ins_moiveID, writedBy, ins_writers[w]))
                g.add((ins_moiveID, involve, ins_writers[w]))
                relationNum += 2
            
            if 'actors' in moive_dict[k]:
                for a in moive_dict[k]['actors']:
                    if a not in ins_actors:
                        ins_actors[a] = URIRef(url_prefix+'_actor_'+a.replace(' ', '_').replace('"', ''))
                        g.add((ins_actors[a], RDF.type, actor))
                        g.add((ins_actors[a], RDFS.label, Literal(a)))
                        entityNum += 1
                        relationNum += 2
                    g.add((ins_moiveID, starredBy, ins_actors[a]))
                    g.add((ins_moiveID, involve, ins_actors[a]))
                    relationNum += 2
    print('totle entity:', entityNum, 'total relation', relationNum)
    g.bind("ns", my_namespace)
    g.serialize("imdb_data/moiveDatabase.ttl", format= "xml")

if __name__ == '__main__':
    buildGraph()