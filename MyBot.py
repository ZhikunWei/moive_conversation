from rdflib import Graph
import pickle
import random

class MoiveConversationManager:
    def __init__(self, log = False):
        self.g = Graph()
        self.readGraph()
        with open('./imdb_data/moive_dict.pkl', 'rb') as f:
            self.moive_dict = pickle.load(f)
        self.curIntent = None
        self.curData = None
        self.lastAnswerMoive = None
        self.person2relation = {}
        self.resetHistory()
        self.log = log
        print('finish mcm init')
        
    def readGraph(self):
        print('loading grpha')
        self.g.parse('./imdb_data/moiveDatabase.ttl', format='xml')

    def getDataFromNLU(self, data):
        self.curData = data
        self.curQueryData = data
        self.curIntent = data['intent']

    def resetHistory(self):
        self.lastType = None
        self.lastAnswerPerson = None
        self.lastAnswerDirector = None
        self.lastAnswerWriter = None
        self.lastAnswerActor = None
        self.lastAnswerGenre = None
        self.lastAnswerYear = None
        self.lastAnswerRating = None

    def getQueryResult(self):
        result = {}
        if self.curIntent == 'recommendation':
            qrs = self.recommendQuery()
            queryResult = []
            for i in range(len(qrs)):
                queryResult.append(self.moive_dict[str(qrs[i])]['primaryTitle'])    
                self.lastAnswerMoive = queryResult[0]
                self.resetHistory()
                self.lastType = 'moive'
        elif self.curIntent == 'factoid':
            queryResult = self.factQuery()
            if len(queryResult) >= 1:
                queryResult = queryResult[-1:]
        
        result['intent'] = self.curIntent
        result['origin'] = self.curQueryData
        result['data'] = queryResult
        return result

    def factQuery(self):
        results = []
        for h,r,t in self.curData['data']:
            if self.log:
                print(self.lastType, self.lastAnswerMoive)
            if (t == 'LAST' ) and self.lastAnswerMoive != None:
                t = self.lastAnswerMoive
                if self.log:
                    print('replace t to', t)
            if t == 'LAST?' and h == 'LAST?':
                h = None
                t = None
                if self.lastType == 'moive' and self.lastAnswerMoive != None:
                    t = self.lastAnswerMoive
                elif self.lastType == 'director':
                    h = self.lastAnswerDirector
                elif self.lastType == 'actor':
                    h = self.lastAnswerActor
                elif self.lastType == 'writter':
                    h = self.lastAnswerWriter
                elif self.lastType == 'genre':
                    h = self.lastAnswerGenre
                elif self.lastType == 'year':
                    h = self.lastAnswerYear
            
            if r == 'person_of':
                r = self.lastAnswerPerson
            if self.log:
                print('before', h, r, t)
            if r == 'director_of':
                if h == 'LAST':
                    h = self.lastAnswerDirector
                if h == None :
                    q = 'select ?directorName where {?insTitle rdfs:label "%s". ?moive ns:hasTitle ?insTitle. ?moive ns:directedBy ?person. ?person rdfs:label ?directorName} LIMIT 1'%(t)
                    for x in list(self.g.query(q)):
                        results.append((str(x[0]),r,t))
                        self.lastAnswerPerson = r
                        self.lastAnswerDirector = str(x[0])
                        self.lastType = 'director'
                elif t == None:
                    q='select ?moiveName where {?insTitle rdfs:label ?moiveName. ?moive ns:hasTitle ?insTitle. ?moive ns:directedBy ?person. ?person rdfs:label "%s"} LIMIT 1'%(h)
                    for x in list(self.g.query(q)):
                        if str(x[0]) == self.lastAnswerMoive:
                            continue
                        results.append((h, r, str(x[0])))
                        self.lastAnswerMoive = str(x[0])
                        self.lastType = 'moive'
            elif r == 'act_in':
                if h == 'LAST':
                    h = self.lastAnswerActor
                if h == None:
                    q = 'select ?actorName where {?insTitle rdfs:label "%s". ?moive ns:hasTitle ?insTitle. ?moive ns:starredBy ?person. ?person rdfs:label ?actorName}  LIMIT 10'%(t)
                    for x in list(self.g.query(q)):
                        results.append((str(x[0]),r,t))
                        self.lastAnswerPerson = r
                        self.lastAnswerActor = str(x[0])
                        self.lastType = 'actor'
                elif t == None:
                    q='select ?moiveName where {?insTitle rdfs:label ?moiveName. ?moive ns:hasTitle ?insTitle. ?moive ns:starredBy ?person. ?person rdfs:label "%s"} LIMIT 10'%(h)
                    for x in list(self.g.query(q)):
                        if str(x[0]) == self.lastAnswerMoive:
                            continue
                        results.append((h, r, str(x[0])))
                        self.lastAnswerMoive = str(x[0])
                        self.lastType = 'moive'
            elif r == 'writter_of':
                if h == 'LAST':
                    h = self.lastAnswerWriter
                if h == None:
                    q = 'select ?writerName where {?insTitle rdfs:label "%s". ?moive ns:hasTitle ?insTitle. ?moive ns:writedBy ?person. ?person rdfs:label ?writerName} LIMIT 1'%(t)
                    for x in list(self.g.query(q)):
                        results.append((str(x[0]),r,t))
                        self.lastAnswerPerson = r
                        self.lastAnswerWriter = str(x[0])
                        self.lastType = 'writter'
                elif t == None:
                    q='select ?moiveName where {?insTitle rdfs:label ?moiveName. ?moive ns:hasTitle ?insTitle. ?moive ns:writedBy ?person. ?person rdfs:label "%s"} LIMIT 10'%(h)
                    for x in list(self.g.query(q)):
                        if str(x[0]) == self.lastAnswerMoive:
                            continue
                        results.append((h, r, str(x[0])))
                        self.lastAnswerMoive = str(x[0])
                        self.lastType = 'moive'
            elif r == 'rate_of':
                if h == 'LAST':
                    h = self.lastAnswerRating
                if h == None:
                    q = 'select ?rate where {?insTitle rdfs:label "%s". ?moive ns:hasTitle ?insTitle. ?moive ns:ratedBy ?ins_rate. ?ins_rate rdfs:label ?rate} LIMIT 1' % (t)
                    for x in list(self.g.query(q)):
                        results.append((float(x[0]),r,t))
                        self.lastAnswerRating = float(x[0])
                elif t == None:
                    q='select ?moiveName where {?insTitle rdfs:label ?moiveName. ?moive ns:hasTitle ?insTitle. ?moive ns:ratedBy ?ins. ?ins rdfs:label "%s"} LIMIT 10'%(h)
                    for x in list(self.g.query(q)):
                        if str(x[0]) == self.lastAnswerMoive:
                            continue
                        results.append((h, r, str(x[0])))
                        self.lastAnswerMoive = str(x[0])
                        self.lastType = 'moive'
            elif r == 'year_of':
                if h == 'LAST':
                    h = self.lastAnswerYear
                if h == None :
                    q = 'select ?year where {?insTitle rdfs:label "%s". ?moive ns:hasTitle ?insTitle. ?moive ns:showInYear ?ins. ?ins rdfs:label ?year} LIMIT 1'%(t)
                    for x in list(self.g.query(q)):
                        results.append((int(x[0]),r,t))
                        self.lastAnswerYear = int(x[0])
                        self.lastType = 'year'
                elif t == None:
                    q='select ?moiveName where {?insTitle rdfs:label ?moiveName. ?moive ns:hasTitle ?insTitle. ?moive ns:showInYear ?ins. ?ins rdfs:label ?year. FILTER(?year >= %d). FILTER (?year <= %d)} LIMIT 10' % (h, h)
                    for x in list(self.g.query(q)):
                        if str(x[0]) == self.lastAnswerMoive:
                            continue
                        results.append((h, r, str(x[0])))
                        self.lastAnswerMoive = str(x[0])
                        self.lastType = 'moive'
            elif r == 'genre_of':
                if h == 'LAST':
                    h = self.lastAnswerGenre
                if h == None:
                    q = 'select ?genre where {?insTitle rdfs:label "%s". ?moive ns:hasTitle ?insTitle. ?moive ns:belongsToGenre ?ins. ?ins rdfs:label ?genre} LIMIT 1'%(t)
                    for x in list(self.g.query(q)):
                        results.append((str(x[0]),r,t))
                        self.lastAnswerGenre = str(x[0])
                        self.lastType = 'genre'
                elif t == None:
                    q='select ?moiveName where {?insTitle rdfs:label ?moiveName. ?moive ns:hasTitle ?insTitle. ?moive ns:belongsToGenre ?ins. ?ins rdfs:label "%s"} LIMIT 10'%(h)
                    for x in list(self.g.query(q)):
                        if str(x[0]) == self.lastAnswerMoive:
                            continue
                        results.append((h, r, str(x[0])))
                        self.lastAnswerMoive = str(x[0])
                        self.lastType = 'moive'
            elif r == 'director_of?':
                res = []
                q = "select ?directorName where {?insTitle rdfs:label '%s'. ?moive ns:hasTitle ?insTitle. ?moive ns:directedBy ?person. ?person rdfs:label ?directorName} LIMIT 1"%(t)
                for x in list(self.g.query(q)):
                    res.append(str(x[0]))
                results.append((res, 'director_of', t))
            elif r == 'act_in?':
                res = []
                q = "select ?actorName where {?insTitle rdfs:label '%s'. ?moive ns:hasTitle ?insTitle. ?moive ns:starredBy ?person. ?person rdfs:label ?actorName} LIMIT 1"%(t)
                for x in list(self.g.query(q)):
                    res.append(str(x[0]))
                results.append((res, 'act_in', t))
            elif r == 'writter_of?':
                res = []
                q = "select ?writerName where {?insTitle rdfs:label '%s'. ?moive ns:hasTitle ?insTitle. ?moive ns:writedBy ?person. ?person rdfs:label ?writerName} LIMIT 1"%(t)
                for x in list(self.g.query(q)):
                    res.append(str(x[0]))
                results.append((res, 'writter_of', t))
            elif r == 'rate_of?':
                res = []
                q = "select ?rate where {?insTitle rdfs:label '%s'. ?moive ns:hasTitle ?insTitle. ?moive ns:ratedBy ?ins_rate. ?ins_rate rdfs:label ?rate} LIMIT 1"%(t)
                for x in list(self.g.query(q)):
                    res.append(str(x[0]))
                results.append((res, 'rate_of', t))
            elif r == 'year_of?':
                res = []
                q = "select ?year where {?insTitle rdfs:label '%s'. ?moive ns:hasTitle ?insTitle. ?moive ns:showInYear ?ins. ?ins rdfs:label ?year} LIMIT 1"%(t)
                for x in list(self.g.query(q)):
                    res.append(str(x[0]))
                results.append((res, 'year_of', t))
            elif r == 'genre_of?':
                res = []
                q = "select ?genre where {?insTitle rdfs:label '%s'. ?moive ns:hasTitle ?insTitle. ?moive ns:belongsToGenre ?ins. ?ins rdfs:label ?genre} LIMIT 1"%(t)
                for x in list(self.g.query(q)):
                    res.append(str(x[0]))
                results.append((res, 'genre_of', t))
            if self.log:
                print('after', h, r, t)

                
        return results

    def recommendQuery(self):
        hasCondition_person, candidate_moives_person = self.queryByPerson()
        hasCondition_year, candidata_moive_year = self.queryByYear()
        hasCondition_genre, candidate_moive_genre = self.queryByGenre()
        hasCondition_rating, candidate_moive_rating = self.queryByRating()
        candidate_moives = []
        candidate_moives = candidate_moives_person + candidata_moive_year + candidate_moive_genre + candidate_moive_rating
        if len(candidate_moives) > 0:
            candidate_moives = [random.choice(candidate_moives)]
        return candidate_moives
        
    def queryByPerson(self):
        hasCondition = True
        candidate_moives = []
        positivePersonList = self.curData['data']['like']['person']
        negativePersonList = self.curData['data']['dislike']['person']
        for person in positivePersonList:
            hasCondition = False
            q="select ?moiveID where {?moive rdfs:label ?moiveID. ?moive ns:involve ?person. ?person rdfs:label '%s'}  LIMIT 10" % (person)
            results = self.g.query(q)
            for x in list(results):
                candidate_moives.append(str(x[0]))
        for person in negativePersonList:
            q="select ?moiveID where {?moive rdfs:label ?moiveID. ?moive ns:involve ?person. ?person rdfs:label '%s'}  LIMIT 10" % (person)
            results = self.g.query(q)
            for x in list(results):
                if x[0] in candidate_moives:
                    candidate_moives.remove(str(x[0]))
        return hasCondition, candidate_moives

    def queryByYear(self):
        candidate_moives = []
        hasCondition = True
        for (min_year, max_year) in self.curData['data']['like']['year']:
            hasCondition = False
            q= "select ?moiveID where {?moive rdfs:label ?moiveID. ?moive ns:showInYear ?ins_year. ?ins_year rdfs:label ?year.\
             FILTER(?year >= %d). FILTER (?year <= %d) }  LIMIT 10" % (min_year, max_year)
            for x in list(self.g.query(q)):
                if x[0] not in candidate_moives:
                    candidate_moives.append(str(x[0]))
        for (min_year, max_year) in self.curData['data']['dislike']['year']:
            q= "select ?moiveID where {?moive rdfs:label ?moiveID. ?moive ns:showInYear ?ins_year. ?ins_year rdfs:label ?year.\
             FILTER(?year >= %d). FILTER (?year <= %d) }  LIMIT 10" % (min_year, max_year)
            for x in list(self.g.query(q)):
                if x[0] in candidate_moives:
                    candidate_moives.remove(str(x[0]))
        return hasCondition, candidate_moives
            
    def queryByGenre(self):
        candidate_moives = []
        hasCondition = True
        for g in self.curData['data']['like']['genre']:
            hasCondition = False
            q="select ?moiveID where {?moive rdfs:label ?moiveID. ?moive ns:belongsToGenre ?ins_genre. ?ins_genre rdfs:label '%s'}  LIMIT 10" % (g)
            # print(q)
            for x in list(self.g.query(q)):
                if x[0] not in candidate_moives:
                    candidate_moives.append(str(x[0]))
        for g in self.curData['data']['dislike']['genre']:
            q="select ?moiveID where {?moive rdfs:label ?moiveID. ?moive ns:belongsToGenre ?ins_genre. ?ins_genre rdfs:label '%s'} " % (g)
            for x in list(self.g.query(q)):
                if x[0] in candidate_moives:
                    if self.log:
                        print('remove', str(x[0]))
                    candidate_moives.remove(str(x[0]))
        return hasCondition, candidate_moives
    
    def queryByRating(self):
        candidate_moives = []
        hasCondition = True
        for (min_rating, max_rating) in self.curData['data']['like']['ratings']:
            hasCondition = False
            q= "select ?moiveID where {?moive rdfs:label ?moiveID. ?moive ns:ratedBy ?ins_rating. ?ins_rating rdfs:label ?rating.\
             FILTER(?rating >= %d). FILTER (?rating <= %d) } LIMIT 10" % (min_rating, max_rating)
            for x in list(self.g.query(q)):
                if x[0] not in candidate_moives:
                    candidate_moives.append(str(x[0]))
        for (min_rating, max_rating) in self.curData['data']['dislike']['ratings']:
            q= "select ?moiveID where {?moive rdfs:label ?moiveID. ?moive ns:ratedBy ?ins_rating. ?ins_rating rdfs:label ?rating.\
             FILTER(?rating >= %d). FILTER (?rating <= %d) } LIMIT 10" % (min_rating, max_rating)
            for x in list(self.g.query(q)):
                if x[0] in candidate_moives:
                    candidate_moives.remove(str(x[0]))
        return hasCondition, candidate_moives


if __name__ == '__main__':
    mcm = MoiveConversationManager()
    data = {'intent':'recommendation', 'data':{'like':{'person':[],'year':[],'genre':['Drama'],'rate':[],'time':[]},
    'dislike':{'person':[],'year':[],'genre':[],'rate':[],'time':[]}}}
    data = {'intent': 'factoid', 'data': [(None, 'writter_of', 'Illusionen')]}
    mcm.getDataFromNLU(data)
    result = mcm.getQueryResult()
    print(result)

