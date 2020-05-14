import pickle

def chose_moive():
    cnt = 0
    rated_movie = []
    with open('./imdb_data/title.ratings.tsv') as f:
        f.readline()
        for line in f:
            line = line.strip('\n').split('\t')
            tconst, averageRating, numVote = line
            # print(line)
            rated_movie.append([tconst,float(averageRating), int(numVote)])
            cnt += 1
            if cnt % 1000 == 0:
                print(line)
                print(cnt)
    rated_movie = sorted(rated_movie, key= lambda x: x[1], reverse=True)
    rated_movie = rated_movie[:50000]
    import pickle
    with open('./imdb_data/choosen_moive.pkl', 'wb') as f:
        pickle.dump(rated_movie, f)
    print(cnt)

def get_moive_dict():
    with open('./imdb_data/choosen_moive.pkl', 'rb') as f:
        moive = pickle.load(f)
    moive_dict = {}
    for m in moive:
        if m[0] in moive_dict:
            print('already exist!')
        moive_dict[m[0]] = {}
        moive_dict[m[0]]['rating'] = m[1]
        moive_dict[m[0]]['numVote'] = m[2]

    with open('./imdb_data/title.basics.tsv') as f:
        f.readline()
        for line in f:
            line = line.strip('\n').split('\t')
            if line[0] in moive_dict:
                moive_dict[line[0]]['titleType'] = line[1]
                moive_dict[line[0]]['primaryTitle'] = line[2]
                moive_dict[line[0]]['originalTitle'] = line[3]
                moive_dict[line[0]]['startYear'] = line[5]
                moive_dict[line[0]]['genres'] = line[8]

    nm2name = {}
    nm2tt = {}
    with open('./imdb_data/name.basics.tsv') as f:
        f.readline()
        for line in f:
            line = line.strip('\n').split('\t')
            nm2name[line[0]] = line[1]
            relatedMoives = line[-1].split(',')
            nm2tt[line[0]] = []
            for m in relatedMoives:
                nm2tt[line[0]].append(m)

    with open('./imdb_data/title.crew.tsv') as f:
        f.readline()
        for line in f:
            line = line.strip('\n').split('\t')
            if line[0] in moive_dict:
                directors = line[1].split(',')
                moive_dict[line[0]]['directors'] = []
                for dirc in directors:
                    if dirc != '\\N':
                        moive_dict[line[0]]['directors'].append(nm2name[dirc])
    
                writers = line[2].split(',')
                moive_dict[line[0]]['writers'] = []
                for writer in writers:
                    if writer != '\\N':
                        moive_dict[line[0]]['writers'].append(nm2name[writer])

    moive2actors = {}
    with open('./imdb_data/title.principals.tsv') as f:
        f.readline()
        for line in f:
            line = line.strip('\n').split('\t')
            if line[0] in moive_dict and (line[3] == 'actor' or line[3] == 'actress'):
                if line[0] not in moive2actors:
                    moive2actors[line[0]] = []
                moive2actors[line[0]].append(line[2])

    for m in moive2actors:
        if m in moive_dict:
            moive_dict[m]['actors'] = []
            for nm in moive2actors[m]:
                print(m, nm)
                if nm in nm2name:
                    moive_dict[m]['actors'].append(nm2name[nm])
                


    with open('./imdb_data/moive_dict.pkl', 'wb') as f:
        pickle.dump(moive_dict, f)
    
    
def checkFile():
    cnt  = []
    with open('./imdb_data/title.principals.tsv') as f:
        f.readline()
        for line in f:
            line = line.strip('\n').split('\t')
            if len(cnt) < 300:
                cnt.append(line)
            else:
                break
            if line[0] == '\\N':
                print(line)
    for c in cnt:
        print(c)

def check_moive_dict():
    with open('./imdb_data/moive_dict.pkl', 'rb') as f:
        moive_dict = pickle.load(f)

    for k, v in moive_dict.items():
        print(k, v)
    print(len(moive_dict))

if __name__ == "__main__":
    # chose_moive()
    # get_moive_dict()
    # checkFile()
    check_moive_dict()



    