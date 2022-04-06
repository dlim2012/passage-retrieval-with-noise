import numpy as np
import csv

def read_texts(path):
    """
    read queries and passages:
        "data/ms_marco/original/queries.train.tsv"
        "data/ms_marco/original/queries.dev.tsv"
        "data/ms_marco/original/collection.tsv"
    return: (dict) key(id), value(text)
    """
    result = dict()
    with open(path, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        for line in reader:
            result[int(line[0])] = line[1]
    return result

def read_qidpidtriples(path, shuffle=False):
    """
    read qidpid triples:
        "data/ms_marco/original/qidpidtriples.train.full.tsv"
        "data/ms_marco/preprocessed/qidpidtriples.train.full.shuffled.tsv"
    return: (list) each row is [qid, relevant pid, not relevant pid]
    """
    result = []
    
    with open(path, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        for line in reader:
            result.append((int(text_id) for text_id in line))
               
    if shuffle:
        random.shuffle(result)
                
    return result

def read_top1000_npy(path):
    """
    read top1000 npy files:
        "data/ms_marco/preprocessed/top1000.train.id.all.npy" -> Use this one.
        "data/ms_marco/preprocessed/top1000.dev.id.all.npy" -> Use this one.
        "data/ms_marco/preprocessed/top1000.train.id.npy" -> probably not needed.
        "data/ms_marco/preprocessed/top1000.dev.id.npy" -> probably not needed.
        
    return: (dict) key(qid), value([top1000 pids with -1 for padding])
    
    About 90-95% queries seem to have exactly 1000 pids
    There could be less than or more than 1000 pids for each query
    
    """
    result = dict()
    
    read = np.load(path)
    
    for line in read:
        result[line[0]] = line[1:]
    return result

    
def read_qrels(path):
    """
    read qrel files:
        "data/ms_marco/original/qrels.train.csv"
        "data/ms_marco/original/qrels.dev.csv"
    return: (dict) key(qid), value([relevant pids])
    
    There could be no or multiple relevant pids for each query
    """
    result = dict()
    with open(path, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        for line in reader:
            qid, pid = int(line[0]), int(line[2])
            
            if qid in result.keys():
                result[qid].append(pid)
            else:
                result[qid] = [pid]
    return result
