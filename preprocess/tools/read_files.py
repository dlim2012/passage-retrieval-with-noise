from collections import defaultdict
import csv

def read_texts(path):
    """
    read queries and passages: query_train_file, query_dev_file, passage_file
    return: (dict) key(id), value(text)
    """
    result = dict()
    with open(path, 'r') as file:
        for i, line in enumerate(file):
            line = line.split('\t')
            result[int(line[0])] = line[1][:-1] # remove the line split
    return result

def read_texts_ids_to_idx(text_file, only_ids=False):
    # Read passages and queries
    if not only_ids:
        text_ids, texts = dict(), []
    else:
        text_ids = dict()

    with open(text_file, 'r') as file:
        for i, line in enumerate(file):
            line = line.split('\t')
            text_ids[int(line[0])] = i
            if not only_ids:
                texts.append(line[1])
    if not only_ids:
        return text_ids, texts
    return text_ids

def read_qrels(qrels_file):
    # Read qrels
    qrels = defaultdict(lambda: [])
    with open(qrels_file, 'r') as file:
        for i, line in enumerate(file):
            line = line.split('\t')
            qid, pid = int(line[0]), int(line[2])
            qrels[qid].append(pid)
    return qrels

def read_top1000(top1000_file):
    # Read top1000
    top1000 = dict()
    with open(top1000_file, 'r') as file:
        for i, line in enumerate(file):
            line = line.split('\t')
            line = list(map(int, line))
            #for j, text_id in enumerate(line):
                #if text_id == -1:
                #    continue # Only using top 1000
            top1000[line[0]] = line[1:]
    return top1000
