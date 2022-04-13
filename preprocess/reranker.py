# from datasets import load_dataset
import os
import csv
import random

# import from other files
from tools.paths import *
from tools.read_files import read_texts
from tools.tools import shuffle_text


# negative - positive ratio
neg_per_pos = 4
ratio = 1 / neg_per_pos

# save directory and the number of pairs to save
save_dir = os.path.join(reranker_dir, 'train')
save_path = os.path.join(save_dir, 'qidpidlabel.npr%d.text.tsv' % neg_per_pos)
max_size = 100000 * 128

def preprocess_train():
    """
    preprocess train
        1. shuffle: qidpidtriples_file > qidpidtriples_shuffled_file
        2. For each line in qidpidtriples_shuffled_file
            Use relevant passage with probability '1/neg_per_pos'
            Use all irrelevant passage
        3. convert ids to text and save to 'qidpidlabel.text.tsv'
    """
    # Make save directory if not exists
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. shuffle lines in the qidpidtriples file downloaded from the official website
    if not os.path.exists(qidpidtriples_shuffled_file):
        shuffle_text(qidpidtriples_file, qidpidtriples_shuffled_file)

    # Read queries and passages
    queries = read_texts(train_query_file)
    passages = read_texts(passage_file)

    # Open write file
    write_file = open(save_path, 'w')

    # write lines until the number of lines reaches the target number
    count = 0
    with open(qidpidtriples_shuffled_file, 'r') as read_file:
        reader = csv.reader(read_file, delimiter='\t')
        for i, line in enumerate(reader):
            qid, pid1, pid2 = map(int, line)
            
            # write a positive pair with probability '1/neg_per_pos'
            if random.random() < ratio:
                write_file.write('%s\t%s\t%d\n' % (queries[qid], passages[pid1], 1))
                count += 1

            if count == max_size:
                break

            # write a negative pair
            write_file.write('%s\t%s\t%d\n' % (queries[qid], passages[pid2], 0))
            count += 1

            if count == max_size:
                break

    write_file.close()

    # shuffle the written text
    shuffle_text(save_path, save_path)

if __name__ == '__main__':
    
    # preprocess train data in format 'query, tab, passage, tab, label)
    preprocess_train()
