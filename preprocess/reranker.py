"""
1. shuffle: qidpidtriples_file > qidpidtriples_shuffled_file
2. For each line in qidpidtriples_shuffled_file
    Use relevant passage with probability '1/neg_per_pos'
    Use all irrelevant passage
3. convert ids to text and save to 'qidpidlabel.text.tsv'
"""


# from datasets import load_dataset
import os
import csv
import random

from tools.paths import *
from tools.read_files import read_texts
from tools.tools import shuffle_text

save_dir = os.path.join(reranker_dir, 'train')
save_path = os.path.join(save_dir, 'qidpidlabel.text.tsv')
max_size = 100000 * 128

# negative - positive ratio
neg_per_pos = 4
ratio = 1 / neg_per_pos


def preprocess_train():
    os.makedirs(save_dir, exist_ok=True)

    queries = read_texts(train_query_file)
    passages = read_texts(passage_file)

    write_file = open(save_path, 'w')

    count = 0
    with open(qidpidtriples_shuffled_file, 'r') as read_file:
        reader = csv.reader(read_file, delimiter='\t')
        for i, line in enumerate(reader):
            qid, pid1, pid2 = map(int, line)

            if random.random() < ratio:
                write_file.write('%s\t%s\t%d\n' % (queries[qid], passages[pid1], 1))
                count += 1

            if count == max_size:
                break

            write_file.write('%s\t%s\t%d\n' % (queries[qid], passages[pid2], 0))
            count += 1

            if count == max_size:
                break

    write_file.close()

    shuffle_text(save_path, save_path)

if __name__ == '__main__':
    preprocess_train()
