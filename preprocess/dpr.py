"""
1. Shuffled data: qidpidtriples_file > qidpidtriples_shuffled_file
2. Filter qidpidtriples_shuffled_file so that negative passages are not relevant to any train query
3. convert ids to text and save to 'qidpidtriples.train.full.filtered.text.tsv'

"""

import csv
import os

from tools.paths import *
from tools.read_files import read_texts
from tools.tools import shuffle_text

save_dir = os.path.join(dpr_dir, 'train')
save_path = os.path.join(save_dir, 'qidpidtriples.train.full.filtered.text.tsv')
max_size = 200000 * 32


def preprocess_train():
    if not os.path.exists(qidpidtriples_shuffled_file):
        shuffle_text(qidpidtriples_file, qidpidtriples_shuffled_file)

    os.makedirs(save_dir, exist_ok=True)

    rel_ids = set()
    with open(train_qrels_file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for i, line in enumerate(reader):
            rel_ids.add(int(line[2]))

    queries = read_texts(train_query_file)
    passages = read_texts(passage_file)

    write_file = open(save_path, 'w')

    count = 0
    with open(qidpidtriples_shuffled_file, 'r') as read_file:
        reader = csv.reader(read_file, delimiter='\t')
        for i, line in enumerate(reader):
            negative_pid = int(line[2])
            if negative_pid in rel_ids:
                continue

            line = '\t'.join([queries[int(line[0])], passages[int(line[1])], passages[int(line[2])]]) + '\n'

            write_file.write(line)
            count += 1

            if count == max_size:
                break

    write_file.close()


if __name__ == '__main__':
    preprocess_train()

