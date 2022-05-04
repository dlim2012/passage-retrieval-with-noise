"""
Make separate training/validation data using shuffled version of qidpidtriples.
"""
import os
from tools.paths import *
from tools.read_files import read_texts

n_validation_relevant_pair = 5000
n_train_data = 4800000

train_file = os.path.join(write_dir, 'train.tsv')
validation_file = os.path.join(write_dir, 'validation.tsv')

train_writer = open(train_file, 'w')
validation_writer = open(validation_file, 'w')


queries = read_texts(train_query_file)
passages = read_texts(passage_file)


validation_pairs = set()
train_datapoint_count = 0
with open(qidpidtriples_shuffled_file, 'r') as read_file:

    for i, line in enumerate(read_file):
        qid, pid1, pid2 = map(int, line[:-1].split('\t'))

        if i < n_validation_relevant_pair:

            validation_pairs.add(qid)

            validation_writer.write('%s\t%s\t%s\n' % (queries[qid], passages[pid1], passages[pid2]))
        else:
            if not qid in validation_pairs:
                train_writer.write('%s\t%s\t%s\n' % (queries[qid], passages[pid1], passages[pid2]))
                train_datapoint_count += 1

                if train_datapoint_count == n_train_data:
                    break


