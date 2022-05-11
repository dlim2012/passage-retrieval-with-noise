"""
Make separate training/validation data using shuffled version of qidpidtriples.
"""
import os
from tools.paths import *
from tools.read_files import read_texts
from tools.tools import shuffle_text
from collections import defaultdict
import csv

# number of qids to use for validation
n_validation_qid = 48000

# maximum number of validation data per qid
n_validation_data_per_qid = 2

# maximum total number of validation data to make
n_validation_data = n_validation_qid * n_validation_data_per_qid

# maximum total number of training data to make
n_train_data = 9600000


def main():
    # Shuffle 'qidpidtriples.train.full.tsv' again before making new datasets
    shuffle_text(qidpidtriples_file, qidpidtriples_shuffled_file)

    # Open files to write dataset
    train_writer = open(train_data_file, 'w')
    validation_writer = open(validation_data_file, 'w')

    # Read queries and passages as dictionary (key: text id, value: text)
    queries = read_texts(query_train_file)
    passages = read_texts(passage_file)

    # Read qrels train file and gather pids of all passages that are relevant to any pids
    rel_ids = set()
    with open(qrels_train_file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for i, line in enumerate(reader):
            rel_ids.add(int(line[2]))

    # Dictionary to count the number of
    validation_qids = defaultdict(lambda: 0)

    # Count of datasets for train and validation
    train_datapoint_count = 0
    validation_datapoint_count = 0

    # Open qidpidtriples file
    with open(qidpidtriples_shuffled_file, 'r') as read_file:

        for i, line in enumerate(read_file):

            # Stop if maximum number of datapoints are written to file
            if train_datapoint_count == n_train_data and validation_datapoint_count == n_validation_data:
                break

            # Read a triple
            qid, pid1, pid2 = map(int, line[:-1].split('\t'))

            # Write validation data when the maximum number of qids or maximum number of data point for qid is not met
            if validation_datapoint_count < n_validation_data:
                if validation_qids[qid] >=n_validation_data_per_qid:
                    pass
                elif len(validation_qids) < n_validation_qid or qid in validation_qids:
                    validation_qids[qid] += 1
                    validation_writer.write('%s\t%s\t%s\n' % (queries[qid], passages[pid1], passages[pid2]))
                    validation_datapoint_count += 1
                    continue

            # Write a train data if maximum number of train datapoint is not met
            if train_datapoint_count < n_train_data:

                # Exclude any pids that are related to any qids in the training set
                if pid2 in rel_ids:
                    continue

                train_writer.write('%s\t%s\t%s\n' % (queries[qid], passages[pid1], passages[pid2]))
                train_datapoint_count += 1

    # Close write files
    train_writer.close()
    validation_writer.close()


if __name__ == '__main__':
    main()

