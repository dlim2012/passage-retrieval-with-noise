"""
Make top1000.train.id.tsv, top1000.dev.id.tsv
    Each line is 'qid, pid1, pid2, ...' where pids are the top1000 bm25
Also make queries.train.top1000.tsv, queries.dev.top1000.tsv
    Each line is 'qid, query' where only queries that have data for top1000 are saved.
"""

from collections import defaultdict
import argparse

from tools.paths import *
from tools.read_files import read_texts


def parse():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_type', type=str, required=True) # "dev", "train"

    args = parser.parse_args()
    assert args.data_type in ['dev', 'train']
    return args

args = parse()

# Path to the top1000 files downloaded from MS Marco official site.
read_files = [top1000_train_file, top1000_dev_file]

# Path to the query files
query_files = [query_train_file, query_dev_file]

# Path to write the ids of the top 1000 pids retrieved based on BM25
write_files = [top1000_train_id_file, top1000_dev_id_file]

# Path to write information about queries that are used in top 1000 files
query_top1000_files = [query_train_top1000_file, query_dev_top1000_file]


if __name__ == '__main__':

    # Choose the type of the file
    if args.data_type == 'dev':
        i = 1
    elif args.data_type == 'train':
        i = 0
    read_file, query_file = read_files[i], query_files[i]
    write_file, query_top1000_file = write_files[i], query_top1000_files[i]

    # Get the query ids in the same order as in the query files
    queries = read_texts(query_file)

    # Print file names
    print("read_path:", read_file)
    print("write_path:", write_file)

    # Declare a dictionary (key: query_id, value: passage_ids in list)
    to_save = defaultdict(lambda: [])

    # Open file
    with open(read_file, 'r') as file:

        # Read each line and save label for each query_id in to_save
        for i, line in enumerate(file):
            qid, pid = map(int, line.split('\t')[:2])
            to_save[qid].append(pid)

    # Print total number of qids in the query file and the top1000 file.
    print('len(to_save.keys()):', len(to_save.keys()))
    print('len(qids)', len(queries.keys()))

    # Sae results in a tsv format
    with open(write_file, 'w') as file1:
        with open(query_top1000_file, 'w') as file2:
            for qid in to_save.keys():
                file1.write('\t'.join([str(qid)] + [str(pid) for pid in to_save[qid]]) + '\n')
                file2.write('\t'.join([str(qid), queries[qid]]) + '\n')

    print()
