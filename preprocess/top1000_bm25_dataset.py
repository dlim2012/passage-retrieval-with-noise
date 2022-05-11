"""
Make a new dataset using the top 1000 file ranked using BM25
Choose harder selections for hard negatives
"""
import csv
import random
from tqdm import tqdm

from tools.paths import *
from tools.read_files import read_texts, read_qrels
from tools.tools import shuffle_text


n_per_rel = 20

# Path to save the file
save_path = train_top1000_id_bm25_pair_file % n_per_rel

def main():

    # Get all pids that are relevant to any queries in the dataset
    rel_ids = set()
    with open(qrels_train_file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for i, line in enumerate(reader):
            rel_ids.add(int(line[2]))

    # Read queries and passages
    queries = read_texts(query_train_file)
    print('len(queries.keys())', len(queries.keys()))
    passages = read_texts(passage_file)
    print('len(passages.keys())', len(passages.keys()))
    qrels = read_qrels(qrels_train_file)
    print('len(qrels.keys())', len(qrels.keys()))

    # Count the lines in the top 1000 file to use it for tqdm
    n_lines = 0
    with open(top1000_train_id_bm25_file, 'r') as file:
        for line in file:
            n_lines += 1
    print(n_lines)

    # Open write file
    write_file = open(save_path, 'w')

    # Write lines until the number of lines reaches the target number
    count = 0
    with open(top1000_train_id_bm25_file, 'r') as read_file:
        reader = csv.reader(read_file, delimiter='\t')
        for i, line in tqdm(enumerate(reader), total=n_lines):
            qid = int(line[0])

            count_per_qid = 0

            positive_pids = qrels[qid] * n_per_rel
            random.shuffle(positive_pids)

            # For each of the pids sorted based on BM25 scores
            # If the pid is not relevant to any training qid add the pid to the dataset
            # Repeat until 20 pids are added
            for pid in line[1:]:
                negative_pid = int(pid)
                if negative_pid in rel_ids:
                    continue

                positive_pid = positive_pids[count_per_qid]

                to_write = '\t'.join([queries[qid], passages[positive_pid], passages[negative_pid]]) + '\n'
                write_file.write(to_write)

                count_per_qid += 1
                count += 1

                if count_per_qid + 1 == len(positive_pids):
                    break

    write_file.close()

    shuffle_text(top1000_train_id_bm25_file, top1000_train_id_bm25_file)

    print(count)

if __name__ == '__main__':
    main()
