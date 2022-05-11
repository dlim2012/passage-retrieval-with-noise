"""
Rerank the top 1000 bm25 retrieval results using bm25 to get more exact bm25 rankings
Warning: 60~64 GB of memory will be needed and the runtime is about 30 hours
References:
    # https://pypi.org/project/rank-bm25/
    # https://github.com/dorianbrown/rank_bm25/blob/master/rank_bm25.py
"""

import numpy as np
import csv
from rank_bm25 import BM25Okapi
from transformers import BertTokenizer
from tqdm import tqdm

from tools.paths import *


def main():
    # Get the tokenizing function of the BERT tokenizer
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    tokenize = tokenizer.tokenize

    # Read the passages
    ids_dict, passages = dict(), []
    with open(passage_file, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        for i, line in enumerate(reader):
            ids_dict[int(line[0])] = i
            passages.append(line[1])

    # Initialize the BM25 ranker
    bm25 = BM25Okapi(passages, tokenize)
    del passages
    print('bm25 initialized.')

    # Path to top 1000 id files
    top1000_files = [top1000_train_id_file, top1000_dev_id_file]

    # Path to queries for top 1000
    query_files = [query_train_top1000_file, query_dev_top1000_file]

    # Path to write the reranked results
    write_files = [top1000_train_id_bm25_file, top1000_dev_id_bm25_file]

    for i in reversed(range(2)):
        # Open files to read and write
        write_file = open(write_files[i], 'w')
        query_reader = csv.reader(open(query_files[i], 'r'), delimiter='\t')

        # Count the number of lines in top 1000 file to use as input to tqdm
        line_count = 0
        with open(top1000_files[i], 'r') as file:
            for line in file:
                line_count += 1

        with open(top1000_files[i], 'r') as file:
            reader = csv.reader(file, delimiter='\t')

            # Rerank for queries in each line of the top 1000 file
            for j, line in tqdm(enumerate(reader), total = line_count):
                # Get the next query information
                qid, query = next(query_reader)

                # Check that query id is as expected
                assert qid == line[0]

                # Get the top 1000 pids for the query
                pids = list(map(int, line[1:]))

                # Calculate the BM25 scores for the top 1000 pids
                scores = bm25.get_batch_scores(tokenize(query), [ids_dict[pid] for pid in pids])

                # Sort the pids
                pids = np.array(pids)[np.argsort(scores)[::-1]].tolist()

                # Write down the results
                write_file.write('\t'.join(map(str, [qid] + pids)) + '\n')

        write_file.close()
        print(write_files[i], 'done')

if __name__ == '__main__':
    main()