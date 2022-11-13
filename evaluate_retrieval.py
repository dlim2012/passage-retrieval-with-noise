import csv
import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from tqdm import tqdm
from collections import defaultdict
import sys

import csv
import argparse

from preprocess.tools.read_files import read_texts, read_texts_ids_to_idx, read_qrels
from preprocess.tools.paths import ckpt_dir, vectors_dir
from preprocess.tools.paths import noisy_dir, noisy_queries_filenames, noisy_passage_filenames
from preprocess.tools.paths import query_dev_file, query_dev_top1000_file, passage_file, qrels_dev_file

def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, default='dpr')
    parser.add_argument('--ckpt_dir', type=str, default='default')
    parser.add_argument('--ckpt_version', type=str, default='test_dpr')
    parser.add_argument('--ckpt_name', type=str, default='steps=68000_loss-interval=0.1008.pth')
    parser.add_argument('--vectors_dir', type=str, default='../vectors')
    parser.add_argument('--batch_size', type=int, default=22)

    parser.add_argument('--mode', type=str, default='original') # noisy_queries, noisy_passages

    args = parser.parse_args()

    assert args.model_name in ['dpr']
    assert args.mode in ['original', 'noisy_queries', 'noisy_passages']

    return args



def main():
    args = parse()

    # Change directories if specified through arguments
    if args.ckpt_dir != 'default':
        ckpt_dir = args.ckpt_dir
    if args.vectors_dir != 'default':
        vectors_dir = args.vectors_dir

    # Checkpoint path
    checkpoint_path = os.path.join(ckpt_dir, args.model_name, args.ckpt_version, args.ckpt_name)
    passage_vectors_dir = os.path.join(vectors_dir, args.ckpt_version, args.ckpt_name[:-4])

    # Tokenizer
    tokenizer_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

    # Device
    device = torch.device('cuda')

    # Number of passages
    N_PASSAGES = 8841823
    
    # Load the model
    model = torch.load(checkpoint_path).to(device)
    model.train(False)
    print('model loaded from checkpoint %s' % checkpoint_path)

    # Read queries, qids, and qrels
    qids = list(read_texts_ids_to_idx(query_dev_top1000_file, only_ids=True).keys())
    pids, passages_list = read_texts_ids_to_idx(passage_file)
    qrels = read_qrels(qrels_dev_file)
    assert len(pids.keys()) == N_PASSAGES

    # Set the query files and passage vectors files according to the evaluation mode
    query_files = [query_dev_file]
    passage_vectors_files = [os.path.join(passage_vectors_dir, 'collection.npy')]
    assert args.mode in ['original', 'noisy_queries', 'noisy_passages']
    if args.mode == 'noisy_queries':
        query_files = [os.path.join(noisy_dir, query_filename) for query_filename in noisy_queries_filenames]
    elif args.mode == 'noisy_passages':
        passage_vectors_files = [os.path.join(passage_vectors_dir, vector_filename + '.npy') for vector_filename in noisy_passage_filenames]

    # For each passage_vectors file
    for passage_vectors_file in passage_vectors_files:

        # Read pids and precomputed passage vectors
        passage_vectors = torch.tensor(np.load(passage_vectors_file))
        assert passage_vectors.shape[0] == N_PASSAGES

        # For each query files
        for query_file in query_files:

            print('#############################################################')
            print('query file:', query_file)
            print('passage vector file:', passage_vectors_file)

            # Read the query file
            qids_ids_to_idx, queries_list = read_texts_ids_to_idx(query_file)
            queries = [queries_list[qids_ids_to_idx[qid]] for qid in qids]
            n_queries = len(queries)

            # Calculate query vectors
            assert len(queries) == 6980
            query_vectors = torch.tensor(model.dense_vectors(queries, args.batch_size, {'mode': 'precompute'})).cpu()

            # List and variable to save results
            recall_1000 = []  # for recall@1000
            reciprocal_ranks_mrr_10 = []
            no_relevant_count = 0

            pbar = tqdm(range(n_queries), desc='ranking')
            for i in pbar:
                print('-------------------------------------------------------------------------------------------------')

                # Calculate the scores between each query and each passage
                scores = torch.matmul(query_vectors[i:i+1], passage_vectors.T)[0]

                # Get the top 10 and top 1000 indices
                top10_indices = torch.topk(scores, 10).indices.tolist()
                top1000_indices = torch.topk(scores, 1000).indices.tolist()

                # Get the next qid and its relevant pids' indices in the passage vectors
                qid = qids[i]
                rel_indices = [(pids[pid], pid) for pid in qrels[qid] if pid in pids.keys()]

                # If no relevant indices: continue (There were no such cases among 6980 queries in the dev top 1000 set)
                if len(rel_indices) == 0:
                    no_relevant_count += 1
                    reciprocal_ranks_mrr_10.append(0)
                    continue

                # Get the rank of the pid if it is in top 1000 or top 10
                recall_1000_count = 0
                min_rank = sys.maxsize
                rank_from_1000 = -1
                for rel_index, rel_pid in rel_indices:
                    if rel_index in top1000_indices:
                        recall_1000_count += 1
                        rank_from_1000 = top1000_indices.index(rel_index) + 1
                    if rel_index in top10_indices:
                        try:
                            rank = top10_indices.index(rel_index) + 1
                        except:
                            continue
                        min_rank = min(min_rank, rank)

                    # Get the rank 1 passage
                    rank1_index = top1000_indices[0]

                    # Print rank, query, relevant passage, rank 1 passage for further analysis
                    print('[rank]: %d (-1 if not in top 1000)' % rank_from_1000)
                    print('[qid %d]:' % qid, queries_list[qids_ids_to_idx[qid]])
                    print('[rel_pid %d]:' % rel_pid, passages_list[pids[rel_pid]])
                    print('[rank1]:', passages_list[rank1_index])
                    print()

                # Append result to list to calculate MRR@10 and Recall@1000
                recall_1000.append(recall_1000_count / len(rel_indices))
                if min_rank <= 11:
                    reciprocal_ranks_mrr_10.append(1 / min_rank)
                else:
                    reciprocal_ranks_mrr_10.append(0)

                # Print intermediate results in the progress bar
                postfix = {'mrr': np.mean(reciprocal_ranks_mrr_10),
                           'recall': np.mean(recall_1000)
                           }

                pbar.set_postfix(postfix)

            # Print out the final results
            print('passage_vectors_file:', passage_vectors_file)
            print('query_file:', query_file)
            print('\trecall@1000:', np.mean(recall_1000))
            print('\tmrr@10:', np.mean(reciprocal_ranks_mrr_10))
            print('\tno_relevant_count: %d/%d' % (no_relevant_count, n_queries))

        # Delete vectors before repetition
        del query_vectors, passage_vectors


if __name__ == '__main__':
    main()
