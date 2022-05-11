import numpy as np
import torch
import os
from tqdm import tqdm
import sys

import argparse

from preprocess.tools.paths import query_dev_file, passage_file, qrels_dev_file, top1000_dev_id_file
from preprocess.tools.paths import noisy_queries_filenames, noisy_passage_filenames, noisy_dir
from preprocess.tools.read_files import read_texts, read_qrels, read_top1000
from models.tools.evaluation import mrr, recall


def parse():
    """
    parse arguments
    """
    parser = argparse.ArgumentParser()

    # (Required arguments) Model name
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=22)

    # Details about checkpoints
    # ckpt_dir: checkpoint base directory, ckpt_version: version name used at the training time
    # ckpt_name: checkpoint name
    parser.add_argument('--ckpt_dir', type=str, default='default')
    parser.add_argument('--ckpt_version', type=str, default='final_0_lr3e-7_ts2e5')
    parser.add_argument('--ckpt_name', type=str, default='steps=116000_loss-interval=0.2303.pth')

    # If mode is 'original': measure baseline model
    # If mode is 'noisy_queries': measure performance on noisy queries
    # If mode is 'noisy_passages': measure performance on noisy passages
    parser.add_argument('--mode', type=str, default='original')

    # Only for reranker: collate_fn2 truncates queries at size 64
    parser.add_argument('--collate_fn2', default=False, action='store_true')

    # Only for ColBERT (N_q: query token length, vector_size: vector size for each token)
    parser.add_argument('--N_q', type=int, default=32)
    parser.add_argument('--vector_size', type=int, default=128)

    args = parser.parse_args()

    assert args.model_name in ['reranker', 'dpr', 'colbert']
    assert args.mode in ['original', 'noisy_queries', 'noisy_passages']
    return args

args = parse()

if args.ckpt_dir != 'default':
    ckpt_dir = args.ckpt_dir

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Set dataloader arguments for each possible model
    if args.model_name == 'reranker':
        dataloader_args = {'use_collate_fn2': args.collate_fn2}
    elif args.model_name == 'dpr':
        dataloader_args = {'mode': 'rerank'}
    elif args.model_name == 'colbert':
        dataloader_args = {'N_q': args.N_q, 'mode': 'rerank'}
    else:
        print("Invalid argument: --model_name %s (not in ['reranker', 'dpr', 'colbert'])" % args.model_name)
        sys.exit(1)

    # Checkpoint path
    checkpoint_path = os.path.join(ckpt_dir, args.model_name, args.ckpt_version, args.ckpt_name)

    # Load model
    model = torch.load(checkpoint_path).to(device)
    model.train(False)
    print('model loaded from checkpoint %s' % checkpoint_path)

    # Read qrels and top 1000.
    qrels = read_qrels(qrels_dev_file)
    top1000 = read_top1000(top1000_dev_id_file)

    # Set the query files to evaluate based on the 'mode' argument
    query_files_to_evaluate = [query_dev_file]
    if args.mode == 'noisy_queries':
        query_files_to_evaluate = [os.path.join(noisy_dir, query_filename) for query_filename in noisy_queries_filenames]

    # Set the passage files to evaluate based on the 'mode' argument
    passage_files_to_evaluate = [passage_file]
    if args.mode == 'noisy_passages':
        passage_files_to_evaluate = [os.path.join(noisy_dir, passage_filename) for passage_filename in noisy_passage_filenames]

    # Repeat on all possible query file and passage file combinations
    for passage_file_to_evaluate in passage_files_to_evaluate:
        passages = read_texts(passage_file_to_evaluate)

        for query_file_to_evaluate in query_files_to_evaluate:
            queries = read_texts(query_file_to_evaluate)

            print('#############################################################')
            print('query file to evaluate:', query_file_to_evaluate)
            print('passage file to evaluate:', passage_file_to_evaluate)

            # A list to save minimum ranks of relevant passages for each query
            # If no relevant passage: save 1001
            min_ranks = []

            # A list to save all ranks for relevant passages
            ranks_all = []

            # Count number of queries that doesn't have relevant passages in the ~1000 candidates
            no_relevant_count = 0

            with tqdm(sorted(top1000.keys())) as pbar:

                for qid in pbar:
                    print('-------------------------------------------------------------------------------------------------')

                    # Get the top 1000 pids for the query
                    top1000_pids = top1000[qid]

                    # Gather indices of relevant pids in the top 1000
                    rel_indices = []
                    for pid in qrels[qid]:
                        # If a relevant passage was not in the top 1000 passages, continue
                        # Else, get the index of the relevant passage
                        try:
                            idx = top1000_pids.index(pid)
                        except ValueError:
                            continue
                        rel_indices.append(idx)

                    # Append 1001 and continue if there are no relevant passages
                    if len(rel_indices) == 0:
                        min_ranks.append(1001)
                        no_relevant_count += 1

                        # Print the ranking result for this query
                        print('qid: {}, ranks: {} / {}'.format(qid, [], len(top1000_pids)))

                        continue

                    # Calculate scores between the query and top 1000 passages
                    scores = model.scores_one_query(
                        query=queries[qid],
                        passages=[passages[pid] for pid in top1000_pids],
                        batch_size=args.batch_size,
                        dataloader_args=dataloader_args
                    )

                    # Argsort the similarity scores in descending order
                    argsort = np.argsort(scores)[::-1].tolist()

                    # For each passage that are relevant to the query
                    ranks_qid = []
                    for idx in rel_indices:

                        # Find the rank
                        rank = argsort.index(idx) + 1
                        ranks_qid.append(rank)

                        # Print the query, a relevant passage, rank of the relevant passage, and the rank 1 retrieved passage
                        print('[rank]:', rank)
                        print('[qid %d]:' % qid, queries[qid], end='')
                        print('[rel_pid %d]:' % pid, passages[pid], end='')
                        rank1_pid = top1000_pids[argsort[0]]
                        print('[rank1_pid %d]:' % rank1_pid, passages[rank1_pid])
                        print()

                    # Print the ranking result for this query
                    print('qid: {}, ranks: {} / {}'.format(qid, ranks_qid, len(top1000_pids)))

                    # Append results
                    min_ranks.append(min(ranks_qid))
                    ranks_all.append(ranks_qid)

                    # Print imtermediate results in the progress bar
                    postfix = {'mrr@10': mrr(min_ranks, 10),
                               'mrr@1000': mrr(min_ranks, 1000),
                               'recall@10': recall(ranks_all, 10)
                    }

                    pbar.set_postfix(postfix)

                # Print out the measurements
                print('checkpoint path:', checkpoint_path)
                print('\tqids with no relevant in top1000:', no_relevant_count)
                print('\tmrr@10: %.4f' % mrr(min_ranks, 10))
                print('\tmrr@1000: %.4f' % mrr(min_ranks, 1000))
                print('\trecall@10: %.4f' % recall(ranks_all, 10))


if __name__ == '__main__':
    main()
