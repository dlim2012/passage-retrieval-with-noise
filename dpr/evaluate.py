import csv
import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from tqdm import tqdm
from collections import defaultdict

import csv
import argparse


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vectors_dir', type=str, default='/mnt/ssd/data/ms_marco/vectors/dpr')
    parser.add_argument('--ckpt_dir', type=str, default='/mnt/ssd/checkpoints/dpr')
    parser.add_argument('--ckpt_version', type=str, default='0421_1_lr3e-7_ts2e5')
    parser.add_argument('--ckpt_name', type=str, default='steps=94000_loss-interval=0.1027.pth')

    parser.add_argument('--noisy_text_type', type=str, default='queries') # queries / passages
    return parser.parse_args()

args = parse()

assert args.noisy_text_type in ['queries', 'passages']

# Checkpoint path
checkpoint_path = os.path.join(args.ckpt_dir, args.ckpt_version, args.ckpt_name)

# Batch size to use to encode vectors using GPU
batch_size = 24

# Path to the queries, passages, and top1000
passage_file = '/mnt/ssd/data/ms_marco/passages/collection.tsv'
query_file_0 = '/mnt/ssd/data/ms_marco/preprocessed/queries.dev.top1000.tsv'
dev_qrels_file = '/mnt/ssd/data/ms_marco/original/qrels.dev.tsv'

#
passage_dir = '/mnt/ssd/data/ms_marco/passages'
passage_vectors_dir = os.path.join(args.vectors_dir, args.ckpt_version, args.ckpt_name[:-4])
passage_names = ['collection']
if args.noisy_text_type == 'passages':
    passage_vector_files = [os.path.join(passage_vectors_dir, name + '.npy') for name in passage_names]
else:
    passage_vector_files = [os.path.join(passage_vectors_dir, 'collection.npy')]

query_dir = '/mnt/ssd/data/ms_marco/queries'
query_names = ['queries.dev.top1000']
if args.noisy_text_type == 'queries':
    query_files = [os.path.join(query_dir, name + '.tsv') for name in query_names]
else:
    query_files = [os.path.join(query_dir, 'queries.dev.top1000.tsv')]


# tokenizer
tokenizer_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

# device
device = torch.device('cuda')

#
N_PASSAGES = 8841823

class text_dataset(Dataset):
    """
    A Dataset class made to use PyTorch DataLoader
    """
    def __init__(self, inputs):
        self.data = inputs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(lines):
    """
    Convert query, positive passage, and negative passage to input_types
    """
    return tokenizer(lines, padding="longest", max_length=512, truncation=True, return_tensors='pt')

def read_texts(text_file, only_ids=False):
    # Read passages and queries
    if not only_ids:
        text_ids, texts = dict(), []
    else:
        text_ids = dict()

    with open(text_file, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        for i, line in enumerate(reader):
            text_ids[int(line[0])] = i
            if not only_ids:
                texts.append(line[1])
    if not only_ids:
        return text_ids, texts
    return text_ids


def read_qrels(qrels_file):
    # Read qrels
    qrels = defaultdict(lambda: [])
    with open(qrels_file, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        for i, line in enumerate(reader):
            qid, pid = int(line[0]), int(line[2])
            qrels[qid].append(pid)
    return qrels


def mrr(ranks, Q=1000):
    """
    Calculate the MRR(mean reciprocal rank) given the minimum ranks and Q
    """
    reciprocal_ranks_Q = []
    for rank in ranks:
        if rank <= Q:
            reciprocal_ranks_Q.append(1/rank)
        else:
            reciprocal_ranks_Q.append(0)
    return np.mean(reciprocal_ranks_Q)

def recall(ranks_all, k=10):
    results = []
    for ranks in ranks_all:
        if len(ranks) == 0:
            continue
        count = 0
        for rank in ranks:
            if rank <= k:
                count += 1
        results.append(count / len(ranks))
    return np.mean(results)

def calculate_dense_vectors(encoder, texts, tag):

    dataset = text_dataset(texts)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=10, collate_fn=collate_fn)

    results = []
    for batch in tqdm(dataloader, desc=tag):
        batch = {key: value.to(device) for key, value in batch.items()}

        # Calculate the <CLS> vectors for the top 1000 passages
        vectors = encoder(**batch).last_hidden_state[:, 0, :].cpu().detach().numpy().astype(np.float32)

        results.append(vectors)

    # Get the similarity scores between query and its top 1000 passages in a 1-dimensional vector
    results = np.concatenate(results, axis=0)

    return results

def calculate_scores(query_vectors, passage_vectors):
    return torch.matmul(query_vectors, passage_vectors.T).cpu()

def main():
    model = torch.load(checkpoint_path).to(device)
    model.train(False)
    print('model loaded from checkpoint %s' % checkpoint_path)

    pids = read_texts(passage_file, only_ids=True)
    qrels = read_qrels(dev_qrels_file)
    assert len(pids.keys()) == N_PASSAGES

    for passage_vector_file in passage_vector_files:

        # Read pids and precomputed passage vectors
        passage_vectors = np.load(passage_vector_file)
        assert passage_vectors.shape[0] == N_PASSAGES

        # For each query files
        for query_file in query_files:
            # Read the query file
            qids, queries = read_texts(query_file)
            n_queries = len(queries)

            # Calculate query vectors
            assert len(queries) == 6980
            query_vectors = calculate_dense_vectors(model.query_encoder, queries, 'queries')  # (N_q, d)
            del queries

            qid_list = list(qids.keys())

            recall_1000 = []  # for recall@1000
            reciprocal_ranks_mrr_10 = []
            no_relevant_count = 0


            pbar = tqdm(range(n_queries), desc='ranking')
            for i in pbar:

                # Calculate the scores between each query and each passage
                scores = torch.tensor(np.dot(query_vectors[i:i+1], passage_vectors.T)[0])

                # Get the top 10 and top 1000 indices
                top10_indices = torch.topk(scores, 10).indices.tolist()
                top1000_indices = torch.topk(scores, 1000).indices.tolist()

                qid = qid_list[i]
                rel_indices = [pids[pid] for pid in qrels[qid] if pid in pids.keys()]

                if len(rel_indices) == 0:
                    no_relevant_count += 1
                    reciprocal_ranks_mrr_10.append(0)
                    continue

                recall_1000_count = 0
                min_rank = 11
                for rel_index in rel_indices:
                    if rel_index in top1000_indices:
                        recall_1000_count += 1
                    if rel_index in top10_indices:
                        try:
                            rank = top10_indices.index(rel_index) + 1
                        except:
                            continue
                        min_rank = min(min_rank, rank)


                recall_1000.append(recall_1000_count / len(rel_indices))
                if min_rank != 11:
                    reciprocal_ranks_mrr_10.append(1 / min_rank)
                else:
                    reciprocal_ranks_mrr_10.append(0)

                # Print imtermediate results in the progress bar
                postfix = {'mrr': np.mean(reciprocal_ranks_mrr_10),
                           'recall': np.mean(recall_1000)
                           }

                pbar.set_postfix(postfix)

            print('passage_vector_file:', passage_vector_file)
            print('query_file:', query_file)
            print('\trecall@1000:', np.mean(recall_1000))
            print('\tmrr@10:', np.mean(reciprocal_ranks_mrr_10))
            print('\tno_relevant_count: %d/%d' % (no_relevant_count, n_queries))


if __name__ == '__main__':
    main()
