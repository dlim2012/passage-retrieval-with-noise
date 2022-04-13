import csv
import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from tqdm import tqdm
from collections import defaultdict

import csv

# Import the model
from model import DPR

# Checkpoint path
checkpoint_dir = 'checkpoints/dpr'
train_version = '0411_1'
checkpoint_name = 'steps=13100-loss_interval=1.2731e-01-acc=0.9544.ckpt'
checkpoint_path = os.path.join(checkpoint_dir, train_version, checkpoint_name)

# Batch size to use to encode vectors using GPU
batch_size = 24

# Path to the queries, passages, and top1000
query_file = 'data/ms_marco/original/queries.dev.tsv'
passage_file = 'data/ms_marco/original/collection.tsv'
top1000_file = 'data/ms_marco/preprocessed/top1000.dev.id.tsv'
qrels_file = 'data/ms_marco/original/qrels.dev.tsv'

# Trying it with train data
###################################################################
#query_file = 'data/ms_marco/original/queries.train.tsv'
#qrels_file = 'data/ms_marco/original/qrels.train.tsv'
#top1000_file = 'data/ms_marco/preprocessed/top1000.train.id.tsv'
###################################################################

# tokenizer
tokenizer_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

# device
device = torch.device('cuda')

class passage_dataset(Dataset):
    """
    A Dataset class made to use PyTorch DataLoader
    """
    def __init__(self, inputs):
        self.data = inputs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def tokenize(texts):
    """
    tokenize texts and return input ids and attention masks
    """
    inputs = tokenizer(texts, padding="longest", max_length=512, truncation=True)
    return torch.tensor(inputs["input_ids"]), torch.tensor(inputs["attention_mask"])


def collate_fn(lines):
    """
    Convert query, positive passage, and negative passage to input ids and attention masks
    """
    x = dict()
    x['passage_input_ids'], x['passage_attention_mask'] = tokenize(lines)
    return x

def mrr(ranks, Q):
    """
    Calculate the MRR(mean reciprocal rank) given the minimum ranks and Q
    """
    reciprocal_ranks = []
    for rank in ranks:
        if rank <= Q:
            reciprocal_ranks.append(1/rank)
    return np.mean(reciprocal_ranks)

def recall(ranks, k):
    count = 0
    for rank in ranks:
        if rank <= k:
            count += 1
    return count / len(ranks)

def main():
    model = DPR.load_from_checkpoint(checkpoint_path).to(device).eval()
    print('model loaded from checkpoint %s' % checkpoint_path)

    #passage_vectors = torch.tensor(np.load(passage_vectors_npy)) # shape: (N, d) N ~= 8 million

    # Read passage ids
    passages = dict()
    with open(passage_file, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        for i, line in enumerate(reader):
            pid, passage = int(line[0]), line[1]
            passages[pid] = passage

    # Read queries
    queries = dict()
    with open(query_file, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        for i, line in enumerate(reader):
            qid, query = int(line[0]), line[1]
            queries[qid] = query

    # Read qrels
    qrels = defaultdict(lambda: [])
    with open(qrels_file, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        for i, line in enumerate(reader):
            qid, pid = int(line[0]), int(line[2])
            qrels[qid].append(pid)

    # Read top1000
    top1000 = dict()
    with open(top1000_file, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        for i, line in enumerate(reader):
            line = list(map(int, line))
            #for j, text_id in enumerate(line):
                #if text_id == -1:
                #    continue # Only using top 1000
            top1000[line[0]] = line[1:]

    ranks = []
    reciprocal_ranks = []
    for qid in tqdm(sorted(top1000.keys()), desc='queries'):

        # Get the top 1000 pids for the query
        top1000_pids = top1000[qid]

        # Gather relevant pids
        rel_pids = []
        for pid in qrels[qid]:
            # If a relevant passage was not in the top 1000 passages, continue
            # Else, get the index of the relevant passage
            try:
                idx = top1000_pids.index(pid)
            except ValueError as e:
                print(e)
                continue
            rel_pids.append(pid)

        # Pass if there are no relevant passages
        if len(rel_pids) == 0:
            print("query with id %d doesn't have any relevant passages" % qid)
            continue

        # Get the query text and compute its <CLS> vector
        query = queries[qid]
        query_input_ids, query_attention_mask = tokenize([query])
        query_vector = model.query_encoder(
            input_ids=query_input_ids.to(device),
            attention_mask=query_attention_mask.to(device)
        ).last_hidden_state[:, 0, :]
        #print('query', query_input_ids.shape, query_attention_mask.shape)

        # Use a dataloader for efficient computation
        dataset = passage_dataset([passages[pid] for pid in top1000_pids])
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=10, collate_fn=collate_fn)

        results = []
        for batch in tqdm(dataloader, leave=False):

            # Calculate the <CLS> vectors for the top 1000 passages
            passage_vectors = model.passage_encoder(
                input_ids=batch['passage_input_ids'].to(device),
                attention_mask=batch['passage_attention_mask'].to(device)
            ).last_hidden_state[:, 0, :]
            #print('passage', batch['passage_input_ids'].shape, batch['passage_attention_mask'].shape)

            # Calculate the scores between query and each passage
            scores = torch.matmul(query_vector, passage_vectors.T).cpu().detach().numpy()

            # Append the similarity scores for passages in a batch to a list
            results.append(scores)

        # Get the similarity scores between query and its top 1000 passages in a vector with size (1000,)
        results = np.concatenate(results, axis=1)[0]

        # argsort the similarity scores in a descending order
        argsort = np.argsort(results)[::-1].tolist()

        # For each passage that are relevant to the query
        ranks_qid = []
        for pid in rel_pids:

            # Find the rank
            rank = argsort.index(idx) + 1
            ranks_qid.append(rank)
            print('\nrank: %d/%d' % (rank, len(top1000_pids)))

        # Append results
        ranks.append(min(ranks_qid))
        reciprocal_ranks.append(1/min(ranks_qid))

        # Print out the measurements
        k = 10
        print('\trecall: (%d: %.4f), (%d: %.4f), (%d: %.4f), (%d: %.4f)'\
              % (10, recall(ranks, 10), 20, recall(ranks, 20), 50, recall(ranks, 50), 100, recall(ranks, 100)))

        print('\tmrr: %.4f' % np.mean(reciprocal_ranks))


if __name__ == '__main__':
    main()
