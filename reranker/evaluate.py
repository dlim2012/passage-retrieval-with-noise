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
from model import Reranker

# Checkpoint path
checkpoint_dir = 'checkpoints/reranker'
train_version = '0412_1'
checkpoint_name = 'epoch=0-steps=10250.ckpt'
checkpoint_path = os.path.join(checkpoint_dir, train_version, checkpoint_name)

# Batch size to use to encode vectors using GPU
batch_size = 24

# Path to the queries, passages, and top1000
dev_query_file = 'data/ms_marco/original/queries.dev.tsv'
passage_file = 'data/ms_marco/original/collection.tsv'
dev_top1000_file = 'data/ms_marco/preprocessed/top1000.dev.id.tsv'
dev_qrels_file = 'data/ms_marco/original/qrels.dev.tsv'

# Trying it with train data
###################################################################
#train_query_file = 'data/ms_marco/original/queries.train.tsv'
#train_qrels_file = 'data/ms_marco/original/qrels.train.tsv'
#train_top1000_file = 'data/ms_marco/preprocessed/top1000.train.id.tsv'
###################################################################

# tokenizer
tokenizer_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

# device
device = torch.device('cuda')


class qidpidlabelDataset(Dataset):
    """
    A Dataset class made to use PyTorch DataLoader
    """

    def __init__(self, inputs):
        self.data = inputs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(data):
    """
    Convert query, passage, and label to input_ids, attention_mask, and label
    """
    input_ids, attention_mask = [], []
    max_len = -1
    for line in data:
        query, passage, label = line

        # Truncate query to have at most 64 tokens
        query_tokens = tokenizer.tokenize(query)[:64]

        # Convert tokens of query to token ids
        query_token_ids = tokenizer.convert_tokens_to_ids(query_tokens)

        # Truncate passage so that [<CLS> query <SEP> passage <SEP>] doesn't exceed 512
        passage_tokens = tokenizer.tokenize(passage)[:509 - len(query_token_ids)]

        # Convert tokens of passage to token ids
        passage_token_ids = tokenizer.convert_tokens_to_ids(passage_tokens)

        # Token ids for input
        token_ids = [101] + query_token_ids + [102] + passage_token_ids + [102]

        # Append input ids, attention masks, and labels to lists
        input_ids.append(token_ids)
        attention_mask.append([1] * len(token_ids))

        # Track the maximum length for padding purpose
        max_len = max(max_len, len(token_ids))

    # Pad to the longest length
    for i in range(len(data)):
        input_ids[i] = input_ids[i] + [0] * (max_len - len(input_ids[i]))
        attention_mask[i] = attention_mask[i] + [0] * (max_len - len(attention_mask[i]))

    x = {'input_ids': torch.tensor(input_ids),
         'attention_mask': torch.tensor(attention_mask)}

    return x

def read_texts(text_file):
    # Read passages and queries
    texts = dict()
    with open(text_file, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        for i, line in enumerate(reader):
            text_id, text = int(line[0]), line[1]
            texts[text_id] = text
    return texts

def read_qrels(qrels_file):
    # Read qrels
    qrels = defaultdict(lambda: [])
    with open(qrels_file, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        for i, line in enumerate(reader):
            qid, pid = int(line[0]), int(line[2])
            qrels[qid].append(pid)
    return qrels

def read_top1000(top1000_file):
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
    return top1000

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

def main():
    model = Reranker.load_from_checkpoint(checkpoint_path).to(device).eval()
    print('model loaded from checkpoint %s' % checkpoint_path)

    # Read passages, queries, qrels and top 1000.
    passages = read_texts(passage_file)
    qrels = read_qrels(dev_qrels_file)
    top1000 = read_top1000(dev_top1000_file)

    ranks = []
    ranks_all = []

    query_files = [dev_query_file]
    for query_file in query_files:
        queries = read_texts(query_file)
        with tqdm(sorted(top1000.keys()), desc='queries') as pbar:

            for qid in pbar:

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

                # Append meaningless results and continue if there are no relevant passages
                if len(rel_indices) == 0:
                    ranks.append(1001)
                    continue


                # Use a dataloader for efficient computation
                dataset = qidpidlabelDataset([[queries[qid], passages[pid]] for pid in top1000_pids])
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=10, collate_fn=collate_fn)

                # Gather scores for each passage in the top 1000
                results = []
                for batch in dataloader:
                    x = {'input_ids': batch['input_ids'].to(device),
                         'attention_mask': batch['attention_mask'].to(device)}

                    # Calculate the <CLS> vectors for the top 1000 passages
                    scores = model.forward(x)[1].cpu().detach().numpy()

                    # Append the similarity scores for passages in a batch to a list
                    results.append(scores)

                # Get the similarity scores between query and its top 1000 passages in a 1-dimensional vector
                results = np.concatenate(results, axis=1)[0]

                # Argsort the similarity scores in descending order
                argsort = np.argsort(results)[::-1].tolist()

                # For each passage that are relevant to the query
                ranks_qid = []
                for idx in rel_indices:

                    # Find the rank
                    rank = argsort.index(idx) + 1
                    ranks_qid.append(rank)

                # Append results
                ranks.append(min(ranks_qid))
                ranks_all.append(ranks_qid)

                # Print imtermediate results in the progress bar
                postfix = {'mmr@e1': mrr(ranks, 10),
                           'mmr@e3': mrr(ranks, 1000),
                           'reca@10': recall(ranks_all, 10)
                }

                pbar.set_postfix(postfix)

            # Print out the measurements
            print('query file:', query_file)
            print('\tmrr@10: %.4f' % mrr(ranks, 10))
            print('\tmrr@1000: %.4f' % mrr(ranks, 1000))
            print('\trecall@10: %.4f' % recall(ranks_all, 10))


if __name__ == '__main__':
    main()
