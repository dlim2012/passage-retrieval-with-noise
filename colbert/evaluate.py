import csv
import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from tqdm import tqdm
from collections import defaultdict
import string

import csv

# Import the model
from model import ColBERT

# model, tokenizer configuration
model_name = 'bert-base-uncased'
#tokenizer = BertTokenizer.from_pretrained(model_name)
#tokenizer.add_special_tokens({'additional_special_tokens': ['[Q]', '[D]']})

#tokenizer = model.tokenizer
#Q_token_id, D_token_id = tokenizer.convert_tokens_to_ids(['[Q]', '[D]'])
#punctuation_ids = set(tokenizer.convert_tokens_to_ids([ch for ch in string.punctuation]))

# Checkpoint path
checkpoint_dir = 'checkpoints/colbert'
train_version = 'default'
checkpoint_name = 'steps=22500-loss_interval=9.785e-02-acc=0.960.ckpt'
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
#tokenizer_name = 'bert-base-uncased'
#tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

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

def tokenize(texts, special_token_id):
    n = len(texts)
    inputs = tokenizer(texts, padding="longest", max_length=511, truncation=True)
    input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]
    punctuation_mask = []
    for i in range(n):
        input_ids[i] = [input_ids[i][0]] + [special_token_id] + input_ids[i][1:]
        attention_mask[i] = [1] + attention_mask[i]
        if special_token_id == Q_token_id:
            continue
        punctuation_mask.append([])
        for j in range(len(input_ids[0])):
            if input_ids[i][j] == 0:
                break
            elif input_ids[i][j] in punctuation_ids:
                continue
            else:
                punctuation_mask[-1].append(j)
    return torch.tensor(input_ids), torch.tensor(attention_mask), punctuation_mask


def collate_fn(lines):
    """
    Convert query, positive passage, and negative passage to input ids and attention masks
    """
    x = dict()
    x['passage_input_ids'], x['passage_attention_mask'], x['punc_mask'] = tokenize(lines, D_token_id)
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
    model = ColBERT.load_from_checkpoint(checkpoint_path).to(device).eval()
    print('model loaded from checkpoint %s' % checkpoint_path)

    global tokenizer
    tokenizer = model.tokenizer
    global Q_token_id, D_token_id
    Q_token_id, D_token_id = tokenizer.convert_tokens_to_ids(['[Q]', '[D]'])
    global punctuation_ids
    punctuation_ids = set(tokenizer.convert_tokens_to_ids([ch for ch in string.punctuation]))

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

                # Get the query text and compute its <CLS> vector
                query = queries[qid]
                query_input_ids, query_attention_mask, _ = tokenize([query], Q_token_id)
                query_vector = model.encoder(
                    input_ids=query_input_ids.to(device),
                    attention_mask=query_attention_mask.to(device)
                ).last_hidden_state

                # Use a dataloader for efficient computation
                dataset = passage_dataset([passages[pid] for pid in top1000_pids])
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=10, collate_fn=collate_fn)

                # Gather scores for each passage in the top 1000
                results = []
                for batch in dataloader:

                    # Calculate the <CLS> vectors for the top 1000 passages
                    passage_vectors = model.encoder(
                        input_ids=batch['passage_input_ids'].to(device),
                        attention_mask=batch['passage_attention_mask'].to(device)
                    ).last_hidden_state

                    # Calculate the scores between query and each passage
                    scores = model.late_interaction(query_vector, passage_vectors, batch['punc_mask']).detach().numpy()

                    # Append the similarity scores for passages in a batch to a list
                    results.append(scores)

                    #del batch, passage_vectors

                    #torch.cuda.empty_cache()

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
