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
    parser.add_argument('--ckpt_dir', type=str, default='/mnt/ssd/checkpoints/dpr')
    parser.add_argument('--ckpt_version', type=str, default='0421_1_lr3e-7_ts2e5')
    parser.add_argument('--ckpt_name', type=str, default='steps=94000_loss-interval=0.1027.pth')
    return parser.parse_args()

args = parse()

# Checkpoint path
checkpoint_path = os.path.join(args.ckpt_dir, args.ckpt_version, args.ckpt_name)

# Batch size to use to encode vectors using GPU
batch_size = 24

# Directories
read_dir = '/mnt/ssd/data/ms_marco/passages'
save_dir = os.path.join('/mnt/ssd/data/ms_marco/vectors/dpr', args.ckpt_version, args.ckpt_name[:-4])
os.makedirs(save_dir, exist_ok=True)

file_names= ['collection']
passage_files = [os.path.join(read_dir, file_name + '.tsv') for file_name in file_names]
save_files = [os.path.join(save_dir, file_name + '.npy') for file_name in file_names]

# tokenizer
tokenizer_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

# device
device = torch.device('cuda')


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

def read_texts(text_file):
    # Read passages and queries
    text_ids, texts = dict(), []
    with open(text_file, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        for i, line in enumerate(reader):
            #####################################################
            #if i == 10000:
            #    break
            #####################################################
            text_ids[int(line[0])] = i
            texts.append(line[1])
    return text_ids, texts

def calculate_dense_vectors(model, device, encoder, texts, tag):
    model = model.to(device)
    model.train(False)

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

    model.to('cpu')
    return torch.tensor(results)

def main():
    model = torch.load(checkpoint_path)
    model.train(False)
    print('model loaded from checkpoint %s' % checkpoint_path)

    for passage_file, save_file in zip(passage_files, save_files):
        # Read passages, queries, and qrels
        pids, passages = read_texts(passage_file)

        # Calculate passage vectors
        assert len(passages) == 8841823
        passage_vectors = calculate_dense_vectors(model, device, model.passage_encoder, passages, 'passages') # (N_p, d)

        np.save(save_file, passage_vectors)
        print('%s saved' % save_file)

if __name__ == '__main__':
    main()
