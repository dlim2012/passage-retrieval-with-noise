
import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path

import csv
import argparse

import sys
csv.field_size_limit(sys.maxsize)

from preprocess.tools.paths import noisy_dir, noisy_passage_filenames
from preprocess.tools.paths import ckpt_dir, vectors_dir
from preprocess.tools.paths import passage_file
from preprocess.tools.read_files import read_texts_ids_to_idx

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='dpr')
    parser.add_argument('--ckpt_dir', type=str, default='default')
    parser.add_argument('--ckpt_version', type=str, default='test_dpr')
    parser.add_argument('--ckpt_name', type=str, default='steps=5000_loss-interval=0.1255.pth')
    parser.add_argument('--vectors_dir', type=str, default='default')
    parser.add_argument('--batch_size', type=int, default=22)

    parser.add_argument('--mode', type=str, default='original') # noisy

    args = parser.parse_args()

    assert args.model_name in ['dpr']
    assert args.mode in ['original', 'noisy']

    return args

args = parse()

if args.ckpt_dir != 'default':
    ckpt_dir = args.ckpt_dir
if args.vectors_dir != 'default':
    vectors_dir = args.vectors_dir

# Checkpoint path
checkpoint_path = os.path.join(ckpt_dir, 'dpr', args.ckpt_version, args.ckpt_name)


# Directories
save_dir = os.path.join(vectors_dir, args.ckpt_version, args.ckpt_name[:-4])
os.makedirs(save_dir, exist_ok=True)


def main():
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = torch.load(checkpoint_path)
    model.train(False)
    print('model loaded from checkpoint %s' % checkpoint_path)

    # Set the arguments that will be passed to the dataloader
    if args.model_name == 'dpr':
        dataloader_args = {'mode': 'precompute'}
    else:
        print("args.model_name should be in ['dpr']. ({} was given)".format(args.model_name))
        sys.exit(1)

    # Set the passage files and save paths according to the evaluation mode
    if args.mode == 'original':
        passage_files = [passage_file]
        save_files = [os.path.join(save_dir, 'collection.npy')]
    elif args.mode == 'noisy':
        passage_files = [os.path.join(noisy_dir, file_name + '.tsv') for file_name in noisy_passage_filenames]
        save_files = [os.path.join(save_dir, file_name + '.npy') for file_name in noisy_passage_filenames]
    else:
        print("args.mode should be in ['original', 'noisy']. ({} was given)".format(args.mode))
        sys.exit(1)

    # Repeat for each passage
    for passage_file_to_precompute, save_file in zip(passage_files, save_files):
        print('passage_file: %s' % passage_file_to_precompute)
        print('save_file: %s' % save_file)

        # Read passages, queries, and qrels
        pids, passages = read_texts_ids_to_idx(passage_file_to_precompute)

        # Calculate passage vectors
        assert len(passages) == 8841823

        # Set the model in evaluation mode in the device that will be used for computation
        model = model.to(device)
        model.train(False)

        # Calculate the dense vectors
        passage_vectors = model.dense_vectors(passages, args.batch_size, dataloader_args)

        # Save the results
        np.save(save_file, passage_vectors)
        print('%s saved' % save_file)

        # Delete variables before repetition for memory efficiency
        del pids, passages, passage_vectors

if __name__ == '__main__':
    main()
