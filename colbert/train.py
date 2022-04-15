"""
v7: no linear
v8: removed lr schedule, removed a for loop in loss, using punc_mask, batch size 24
v9:
todo: add linear
"""
import string

from torch.utils.data import Dataset, DataLoader
import torch
from model import ColBERT
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import pytorch_lightning as pl
from transformers import BertTokenizer
import csv
import os
import argparse

def parse():
    parser = argparse.ArgumentParser()
    # learning rate
    parser.add_argument('--lr', type=float, default=3e-6)
    parser.add_argument('--lr_schedule', type=bool, default=False)

    # good to use this argument in every training step
    parser.add_argument('--ckpt_dirname',  type=str, default='no_name')

    # other arguments
    parser.add_argument('--vector_size', type=float, default=128)
    parser.add_argument('--early_stop', type=bool, default=True)
    parser.add_argument('--measure_steps', type=int, default=250)
    parser.add_argument('--batch_size', type=float, default=16)

    return parser.parse_args()


qidpidtriples_file = 'data/ms_marco/preprocessed/dpr/train/qidpidtriples.train.full.filtered.text.tsv'

# model, tokenizer configuration
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({'additional_special_tokens': ['[Q]', '[D]']})
Q_token_id, D_token_id = tokenizer.convert_tokens_to_ids(['[Q]', '[D]'])
punctuation_ids = set(tokenizer.convert_tokens_to_ids([ch for ch in string.punctuation]))

# Parse arguments
args = parse()

# Number of training instances and linear scheduler_steps
n_train_triples = args.batch_size * 200000
if not args.lr_schedule:
    linear_scheduler_steps = None
else:
    linear_scheduler_steps = (n_train_triples // (args.batch_size * 10),
                              n_train_triples // args.batch_size + 1)


# Directory to save checkpoints
checkpoint_dir = os.path.join('checkpoints/colbert/', args.ckpt_dirname)

# Directory to save logs
log_dir = "log/colbert/"
os.makedirs(log_dir, exist_ok=True)


class TriplesDataset(Dataset):
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

def collate_fn(data):
    x = dict()
    x['query_input_ids'], x['query_attention_mask'], _ = \
        tokenize([line[0] for line in data], Q_token_id)
    x['positive_input_ids'], x['positive_attention_mask'], x['positive_punc_mask'] = \
        tokenize([line[1] for line in data], D_token_id)
    x['negative_input_ids'], x['negative_attention_mask'], x['negative_punc_mask'] = \
        tokenize([line[2] for line in data], D_token_id)
    return x

def get_dataloader():
    lines = []
    with open(qidpidtriples_file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for i, line in enumerate(reader):
            if i == n_train_triples:
                break
            lines.append(line)
    dataset = TriplesDataset(lines)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, collate_fn=collate_fn)
    return dataloader

def main():

    # Declare a model
    model = ColBERT(
        linear_scheduler_steps=linear_scheduler_steps,
        B=args.batch_size,
        tokenizer=tokenizer,
        lr=args.lr,
        vector_size=args.vector_size,
        measure_steps=args.measure_steps,
        model_name=model_name,
        early_stop=args.early_stop
    )

    # Use a TensorBoardLogger
    logger = pl.loggers.TensorBoardLogger(save_dir="log/dpr/")

    # Save checkpoint: three lowest loss_interval
    regular_checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="{steps:.0f}-{loss_interval:.3e}-{acc:.3f}",
        monitor="loss_interval",
        mode='min',
        every_n_train_steps=args.measure_steps,
        save_top_k=3,
    )

    # Save checkpoint after every epoch
    epoch_checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="{epoch}-end",
        monitor="steps",
    )

    # Log learning rate
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Train the model
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=1,
        logger=logger,
        callbacks=[regular_checkpoint, epoch_checkpoint, lr_monitor]
    )

    # Get dataloader
    train_dataloader = get_dataloader()

    # Train the model
    trainer.fit(model, train_dataloader)

if __name__ == '__main__':
    main()
