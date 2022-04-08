import csv
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from model import Reranker
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import pytorch_lightning as pl
from transformers import BertTokenizer

# Name of the pre-trained BERT to use and some hyperparameters
model_name, batch_size = 'bert-large-uncased', 4
model_name, batch_size = 'bert-base-uncased', 24
tokenizer = BertTokenizer.from_pretrained(model_name)

""" memo
bert-base-uncased: 24(o), 32(x)
bert-large-uncased: 4(o), 8(x)
"""

# Directory to read data from
train_data_file = 'data/ms_marco/preprocessed/reranker/train/qidpidlabel.text.tsv'

# Information needed to set linear learning rate scheduler
n_train_instances = batch_size * 100000 # 12800000
linear_scheduler_steps=(n_train_instances//(batch_size*10),
                        n_train_instances//batch_size + 1)


# Directory to save checkpoints
checkpoint_dir = 'checkpoints/reranker/'
every_n_train_steps = 1000



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

def one_hot_vector(x, n_values=None):
    """
    :param x: one dimensional list or np.ndarray
    return one hot vector
    """
    x = np.array(x)
    if not n_values:
        n_values = np.max(x) + 1
    return np.eye(n_values)[x]

def collate_fn(data):
    """
    Convert query, passage, and label to input_ids, attention_mask, and label
    """
    input_ids, attention_mask, labels = [], [], []
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


        attention_mask.append([1] * len(token_ids))
        input_ids.append(token_ids)
        labels.append(int(label))

        max_len = max(max_len, len(token_ids))

    # Pad to the longest length
    for i in range(len(data)):
        input_ids[i] = input_ids[i] + [0] * (max_len - len(input_ids[i]))
        attention_mask[i] = attention_mask[i] + [0] * (max_len - len(attention_mask[i]))

    x = {'input_ids': torch.tensor(input_ids),
         'attention_mask': torch.tensor(attention_mask),
         'labels': torch.tensor(one_hot_vector(labels, 2))}

    return x


def get_dataloader():
    lines = []
    with open(train_data_file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for i, line in enumerate(reader):
            if i == n_train_instances:
                break
            lines.append(line)

    dataset = qidpidlabelDataset(lines)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=10, collate_fn=collate_fn)
    return dataloader



def main():

    # Declare a model
    model = Reranker(linear_scheduler_steps=linear_scheduler_steps,
                     model_name=model_name)

    # Use a TensorBoardLogger
    logger = pl.loggers.TensorBoardLogger(save_dir="log/rerank/")

    # Save checkpoint: three lowest loss_interval
    regular_checkpoint = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="{epoch}-{steps:.0f}",
            monitor="loss_interval",
            mode='min',
            every_n_train_steps=every_n_train_steps,
            save_top_k=3,
    )

    # Save checkpoint after every epoch
    epoch_checkpoint = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="{epoch}-end",
            monitor="val_loss",
    )

    # Log learning rate
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Train the model
    trainer = pl.Trainer(
            gpus=1,
            max_epochs=9,
            logger=logger,
            callbacks=[regular_checkpoint, epoch_checkpoint, lr_monitor]
    )
    
    
    # Get dataloaders
    train_dataloader = get_dataloader()
    
    trainer.fit(model, train_dataloader)
    
if __name__ == '__main__':
    
    main()
