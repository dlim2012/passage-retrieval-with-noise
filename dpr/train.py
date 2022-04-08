from torch.utils.data import Dataset, DataLoader
import torch
from model import DPR
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import pytorch_lightning as pl
from transformers import BertTokenizer
import csv
import os

model_name, batch_size = 'bert-base-uncased', 16
tokenizer = BertTokenizer.from_pretrained(model_name)

qidpidtriples_file = 'data/ms_marco/preprocessed/dpr/train/qidpidtriples.train.full.filtered.text.tsv'

# Information needed to set linear learning rate scheduler
n_train_triples = batch_size * 200000
linear_scheduler_steps=(n_train_triples//(batch_size*10),
                        n_train_triples//batch_size + 1)


# Directory to save checkpoints
checkpoint_dir = 'checkpoints/dpr/'
every_n_train_steps = 10000

#
log_dir = "log/dpr/"
os.makedirs(log_dir, exist_ok=True)


class TriplesDataset(Dataset):
    def __init__(self, inputs):
        self.data = inputs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def tokenize(texts):
    inputs = tokenizer(texts, padding="longest", max_length=512, truncation=True)
    return torch.tensor(inputs["input_ids"]), torch.tensor(inputs["attention_mask"])

def collate_fn(data):
    x = dict()
    x['query_input_ids'], x['query_attention_mask'] = tokenize([line[0] for line in data])
    x['positive_input_ids'], x['positive_attention_mask'] = tokenize([line[1] for line in data])
    x['negative_input_ids'], x['negative_attention_mask'] = tokenize([line[2] for line in data])
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
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=10, collate_fn=collate_fn)
    return dataloader

def main():
    # Declare a model
    model = DPR(linear_scheduler_steps=linear_scheduler_steps, B=batch_size, model_name=model_name)

    # Use a TensorBoardLogger
    logger = pl.loggers.TensorBoardLogger(save_dir="log/dpr/")

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

    trainer.fit(model, train_dataloader)

if __name__ == '__main__':
    main()
