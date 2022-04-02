import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from model import Reranker
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl

# Name of the pre-trained BERT to use
model_name = 'bert-base-uncased'

# Directory to read data from
data_dir = 'data/ms_marco/rerank'

# Path to read data
train_input_ids_path = os.path.join(data_dir, 'train_input_ids.npy')
train_labels_path = os.path.join(data_dir, 'train_labels.npy')
validation_input_ids_path = os.path.join(data_dir, 'validation_input_ids.npy')
validation_labels_path = os.path.join(data_dir, 'validation_labels.npy')

# Directory to save checkpoints
checkpoint_dir = 'checkpoints/reranker/'

every_n_train_steps = 10000

class TokenizedDataset(Dataset):
    """
    A Dataset class made to use PyTorch DataLoader
    """
    def __init__(self, inputs):
        self.data = inputs

    def __len__(self):
        return len(self.data['input_ids'])
    
    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.data.items()}


if __name__ == '__main__':

    # Read data
    dataset = {'train': dict(), 'validation': dict()}
    dataset['train']['input_ids'] = torch.tensor(np.load(train_input_ids_path))
    dataset['train']['labels'] = torch.nn.functional.one_hot(torch.tensor(np.load(train_labels_path))).type(torch.float32)
    dataset['validation']['input_ids'] = torch.tensor(np.load(validation_input_ids_path))
    dataset['validation']['labels'] = torch.nn.functional.one_hot(torch.tensor(np.load(validation_labels_path))).type(torch.float32)

    # Declare train and validation dataloader
    train_dataset = TokenizedDataset(dataset['train'])
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=10)
    validation_dataset = TokenizedDataset(dataset['validation'])
    validation_dataloader = DataLoader(validation_dataset, batch_size=16, shuffle=False, num_workers=10)

    # Declare a model
    model = Reranker(model_name=model_name)

    # Use a TensorBoardLogger
    logger = pl.loggers.TensorBoardLogger(save_dir="log/rerank/")

    # Save checkpoint at regular intervals
    regular_checkpoint = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="{epoch}-{steps:.0f}",
            monitor="steps",
            every_n_train_steps=every_n_train_steps,
            save_top_k=-1,
    )

    # Save checkpoint after every epoch
    epoch_checkpoint = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="{epoch}-end",
            monitor="val_loss",
    )

    # Train the model
    trainer = pl.Trainer(
            gpus=1,
            max_epochs=1,
            logger=logger,
            callbacks=[regular_checkpoint, epoch_checkpoint]
    )

    trainer.fit(model, train_dataloader, validation_dataloader)
    


