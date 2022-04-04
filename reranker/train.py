import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from model import Reranker
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl

# Name of the pre-trained BERT to use and some hyperparameters
model_name, batch_size = 'bert-large-uncased', 4

model_name, batch_size = 'bert-base-uncased', 24

""" memo
bert-base-uncased: 24(o), 32(x)
bert-large-uncased: 4(o), 8(x)
"""

# Directory to read data from
data_dir = 'data/ms_marco/preprocessed/reranker_0404/'

# path format to read data
input_ids_path = os.path.join(data_dir, '{dataset_type}_input_ids_npr4_{idx}.npy')
labels_path = os.path.join(data_dir, '{dataset_type}_labels_npr4_{idx}.npy')


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

def load(path, one_hot=False):
    return torch.tensor(np.load(path).astype(np.int32))


def get_dataloader(dataset_type, idx, shuffle=False):
    
    dataset = dict()
    
    dataset['input_ids'] = load(input_ids_path.format(dataset_type=dataset_type, idx=idx))
    dataset['labels'] = load(labels_path.format(dataset_type=dataset_type, idx=idx)).type(torch.float32)
    
    train_dataset = TokenizedDataset(dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=10)
    
    return train_dataloader


def main():

    # Declare a model
    linear_scheduler_steps = (50000, 12800000//batch_size+1)
    model = Reranker(linear_scheduler_steps=linear_scheduler_steps,
                     model_name=model_name)

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
            max_epochs=9,
            logger=logger,
            callbacks=[regular_checkpoint, epoch_checkpoint]
    )
    
    # Get dataloaders
    train_dataloader = get_dataloader('train', 'all', shuffle=True)
    validation_dataloader = get_dataloader('validation', '0', shuffle=False) 
    
    trainer.fit(model, train_dataloader, validation_dataloader)
    
if __name__ == '__main__':
    
    main()
    
    
