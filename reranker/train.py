import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from model import Reranker
import pytorch_lightning as pl

model_name = 'bert-base-uncased'

data_dir = 'Data/MS_Marco/rerank'

train_input_ids_path = os.path.join(data_dir, 'train_input_ids.npy')
train_labels_path = os.path.join(data_dir, 'train_labels.npy')
validation_input_ids_path = os.path.join(data_dir, 'validation_input_ids.npy')
validation_labels_path = os.path.join(data_dir, 'validation_labels.npy')

class TokenizedDataset(Dataset):
    def __init__(self, inputs):
        self.data = inputs

    def __len__(self):
        return len(self.data['input_ids'])
    
    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.data.items()}


if __name__ == '__main__':

    dataset = {'train': dict(), 'validation': dict()}

    dataset['train']['input_ids'] = torch.tensor(np.load(train_input_ids_path))
    dataset['train']['labels'] = torch.nn.functional.one_hot(torch.tensor(np.load(train_labels_path)))
    dataset['validation']['input_ids'] = torch.tensor(np.load(validation_input_ids_path))
    dataset['validation']['labels'] = torch.nn.functional.one_hot(torch.tensor(np.load(validation_labels_path)))

    train_dataset = TokenizedDataset(dataset['train'])
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=10)

    validation_dataset = TokenizedDataset(dataset['validation'])
    validation_dataloader = DataLoader(validation_dataset, batch_size=16, shuffle=False, num_workers=10)

    model = Reranker(model_name=model_name)

    trainer = pl.Trainer(gpus=1, max_epochs=99)
    trainer.fit(model, train_dataloader, validation_dataloader)



