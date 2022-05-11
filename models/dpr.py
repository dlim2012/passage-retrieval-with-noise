from transformers import BertModel, BertTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
import sys
import numpy as np
from tqdm import tqdm

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)


class DPR(torch.nn.Module):
    def __init__(self, B, device):
        super(DPR, self).__init__()
        self.query_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.passage_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.B = B
        self.I = torch.eye(self.B)

        self.labels = torch.arange(self.B)
        self.device = device

    def forward(self, inputs):

        inputs = {key1: {key2: value2.to(self.device) for key2, value2 in value1.items()} for key1, value1 in inputs.items()}

        # Calculate the <CLS> vector of queries
        query_vectors = self.query_encoder(**inputs['queries']).last_hidden_state[:, 0, :]    # shape: (B, d)

        # Calculate the <CLS> vectors of positive passages
        positive_vectors = self.passage_encoder(**inputs['positives']).last_hidden_state[:, 0, :]    # shape: (B, d)

        # Calculate the <CLS> vectors of negative passages
        negative_vectors = self.passage_encoder(**inputs['negatives']).last_hidden_state[:, 0, :]    # shape: (B, d)

        # Concatenate the passage vectors
        passage_vectors = torch.cat((positive_vectors, negative_vectors), 0) # shape: (2B, d)

        # Calculate the similarity scores
        scores = torch.matmul(query_vectors, passage_vectors.T) # shape: (B, 2B)

        return scores

    def loss(self, scores, labels=None):

        # Change data type to avoid having 0.0000e+00 for some of the softmax result
        scores = scores.cpu().type(torch.float64)

        # Calculate the probability of predicting correctly
        x = torch.sum(torch.softmax(scores, dim=1)[:, :self.B] * self.I, axis=1)

        # Calculate the negative log likelihood
        x = -1 * torch.log(x)

        # Calculate the mean loss
        loss = torch.mean(x)

        return loss

    def accuracy(self, outputs, labels=None):
        outputs = outputs.cpu()
        if outputs.shape[0] != self.B:
            B = outputs.shape[0]
            labels = torch.arange(B)
            return torch.sum((torch.argmax(outputs, axis=1) == labels).type(torch.int32)).detach().tolist() / B
        return torch.sum((torch.argmax(outputs, axis=1) == self.labels).type(torch.int32)).detach().tolist() / self.B

    def scores_one_query(self, query, passages, batch_size, dataloader_args):

        query_inputs = tokenize([query])
        query_inputs = {key: value.to(self.device) for key, value in query_inputs.items()}
        query_vectors = self.query_encoder(**query_inputs).last_hidden_state[:, 0, :]

        dataloader = get_dataloader(
            data=passages,
            batch_size=batch_size,
            shuffle=False,
            args=dataloader_args
        )

        results = []
        for batch in dataloader:
            passage_inputs = {key: value.to(self.device) for key, value in batch.items()}
            passage_vectors = self.passage_encoder(**passage_inputs).last_hidden_state[:, 0, :]

            scores = torch.matmul(query_vectors, passage_vectors.T)[0].cpu().detach().numpy()
            results.append(scores)

        results = np.concatenate(results, axis=0)

        return results

    def dense_vectors(self, texts, batch_size, dataloader_args):

        dataloader = get_dataloader(
            data=texts,
            batch_size=batch_size,
            shuffle=False,
            args=dataloader_args #'mode': 'precompute'
        )

        results = []
        for i, batch in tqdm(enumerate(dataloader), desc='dense_vectors', total=dataloader.__len__()):
            batch = {key: value.to(self.device) for key, value in batch.items()}

            # Calculate the <CLS> vectors for the top 1000 passages
            vectors = self.passage_encoder(**batch).last_hidden_state[:, 0, :].cpu().detach().numpy().astype(np.float32)

            results.append(vectors)

        del dataloader

        results = np.concatenate(results, axis=0)

        return results


class DPRDataset(Dataset):
    """
    A Dataset class made to use PyTorch DataLoader
    """
    def __init__(self, inputs):
        self.data = inputs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def tokenize(texts):
    """
    tokenize texts and return input data types
    """
    return tokenizer(texts, padding="longest", max_length=512, truncation=True, return_tensors='pt')


def collate_fn(data):
    """
    Convert query, positive passage, and negative passage to input data types
    """
    x = dict()
    x['queries'] = tokenize([line[0] for line in data])
    x['positives'] = tokenize([line[1] for line in data])
    x['negatives'] = tokenize([line[2] for line in data])
    return x


def read_data(data_file, n_instances):

    data = []

    with open(data_file, 'r') as f:   # trying with train + val data ...
        for i, line in enumerate(f):
            line = line.split('\t')
            if len(data) == n_instances:
                break
            data.append(line)

    return data


def get_dataloaders(train_data_file,
                    validation_data_file,
                    n_train_instances,
                    n_validation_instances,
                    batch_size,
                    args=None):

    train_data = read_data(train_data_file, n_train_instances)
    validation_data = read_data(validation_data_file, n_validation_instances)

    train_dataset = DPRDataset(train_data)
    validation_dataset = DPRDataset(validation_data)


    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=10,
                            collate_fn=collate_fn)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=10,
                            collate_fn=collate_fn)

    return train_dataloader, validation_dataloader


def get_dataloader(batch_size, shuffle, args=None, data=None, data_file=None, n_instances=sys.maxsize):

    if not data:
        data = read_data(data_file, n_instances)

    dataset = DPRDataset(data)

    if args['mode'] == 'in-batch train':
        collate_fn_to_use = collate_fn
    elif args['mode'] in ['rerank', 'precompute']:
        collate_fn_to_use = tokenize
    else:
        print("Wrong argument for dataloader args (args['mode'] should be in ['in-batch train', 'rerank'])")
        sys.exit(1)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=10, collate_fn=collate_fn_to_use)

    return dataloader
