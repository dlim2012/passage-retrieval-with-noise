from transformers import BertModel, BertTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
import string
import sys
import numpy as np

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({'additional_special_tokens': ['[Q]', '[D]']})

# Token IDs
Q_token_id, D_token_id = tokenizer.convert_tokens_to_ids(['[Q]', '[D]'])
CLS_token_id, SEP_token_id, MASK_token_id, PAD_token_id = \
    tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]', '[MASK]', '[PAD]'])
punctuation_token_ids = set(tokenizer.convert_tokens_to_ids([ch for ch in string.punctuation]))

assert (CLS_token_id, SEP_token_id, MASK_token_id, PAD_token_id) == (101, 102, 103, 0)


class ColBERT(torch.nn.Module):
    def __init__(self, tokenizer, B, vector_size, device):
        super(ColBERT, self).__init__()
        self.tokenizer = tokenizer
        self.encoder = BertModel.from_pretrained('bert-base-uncased')
        self.encoder.resize_token_embeddings(len(tokenizer))

        self.linear = torch.nn.Linear(in_features=768, out_features=vector_size, bias=True)
        self.B = B
        self.I = torch.eye(self.B)

        self.labels = torch.arange(self.B)
        self.device = device


    def vector_representation(self, x):

        x = self.encoder(**x).last_hidden_state # Shape: (B, token_len, 768)

        x = self.linear(x) # Shape: (B, token_len, vector_size)

        x = torch.nn.functional.normalize(x) # Normalize to calculate the cosine simliarity

        return x

    def forward(self, inputs):

        for key, value in inputs.items():
            if type(value) == dict:
                inputs[key] = {key2: value2.to(self.device) for key2, value2 in value.items()}


        # Calculate the vector representations of queries
        query_vectors = self.vector_representation(inputs['queries'])    # shape: (B, N_q + 3, d)

        # Calculate the vector representations of positive passages
        positive_vectors = self.vector_representation(inputs['positives'])  # shape: (B, N_p + 3, d)

        # Calculate the vector representations of negative passages
        negative_vectors = self.vector_representation(inputs['negatives'])   # shape: (B, N_p + 3, d)

        # Similarity scores
        pos_scores = self.late_interaction(query_vectors, positive_vectors, inputs['positive_filter_indices']) # shape: (B, B)
        neg_scores = self.late_interaction(query_vectors, negative_vectors, inputs['negative_filter_indices']) # shape: (B, B)
        scores = torch.cat((pos_scores, neg_scores), 1) # shape: (B, 2B)

        return scores

    def late_interaction(self, query_vectors, passage_vectors, filter_indices):
        scores = torch.zeros(query_vectors.shape[0], passage_vectors.shape[0])

        #for i in range(query_vectors.shape[0]):
        #    for j in range(passage_vectors.shape[0]):
        #        scores[i, j] = torch.matmul(query_vectors[i], passage_vectors[j][punc_mask[j]].T).max(1).values.sum()

        for i in range(query_vectors.shape[0]):
            # Calculate cosine similarity scores between all query vectors and all passage vectors
            similarities = torch.matmul(query_vectors[i], torch.permute(passage_vectors, (0, 2, 1))).cpu()
            # shape: (B, N_q, N_p)
            for j in range(passage_vectors.shape[0]):
                # Filter punctuation vectors from passage vectors
                # Calculate the maximum similarity for each query vector and sum the results
                scores[i, j] = similarities[j][:, filter_indices[j]].max(1).values.sum()

        return scores

    def loss(self, outputs, labels=None):

        # Change data type to avoid having 0.0000e+00 for some of the softmax result
        scores = outputs.cpu().type(torch.float64)

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

        query_inputs = tokenize_queries([query], dataloader_args['N_q'])
        query_inputs = {key: value.to(self.device) for key, value in query_inputs.items()}
        query_vectors = self.vector_representation(query_inputs)

        dataloader = get_dataloader(
            data=passages,
            batch_size=batch_size,
            shuffle=False,
            args=dataloader_args
        )

        results = []
        for batch in dataloader:
            passage_inputs, filter_indices = batch
            passage_inputs = {key: value.to(self.device) for key, value in passage_inputs.items()}
            passage_vectors = self.vector_representation(passage_inputs)

            scores = self.late_interaction(query_vectors, passage_vectors, filter_indices)[0].cpu().detach().numpy()
            results.append(scores)

        results = np.concatenate(results, axis=0)

        return results


class TriplesDataset(Dataset):
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
    inputs = tokenizer(texts, padding="longest", max_length=511, truncation=True)
    return inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids']


def tokenize_queries(texts, N_q):
    input_ids, attention_mask, token_type_ids = tokenize(texts)

    n_instances = len(texts)

    for i in range(n_instances):
        # length of query tokens excluding '[CLS]', '[SEP]', and '[PAD]' tokens
        query_len = sum(attention_mask[i]) - 2

        # Add '[Q]' token and truncate or pad query tokens with '[MASK]' tokens to N_q
        if query_len > N_q:
            query_tokens = input_ids[i][1: 1 + N_q]
        else:
            query_tokens = input_ids[i][1: 1 + query_len] + [MASK_token_id] * (N_q - query_len)

        # ('[CLS]', '[Q]', q0, q1, ..., ql, '[MASK]', '[MASK]', ..., '[MASK]', '[SEP]')
        input_ids[i] = [CLS_token_id, Q_token_id] + query_tokens + [SEP_token_id]
        attention_mask[i] = [1] * (N_q + 3)
        token_type_ids[i] = [0] * (N_q + 3)

        assert len(input_ids[i]) == len(attention_mask[i])

    inputs = {
        'input_ids': torch.tensor(input_ids, dtype=torch.int32),
        'attention_mask': torch.tensor(attention_mask, dtype=torch.int32),
        'token_type_ids': torch.tensor(token_type_ids, dtype=torch.int32)
    }

    return inputs


def tokenize_passages(texts):
    n_instances = len(texts)

    input_ids, attention_mask, token_type_ids = tokenize(texts)
    filter_indices = [[] for _ in range(n_instances)]

    for i in range(n_instances):
        # Add '[D]' token
        input_ids[i] = [CLS_token_id, D_token_id] + input_ids[i][1:]
        attention_mask[i] = [1] + attention_mask[i]
        token_type_ids[i] = [0] + token_type_ids[i]

        # Get the indices that are not paddings nor punctuations
        for j in range(len(input_ids[i])):
            # Stop at pad token
            if input_ids[i][j] == 0:
                break

            # Append
            if input_ids[i][j] not in punctuation_token_ids:
                filter_indices[i].append(j)

    inputs = {
        'input_ids': torch.tensor(input_ids, dtype=torch.int32),
        'attention_mask': torch.tensor(attention_mask, dtype=torch.int32),
        'token_type_ids': torch.tensor(token_type_ids, dtype=torch.int32)
    }

    return inputs, filter_indices


def collate_fn(data):
    """
    Convert query, positive passage, and negative passage to input data types
    """
    x = dict()
    x['queries'] = tokenize_queries([line[0] for line in data], N_q)
    x['positives'], x['positive_filter_indices'] = tokenize_passages([line[1] for line in data])
    x['negatives'], x['negative_filter_indices'] = tokenize_passages([line[2] for line in data])
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

    global N_q
    N_q = args['N_q']

    train_data = read_data(train_data_file, n_train_instances)
    validation_data = read_data(validation_data_file, n_validation_instances)

    train_dataset = TriplesDataset(train_data)
    validation_dataset = TriplesDataset(validation_data)


    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=10,
                            collate_fn=collate_fn)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=10,
                            collate_fn=collate_fn)

    return train_dataloader, validation_dataloader


def get_dataloader(batch_size, shuffle, args, data=None, data_file=None, n_instances=sys.maxsize):

    global N_q
    N_q = args['N_q']

    if not data:
        data = read_data(data_file, n_instances)

    dataset = TriplesDataset(data)

    if args['mode'] == 'in-batch train':
        collate_fn_to_use = collate_fn
    elif args['mode'] == 'rerank':
        collate_fn_to_use = tokenize_passages
    else:
        print("Wrong argument for dataloader args (args['mode'] should be in ['in-batch train', 'rerank']), {} was given.".format(args['mode']))
        sys.exit(1)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=10, collate_fn=collate_fn_to_use)

    return dataloader