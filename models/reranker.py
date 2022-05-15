import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
import random
import sys
import numpy as np

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
CLS_token_id, SEP_token_id, PAD_token_id = tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]', '[PAD]'])

class Reranker(torch.nn.Module):
    def __init__(self, device):
        super(Reranker, self).__init__()

        #self.encoder = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
        self.encoder = BertModel.from_pretrained('bert-base-uncased')
        self.linear = torch.nn.Linear(in_features=768, out_features=2, bias=True)
        self.criterion = torch.nn.CrossEntropyLoss()

        self.device = device
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, batch):
        inputs = {key: value.to(self.device) for key, value in batch.items() if key != 'labels'}

        outputs = self.linear(self.encoder(**inputs).last_hidden_state[:, 0, :])

        return outputs

    def loss(self, outputs, labels):
        return self.criterion(outputs, labels)

    def accuracy(self, outputs, labels):
        return torch.sum((torch.argmax(outputs.cpu(), dim=1) == labels).type(torch.int32)).item() / len(labels)

    def score(self, inputs):
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        return self.softmax(self.forward(inputs))[:, 1]

    def scores_one_query(self, query, passages, batch_size, dataloader_args):
        """
        scores for each query, passage pair
        :param data:
        :param batch_size:
        :param dataloader_args:
        :return:
        """
        data = [[query, passage] for passage in passages]

        dataloader = get_dataloader(
            data=data,
            batch_size=batch_size,
            shuffle=False,
            args=dataloader_args
        )

        # Gather scores for each passage in the top 1000
        results = []
        for batch in dataloader:
            # Calculate the <CLS> vectors for the top 1000 passages
            scores = self.score(batch).cpu().detach().numpy()

            # Append the similarity scores for passages in a batch to a list
            results.append(scores)

        # Get the similarity scores between query and its top 1000 passages in a 1-dimensional vector
        results = np.concatenate(results, axis=0)

        return results


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


def collate_fn(data):
    """
    Convert query, passage, and label to input_ids, attention_mask, and label
    """

    queries, passages, labels = [], [], []
    for line in data:
        if len(line) == 2:
            query, passage = line
        else:
            query, passage, label = line
            labels.append(label)
        queries.append(query)
        passages.append(passage)

    inputs = tokenizer(queries, passages, padding=True, truncation=True, return_tensors='pt')

    if not len(labels) == 0:
        inputs['labels'] = torch.tensor(labels)

    return inputs


def collate_fn2(data):
    """
    Convert query, passage, and label to input_ids, attention_mask, and label
    """
    input_ids, attention_mask, token_type_ids, labels = [], [], [], []
    max_len = -1
    for line in data:
        if len(line) == 2:
            query, passage = line
        elif len(line) == 3:
            query, passage, label = line
            labels.append(int(label))

        # Truncate query to have at most 64 tokens
        query_tokens = tokenizer.tokenize(query)[:64]

        # Convert tokens of query to token ids
        query_token_ids = tokenizer.convert_tokens_to_ids(query_tokens)

        # Truncate passage so that [<CLS> query <SEP> passage <SEP>] doesn't exceed 512
        passage_tokens = tokenizer.tokenize(passage)[:509 - len(query_token_ids)]

        # Convert tokens of passage to token ids
        passage_token_ids = tokenizer.convert_tokens_to_ids(passage_tokens)

        # Token ids for input
        token_ids = [CLS_token_id] + query_token_ids + [SEP_token_id] + passage_token_ids + [SEP_token_id]

        token_type_id = [0] * (len(query_token_ids) + 2) + [1] * (len(passage_token_ids) + 1)

        # Append input ids, attention masks, and labels to lists
        input_ids.append(token_ids)
        attention_mask.append([1] * len(token_ids))
        token_type_ids.append(token_type_id)


        # Track the maximum length for padding purpose
        max_len = max(max_len, len(token_ids))

    # Pad to the longest length
    for i in range(len(data)):
        input_ids[i] = input_ids[i] + [PAD_token_id] * (max_len - len(input_ids[i]))
        attention_mask[i] = attention_mask[i] + [PAD_token_id] * (max_len - len(attention_mask[i]))
        token_type_ids[i] = token_type_ids[i] + [PAD_token_id] * (max_len - len(token_type_ids[i]))

    x = {'input_ids': torch.tensor(input_ids),
         'attention_mask': torch.tensor(attention_mask)}

    if not len(labels) == 0:
        x['labels']: torch.tensor(labels)

    return {'inputs': x}



def get_dataloaders(train_data_file,
                    validation_data_file,
                    n_train_instances,
                    n_validation_instances,
                    batch_size,
                    args):


    train_data = read_data(train_data_file, args['npr'], n_train_instances)
    validation_data = read_data(validation_data_file, args['npr'], n_validation_instances)

    train_dataset = qidpidlabelDataset(train_data)
    validation_dataset = qidpidlabelDataset(validation_data)

    collate_fn_to_use = collate_fn if not args['use_collate_fn2'] else collate_fn2

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=10,
                            collate_fn=collate_fn_to_use)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=10,
                            collate_fn=collate_fn_to_use)

    return train_dataloader, validation_dataloader

def read_data(data_file, npr, n_instances):
    data = []

    with open(data_file, 'r') as f:
        for i, line in enumerate(f):
            line = line.split('\t')
            if len(data) == n_instances:
                break
            if random.random() < 1 / npr:
                data.append([line[0], line[1], 1])
            if len(data) == n_instances:
                break
            data.append([line[0], line[2], 0])

    return data


def get_dataloader(batch_size, shuffle, args, data=None, data_file=None, n_instances=sys.maxsize):

    if not data:
        data = read_data(data_file, args['npr'], n_instances)

    dataset = qidpidlabelDataset(data)

    collate_fn_to_use = collate_fn if not args['use_collate_fn2'] else collate_fn2

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=10,
                            collate_fn=collate_fn_to_use)

    return dataloader



