import torch
from torch.utils.data import Dataset, DataLoader
import csv, random
from transformers import BertTokenizer
import os
from tqdm import tqdm
from tensorboardX import SummaryWriter
import string

from model import ColBERT
import argparse

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version_name', type=str, required=True)  # ex. 0419_1_lr1e-6...

    parser.add_argument('--train_data_file', type=str,
                        default='/mnt/ssd/data/ms_marco/preprocessed/dpr/qidpidtriples.train.full.filtered.text.tsv')
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--train_steps', type=int, default=200000)
    parser.add_argument('--lr', type=float, default=3e-6)
    parser.add_argument('--not_lr_schedule', default=False, action='store_true')
    parser.add_argument('--measure_steps', type=int, default=2000)

    parser.add_argument('--vector_size', type=int, default=128) # padding length for queries
    parser.add_argument('--N_q', type=int, default=32) # vector size for each token
    return parser.parse_args()


args = parse()


#################### Data preprocessing

# Tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({'additional_special_tokens': ['[Q]', '[D]']})

# Token IDs
Q_token_id, D_token_id = tokenizer.convert_tokens_to_ids(['[Q]', '[D]'])
CLS_token_id, SEP_token_id, MASK_token_id, PAD_token_id = \
    tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]', '[MASK]', '[PAD]'])
punctuation_token_ids = set(tokenizer.convert_tokens_to_ids([ch for ch in string.punctuation]))

assert CLS_token_id == 101
assert SEP_token_id == 102
assert MASK_token_id == 103
assert PAD_token_id == 0

n_train_instances = args.batch_size * args.train_steps  # 12800000

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

def tokenize_queries(texts):
    input_ids, attention_mask, token_type_ids = tokenize(texts)

    n_instances = len(texts)

    for i in range(n_instances):
        # length of query tokens excluding '[CLS]', '[SEP]', and '[PAD]' tokens
        query_len = sum(attention_mask[i]) - 2

        # Add '[Q]' token and truncate or pad query tokens with '[MASK]' tokens to N_q
        if query_len > args.N_q:
            query_tokens = input_ids[i][1: 1+args.N_q]
        else:
            query_tokens = input_ids[i][1: 1+query_len] + [MASK_token_id] * (args.N_q - query_len)
        
        # ('[CLS]', '[Q]', q0, q1, ..., ql, [MASK], [MASK], ..., [MASK], '[SEP]')
        input_ids[i] = [CLS_token_id, Q_token_id] + query_tokens + [SEP_token_id]
        attention_mask[i] = [1] * (args.N_q + 3)
        token_type_ids[i] = [0] * (args.N_q + 3)
        
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
    x['queries'] = tokenize_queries([line[1] for line in data])
    x['positives'], x['positive_filter_indices']= tokenize_passages([line[1] for line in data])
    x['negatives'], x['negative_filter_indices'] = tokenize_passages([line[2] for line in data])
    return x


def batch_to_device(batch, device):
    for key, value in batch.items():
        if type(value) == dict:
            batch[key] = {key2: value2.to(device) for key2, value2 in value.items()}
    return batch



#################### Tools

def linear_learning_rate_scheduler(optimizer, steps, target, warm_up, decay):
    if steps < warm_up:
        running_lr = target * steps / warm_up
    else:
        running_lr = target * (decay - steps) / (decay - warm_up)

    for g in optimizer.param_groups:
        g['lr'] = running_lr
    return optimizer, running_lr



def main():
    #################### Get dataloader
    lines = []
    with open(args.train_data_file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for i, line in enumerate(reader):
            if i == n_train_instances:
                break
            lines.append(line)

    # Make a dataset
    dataset = TriplesDataset(lines)

    # Make a dataloader using collate_fn
    # Use multiple processors and tokenize texts on the fly
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, collate_fn=collate_fn)


    #################### Train

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    model = ColBERT(
        tokenizer=tokenizer,
        B=args.batch_size,
        vector_size=args.vector_size,
        device=device
    ).to(device)

    # self.criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-6, weight_decay=0.01)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Checkpoint directory
    ckpt_dir = os.path.join('/mnt/ssd/checkpoints/colbert', args.version_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_name = os.path.join(ckpt_dir, 'steps=%d_loss-interval=%.4f.pth')
    ckpt_info = [[None, float('inf')] for i in range(5)]

    # Log directory
    log_dir = os.path.join('log/colbert/pytorch_logs', args.version_name)
    # writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    writer = SummaryWriter(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    loss_intervals = []
    accs = []
    labels = torch.arange(args.batch_size)
    for epoch in range(1):
        running_loss = 0.0
        running_accuracy = 0.0
        running_lr = args.lr

        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for i, data in pbar:

            if not args.not_lr_schedule:
                optimizer, running_lr = linear_learning_rate_scheduler(optimizer, i + 1, args.lr, args.train_steps / 10,
                                                                       args.train_steps)

            data = batch_to_device(data, device)

            optimizer.zero_grad()
            scores = model(data).cpu()
            loss = model.loss(scores)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_accuracy += torch.sum((torch.argmax(scores, axis=1) == labels)\
                                          .type(torch.int32)).detach().tolist()
            print(running_loss, running_accuracy)

            if (i + 1) % args.measure_steps == 0:
                # model.train(False)

                avg_loss = running_loss / args.measure_steps
                avg_acc = running_accuracy / (args.batch_size * args.measure_steps)

                # print('%d, %.4f, %.4f' % (i+1, avg_loss, avg_acc))
                writer.add_scalar('loss', avg_loss, i)
                writer.add_scalar('acc', avg_acc, i)

                writer.add_scalar('lr', running_lr, i)
                loss_intervals.append(avg_loss)
                accs.append(avg_acc)

                running_loss = 0.0
                running_accuracy = 0.0
                # writer.flush()

                ckpt_info.sort(key=lambda x: x[1], reverse=True)
                if avg_loss < ckpt_info[0][1]:
                    if ckpt_info[0][0] != None:
                        os.remove(ckpt_info[0][0])
                    save_name = ckpt_name % (i + 1, avg_loss)
                    torch.save(model, save_name)
                    ckpt_info[0] = [save_name, avg_loss]

                postfix = {'loss_interval': avg_loss, 'acc': avg_acc, 'lr': running_lr}
                pbar.set_postfix(postfix)

                # model.train(True)
    save_name = ckpt_name % (i + 1, avg_loss)
    torch.save(model, save_name)

    writer.add_hparams(
        {"lr": args.lr,
         "not_lr_schedule": args.not_lr_schedule,
         "train_steps": args.train_steps,
         "batch_size": args.batch_size,
         "train_data_file": args.train_data_file,
         'ckpt_dir': args.ckpt_dir},
        {'min_loss': min(loss_intervals),
         'max_acc': max(accs)}
    )
    writer.close()

if __name__ == '__main__':
    main()