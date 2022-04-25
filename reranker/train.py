import torch
from torch.utils.data import Dataset, DataLoader
import csv, random
from transformers import BertTokenizer
import os
from tqdm import tqdm
from tensorboardX import SummaryWriter

from model import Reranker
import argparse

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version_name', type=str, required=True) # ex. 0419_1_lr1e-6...
    
    parser.add_argument('--train_data_file', type=str, default='/mnt/ssd/data/ms_marco/original/triples.train.small.tsv')
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--train_steps', type=int, default=100000)
    parser.add_argument('--npr', type=int, default=1)
    parser.add_argument('--lr', type=float, default=3e-6)
    parser.add_argument('--not_lr_schedule', default=False, action='store_true')
    parser.add_argument('--measure_steps', type=int, default=1000)

    parser.add_argument('--collate_fn2', default=False, action='store_true') # default: collate_fn(No queries[:64])
    # Initial version was collate_fn2(queries[:64])
    return parser.parse_args()

args = parse()

#################### Dataloader

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)


n_train_instances = args.batch_size * args.train_steps  # 12800000


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
        query, passage, label = line
        queries.append(query)
        passages.append(passage)
        labels.append(label)

    inputs = tokenizer(queries, passages, padding='longest', truncation=True, return_tensors='pt')
    inputs['labels'] = torch.tensor(labels)

    return inputs

def collate_fn_2(data):
    """
    Convert query, passage, and label to input_ids, attention_mask, and label
    """
    input_ids, attention_mask, token_type_ids, labels = [], [], [], []
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

        token_type_id = [0] * (len(query_token_ids) + 2) + [1] * (len(passage_token_ids) + 1)

        # Append input ids, attention masks, and labels to lists
        input_ids.append(token_ids)
        attention_mask.append([1] * len(token_ids))
        token_type_ids.append(token_type_id)
        labels.append(int(label))

        # Track the maximum length for padding purpose
        max_len = max(max_len, len(token_ids))

    # Pad to the longest length
    for i in range(len(data)):
        input_ids[i] = input_ids[i] + [0] * (max_len - len(input_ids[i]))
        attention_mask[i] = attention_mask[i] + [0] * (max_len - len(attention_mask[i]))
        token_type_ids[i] = token_type_ids[i] + [0] * (max_len - len(token_type_ids[i]))

    x = {'input_ids': torch.tensor(input_ids),
         'attention_mask': torch.tensor(attention_mask),
         'labels': torch.tensor(labels)}

    return x


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
    # Read training data
    lines = []
    with open(args.train_data_file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for i, line in enumerate(reader):
            if len(lines) == n_train_instances:
                break
            if random.random() < 1 / args.npr:
                lines.append([line[0], line[1], 1])
            if len(lines) == n_train_instances:
                break
            lines.append([line[0], line[2], 0])

    # Make a dataset
    dataset = qidpidlabelDataset(lines)

    # Make a dataloader using collate_fn
    # Use multiple processors and tokenize texts on the fly
    if args.collate_fn2:
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=10,
                                collate_fn=collate_fn_2)
    else:
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=10,
                                collate_fn=collate_fn)

    #################### Train
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    
    model = Reranker().to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Save name
    
    # Checkpoint directory
    ckpt_dir = os.path.join('/mnt/ssd/checkpoints/reranker', args.version_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_name = os.path.join(ckpt_dir, 'steps=%d_loss-interval=%.4f.pth')
    ckpt_info = [[None, float('inf')] for i in range(10)]
    
    # Log directory
    log_dir = os.path.join('log/reranker/pytorch_logs', args.version_name)
    #writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    writer = SummaryWriter(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    
    loss_intervals = []
    accs = []
    for epoch in range(1):
        running_loss = 0.0
        running_accuracy = 0.0
        running_lr = args.lr
    
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for i, data in pbar:
    
            if not args.not_lr_schedule:
                optimizer, running_lr = linear_learning_rate_scheduler(optimizer, i + 1, args.lr, args.train_steps / 10, args.train_steps)
    
            data = {key: value.to(device) for key, value in data.items()}
            optimizer.zero_grad()
            outputs = model(data)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
            running_accuracy += torch.sum(
                (torch.argmax(outputs.logits, dim=1) == data['labels']).type(torch.int32)).item() / len(data['labels'])
    
            if (i + 1) % args.measure_steps == 0:
                # model.train(False)

                avg_loss = running_loss / args.measure_steps
                avg_acc = running_accuracy / args.measure_steps
    
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