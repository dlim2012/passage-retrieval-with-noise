import torch
from transformers import BertModel, BertForNextSentencePrediction

#################### Model

class Reranker(torch.nn.Module):
    def __init__(self):
        super(Reranker, self).__init__()
        #self.encoder = BertModel.from_pretrained('bert-base-uncased')
        self.encoder = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
        
    def forward(self, x):
        
        outputs = self.encoder(input_ids=x['input_ids'],
                               attention_mask=x['attention_mask'],
                               labels=x['labels'])
        
        return outputs
    
    
    
from torch.utils.data import Dataset, DataLoader
import csv, random
from transformers import BertTokenizer

#################### Dataloader

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)

train_data_file = '/mnt/ssd/data/ms_marco/original/triples.train.small.tsv'
npr = 1

batch_size = 24
train_steps = 100000
n_train_instances = batch_size * train_steps # 12800000

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

        # Append input ids, attention masks, and labels to lists
        input_ids.append(token_ids)
        attention_mask.append([1] * len(token_ids))
        labels.append(int(label))

        # Track the maximum length for padding purpose
        max_len = max(max_len, len(token_ids))

    # Pad to the longest length
    for i in range(len(data)):
        input_ids[i] = input_ids[i] + [0] * (max_len - len(input_ids[i]))
        attention_mask[i] = attention_mask[i] + [0] * (max_len - len(attention_mask[i]))

    x = {'input_ids': torch.tensor(input_ids),
         'attention_mask': torch.tensor(attention_mask),
         'labels': torch.tensor(labels)}

    return x

lines = []
with open(train_data_file, 'r') as f:
    reader = csv.reader(f, delimiter='\t')
    for i, line in enumerate(reader):
        if len(lines) == n_train_instances:
            break
        if random.random() < 1/ npr:
            lines.append([line[0], line[1], 1])
        if len(lines) == n_train_instances:
            break
        lines.append([line[0], line[2], 0])

# Make a dataset
dataset = qidpidlabelDataset(lines)

# Make a dataloader using collate_fn
# Use multiple processors and tokenize texts on the fly
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=10, collate_fn=collate_fn)

#################### Tools

def linear_learning_rate_scheduler(optimizer, steps, target, warm_up, decay):
    if steps < warm_up:
        running_lr = target * steps / warm_up
    else:
        running_lr = target * (decay - target) / (decay - warm_up)
        
    for g in optimizer.param_groups:
        g['lr'] = running_lr
    return optimizer, running_lr
  
#################### Train 
  
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model = Reranker().to(device)
#self.criterion = torch.nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-6, weight_decay=0.01)
lr = 3e-6
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
lr_schedule = True

#Save name
run_name = '0418_1_train-small-npr1'

# Checkpoint directory
measure_steps = 250
ckpt_dir = os.path.join('/mnt/ssd/checkpoints/reranker', run_name)
os.makedirs(ckpt_dir, exist_ok=True)
ckpt_name = os.path.join(ckpt_dir, 'steps=%d_loss-interval=%.4f.pth')
ckpt_info = [[None, float('inf')] for i in range(5)]

# Log directory
log_dir = os.path.join('log/reranker/pytorch_logs', run_name)
writer = torch.utils.tensorboard.SummaryWriter(log_dir)
os.makedirs(log_dir, exist_ok=True)

loss_intervals = []
accs = []
for epoch in range(1):
    running_loss = 0.0
    running_accuracy = 0.0

    pbar = tqdm(enumerate(dataloader))
    for i, data in pbar:
        
        if lr_schedule:
            optimizer, running_lr = linear_learning_rate_scheduler(optimizer, i+1, lr, train_steps/10, train_steps)
        
        
        data = {key: value.to(device) for key, value in data.items()}
        optimizer.zero_grad()
        outputs = model(data)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        running_accuracy += torch.sum((torch.argmax(outputs.logits, dim=1) == data['labels']).type(torch.int32)).item() / len(data['labels'])
        
        
        if (i+1) % measure_steps == 0:
            #model.train(False)
            
            avg_loss = running_loss / measure_steps
            avg_acc = running_accuracy / measure_steps
            
            #print('%d, %.4f, %.4f' % (i+1, avg_loss, avg_acc))
            writer.add_scalar('loss', avg_loss, i)
            writer.add_scalar('acc', avg_acc, i)
            
            writer.add_scalar('lr', running_lr, i )
            loss_intervals.append(avg_loss)
            accs.append(avg_acc)
            
            running_loss = 0.0
            running_accuracy = 0.0
            #writer.flush()
            
            ckpt_info.sort(key = lambda x: x[1], reverse=True)
            if avg_loss < ckpt_info[0][1]:
                if ckpt_info[0][0] != None:
                    os.remove(ckpt_info[0][0])
                save_name = ckpt_name % (i+1, avg_loss)
                torch.save(model, save_name)
                ckpt_info[0] = [save_name, avg_loss]
                
            postfix = {'loss_interval': avg_loss, 'acc': avg_acc, 'lr': running_lr}
            pbar.set_postfix(postfix)
            
            #model.train(True)
save_name = ckpt_name % (i+1, avg_loss)
torch.save(model, save_name)

writer.add_hparams(
    {"lr": lr,
     "lr_schedule": lr_schedule,
     "train_steps": train_steps,
     "batch_size": batch_size,
     "train_data_file": train_data_file,
    'ckpt_dir': ckpt_dir}
)
writer.close()
