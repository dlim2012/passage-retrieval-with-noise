import torch
from transformers import BertModel, get_linear_schedule_with_warmup
import pytorch_lightning as pl
import numpy as np

class Reranker(pl.LightningModule):
    def __init__(self, linear_scheduler_steps, model_name='bert-large-uncased'):
        super().__init__()

        # Download a pre-trained BERT model
        self.bert = BertModel.from_pretrained(model_name)
        self.linear_scheduler_steps = linear_scheduler_steps

        # Linear layer that will use the <CLS> vector as its input
        if model_name == 'bert-large-uncased':
            self.linear = torch.nn.Linear(in_features=1024, out_features=2, bias=True)
        else: # bert-base-uncased:
            self.linear = torch.nn.Linear(in_features=768, out_features=2, bias=True)
            
        # Cross-entropy loss
        weight = None
        if weight:
            self.criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weight))
        else:
            self.criterion = torch.nn.CrossEntropyLoss()
        
        # Monitor training steps for logging purpose
        self.steps = torch.tensor(0).type(torch.float32)
        
        # counts and step intervals for performance logging
        self.counts = np.zeros(4) + 1e-10
        self.counts_all = np.zeros(4)
        self.measure_steps = 1000
        
    def forward(self, x):
        """
        :param x: x['input_ids'] is the input to the BERT
        """
        
        # Calculate the last hidden layer of the output
        output = self.bert(x['input_ids']).last_hidden_state

        # Use the <CLS> vector only
        output = output[:, 0, :]
        
        # Linear layer
        output = self.linear(output)

        return output
    
    def training_step(self, batch, batch_idx):
        self.steps += 1
        
        # Compute the forward path
        output = self.forward(batch)
        
        # Compute the cross-entropy loss
        loss = self.criterion(output, batch['labels'])
        
        #self.add_results_to_buffer(output, batch['labels'])
        self.add_counts(output, batch['labels'])

        # Save log
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('steps', self.steps, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        if self.steps % self.measure_steps == 0:
            self.log_performance(log_type='train_step', on_step=True)
            
        # Return loss for backpropagation
        return {'loss': loss}
    
    def train_epoch_end(self, outputs):
        self.log_performance(log_type='train_all', on_step=False)
        self.log_performance(log_type='train_step', on_step=True)
        
    def validation_step(self, batch, batch_idx):

        # Compute the forward path
        output = self.forward(batch)

        # Compute the cross-entropy loss
        loss = self.criterion(output, batch['labels'])
        
        # Save log
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
    def validation_epoch_end(self, outputs):
        self.log_performance(log_type='validation_all', on_step=False)
        
    def configure_optimizers(self):
        
        # Using Adam optimizer
        # Configuration as in the paper: lr=3e-6, beta1=0.9, beta2=0.999, L2_weight_decay=0.01
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=3e-6, weight_decay=0.01)
        
        # Configuration as in the paper: learning rate warmup over the first 10,000 steps
        schedulers = [
            {
                'scheduler': get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=self.linear_scheduler_steps[0],
                    num_training_steps=self.linear_scheduler_steps[1]
                ),
                'name': 'warm_up_lr',
                'interval': 'step'
            }
        ]
        
        # Difference from the paper: linear decay of learning rate is not applied and we use smaller batch size
        
        return [optimizer], schedulers
    
    def add_counts(self, output, labels):
        output = np.argmax(output.cpu().detach().clone().numpy(), axis=1)
        labels = np.argmax(labels.cpu().detach().clone().numpy(), axis=1)
        
        tp = np.sum((output == 1) * (labels == 1))
        fp = np.sum((output == 1) * (labels == 0))
        fn = np.sum((output == 0) * (labels == 1))
        tn = np.sum((output == 0) * (labels == 0))
        
        self.counts += np.array([tp, fp, fn, tn])
        self.counts_all += np.array([tp, fp, fn, tn])
        
        
    def performance(self, counts):
        tp, fp, fn, tn = counts
        
        accuracy = (tp + tn) / (tp + fp + tn + fn + 1e-10)
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        return accuracy, precision, recall, f1
    
    def log_performance(self, log_type, on_step=True):
        
        if log_type == 'train_step':
            tags = ['acc', 'prec', 'rec', 'f1']
            counts = self.counts
        elif log_type == 'train_all':
            tags = ['acc_all', 'prec_all', 'rec_all', 'f1_all']
            counts = self.counts_all
        elif log_type == 'validation_step':
            tags = ['val_acc', 'val_prec', 'val_rec', 'val_f1']
            counts = self.counts
        elif log_type == 'validation_all':
            tags = ['val_acc_all', 'val_prec_all', 'val_rec_all', 'val_f1_all']
            counts = self.counts_all

        accuracy, precision, recall, f1 = self.performance(counts)
            
        self.log(tags[0], accuracy, on_step=on_step, on_epoch=True, prog_bar=True, logger=True)
        self.log(tags[1], precision, on_step=on_step, on_epoch=True, prog_bar=True, logger=True)
        self.log(tags[2], recall, on_step=on_step, on_epoch=True, prog_bar=True, logger=True)
        self.log(tags[3], f1, on_step=on_step, on_epoch=True, prog_bar=True, logger=True)

        print("%.6f\t%.6f\t%.6f\t%.6f" % (accuracy, precision, recall, f1))
        self.counts = np.zeros(4)
        

