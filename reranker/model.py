import torch
from transformers import BertModel, get_linear_schedule_with_warmup
import pytorch_lightning as pl
import numpy as np

class Reranker(pl.LightningModule):
    def __init__(self, linear_scheduler_steps=None, measure_steps=250, model_name='bert-large-uncased'):
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
        self.loss_interval = torch.tensor(0).type(torch.float32)
        self.measure_steps = measure_steps
        
    def forward(self, x):
        """
        :param x: x['input_ids'] is the input to the BERT
        """
        
        # Calculate the last hidden layer of the output
        output = self.bert(input_ids=x['input_ids'], attention_mask=x['attention_mask']).last_hidden_state

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
        self.loss_interval += loss.cpu().detach().clone()

        # Save log
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('steps', self.steps, on_step=True, on_epoch=False, prog_bar=False, logger=True)

        if self.steps % self.measure_steps == 0:
            # log average loss of the past 'measure_steps' steps
            loss_interval = self.loss_interval / self.measure_steps
            self.log('loss_interval', loss_interval, on_step=True, on_epoch=False, prog_bar=True, logger=True)

            # log performance of the past 'measure_steps' steps
            accuracy, precision, recall, f1 = self.performance(self.counts)
            self.log('acc', accuracy, on_step=True, on_epoch=False, prog_bar=True, logger=True)
            self.log('prec', precision, on_step=True, on_epoch=False, prog_bar=True, logger=True)
            self.log('rec', recall, on_step=True, on_epoch=False, prog_bar=True, logger=True)
            self.log('f1', f1, on_step=True, on_epoch=False, prog_bar=True, logger=True)
                     
            # reset some measurements
            self.loss_interval = torch.tensor(0).type(torch.float32)
            self.counts = np.zeros(4) + 1e-10

            # print intermediate results
            print('accuracy: %.4f, precision: %.4f, recall: %.4f, f1: %.4f, loss_interval: %.4f' \
                    % (accuracy, precision, recall, f1, loss_interval))

        # Return loss for backpropagation
        return {'loss': loss}

    def train_epoch_end(self, outputs):
        
        # log performance
        accuracy, precision, recall, f1 = self.performance(self.counts_all)
        self.log('acc_train_epoch', accuracy, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        self.log('prec_train_epoch', precision, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        self.log('rec_train_epoch', recall, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        self.log('f1_train_epoch', f1, on_step=True, on_epoch=False, prog_bar=False, logger=True)

        # reset performance measurements
        self.loss_interval = torch.tensor(0).type(torch.float32)
        self.counts = np.zeros(4) + 1e-10
        self.counts_all = np.zeros(4)
        
    def validation_step(self, batch, batch_idx):

        # Compute the forward path
        output = self.forward(batch)

        # Compute the cross-entropy loss
        loss = self.criterion(output, batch['labels'])
        
        # Save log
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
    def validation_epoch_end(self, outputs):
        
        # log performance
        accuracy, precision, recall, f1 = self.performance(self.counts_all)
        self.log('acc_val_epoch', accuracy, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        self.log('prec_val_epoch', precision, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        self.log('rec_val_epoch', recall, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        self.log('f1_val_epoch', f1, on_step=True, on_epoch=False, prog_bar=False, logger=True)

        self.counts_all = np.zeros(4)

        
    def configure_optimizers(self):
        
        # Using Adam optimizer
        # Configuration as in the paper: lr=3e-6, beta1=0.9, beta2=0.999, L2_weight_decay=0.01
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=3e-6, weight_decay=0.01)
        
        # Configuration as in the paper: learning rate warmup over the first 10,000 steps
        if self.linear_scheduler_steps is None:
            return [optimizer]

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
        # Get outputs and labels
        output = np.argmax(output.cpu().detach().clone().numpy(), axis=1)
        labels = np.argmax(labels.cpu().detach().clone().numpy(), axis=1)
        
        # Calculate confusion matrix
        tp = np.sum((output == 1) * (labels == 1))
        fp = np.sum((output == 1) * (labels == 0))
        fn = np.sum((output == 0) * (labels == 1))
        tn = np.sum((output == 0) * (labels == 0))
        
        # Accumulate the confusion matrix
        self.counts += np.array([tp, fp, fn, tn])
        self.counts_all += np.array([tp, fp, fn, tn])
        
        
    def performance(self, counts):
        """
        Calculate accuracy, precision, recall, and f1 score using a accumulated confusion matrix
        """
        tp, fp, fn, tn = counts
        
        accuracy = (tp + tn) / (tp + fp + tn + fn + 1e-10)
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        return accuracy, precision, recall, f1


