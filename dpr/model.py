import torch
from transformers import BertModel, get_linear_schedule_with_warmup
import pytorch_lightning as pl
import numpy as np


class DPR(pl.LightningModule):
    def __init__(self, model_name='bert-base-uncased', B=16, measure_steps=1000, linear_scheduler_steps=None):
        super().__init__()

        # Save information for linear learning rate scheduling and batch size
        self.linear_scheduler_steps = linear_scheduler_steps
        self.B = B

        # Get two pretrained BERT to use as query encoder and passage encoder
        self.query_encoder = BertModel.from_pretrained(model_name)
        self.passage_encoder = BertModel.from_pretrained(model_name)
        
        # Declare a softmax layer
        self.softmax = torch.nn.Softmax(dim=1)
        
        # Correct labels
        self.labels = torch.arange(B)

        # Monitor training steps for logging purpose
        self.steps = torch.tensor(0).type(torch.float32)

        # Counts and step intervals for performance logging
        self.counts = np.zeros(4) # n_correct, n_total, n_correct_epoch, n_total_epoch
        self.loss_interval = torch.tensor(0).type(torch.float32)

        # Performance logging period
        self.measure_steps = measure_steps

    def forward(self, x):

        # Calculate the <CLS> vector of queries
        query_vectors = self.query_encoder(
            input_ids=x['query_input_ids'],
            attention_mask=x['query_attention_mask']
        ).last_hidden_state[:, 0, :]    # shape: (B, d)

        # Calculate the <CLS> vectors of positive passages
        positive_vectors = self.passage_encoder(
            input_ids=x['positive_input_ids'],
            attention_mask=x['positive_attention_mask']
        ).last_hidden_state[:, 0, :]    # shape: (B, d)

        # Calculate the <CLS> vectors of negative passages
        negative_vectors = self.passage_encoder(
            input_ids=x['negative_input_ids'],
            attention_mask=x['negative_attention_mask']
        ).last_hidden_state[:, 0, :]    # shape: (B, d)

        # Concatenate the passage vectors
        passage_vectors = torch.cat((positive_vectors, negative_vectors), 0) # shape: (2B, d)

        # Calculate the similarity scores
        scores = torch.matmul(query_vectors, passage_vectors.T) # shape: (B, 2B)

        return scores

    def loss(self, scores):

        # Scores are divided by 10 to avoid having 0.0000e+00 for some of the softmax result
        scores /= 10  
        #scores = scores.type(torch.float64) # this may work as an alternative

        # Calculate the probability of predicting correctly
        x = torch.sum(torch.softmax(scores, dim=1)[:, :self.B] * torch.eye(self.B), axis=1)

        # Calculate the negative log likelihood
        x = -1 * torch.log(x)

        # Calculate the mean loss
        loss = torch.mean(x)

        return loss


    def training_step(self, batch, batch_idx):
        self.steps += 1

        # Compute the forward path
        scores = self.forward(batch)
        loss = self.loss(scores).cpu()
        scores = scores.cpu()

        # Add results to measurements
        self.add_counts(scores.cpu())
        self.loss_interval += loss

        # Save log
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('steps', self.steps, on_step=True, on_epoch=False, prog_bar=False, logger=True)

        if self.steps % self.measure_steps == 0:
            # Log performance periodically
            accuracy = self.counts[0] / self.counts[1]
            self.loss_interval /= self.measure_steps
            self.log('acc', accuracy, on_step=True, on_epoch=False, prog_bar=True, logger=True)
            self.log('loss_interval', self.loss_interval, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            print(accuracy, self.loss_interval)
            self.counts[:2] = 0
            self.loss_interval = torch.tensor(0).type(torch.float32)

        # Return loss for backpropagation
        return {'loss': loss}

    def train_epoch_end(self, outputs):
        # Log performance at the end of train epoch
        accuracy = self.counts[0] / (self.counts[1] + 1e-10)
        accuracy_epoch = self.counts[2] / self.counts[3]
        self.log('acc', accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('acc_end', accuracy_epoch, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.counts = np.zeros(4)

    def validation_step(self, batch, batch_idx):

        # Compute the forward path
        scores = self.forward(batch)
        loss = self.loss(scores)

        # Save log
        self.log('val_loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)


    def validation_epoch_end(self, outputs):
        # Log performance at the end of validation epoch
        accuracy = self.counts[0] / (self.counts[1] + 1e-10)
        accuracy_epoch = self.counts[2] / self.counts[3]
        self.log('val_acc', accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc_end', accuracy_epoch, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.counts = np.zeros(4)

    def configure_optimizers(self):

        # Using Adam optimizer
        # lr=3e-6, beta1=0.9, beta2=0.999, L2_weight_decay=0.01
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=3e-6, weight_decay=0.01)

        # learning rate warmup: ~self.linear_scheduler_steps[0] steps
        # learning rate linear decay: self.linear_scheduler_steps[1] steps
        if self.linear_scheduler_steps:
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

            return [optimizer], schedulers
        else:
            return [optimizer]

    def add_counts(self, scores):
        """
        Calculate the number of correct predictions and add to measurements
        """
        prediction = torch.argmax(scores, axis=1)
        n_correct = torch.sum((prediction == self.labels).type(torch.int32)).tolist()
        n_total = scores.shape[0]

        self.counts[[0, 2]] += n_correct
        self.counts[[1, 3]] += n_total



