import torch
from transformers import BertModel, get_linear_schedule_with_warmup
import pytorch_lightning as pl
import numpy as np


class ColBERT(pl.LightningModule):
    def __init__(self, linear_scheduler_steps, B, model_name='bert-base-uncased'):
        super().__init__()

        self.query_encoder = BertModel.from_pretrained(model_name)
        self.passage_encoder = BertModel.from_pretrained(model_name)


        self.linear_scheduler_steps = linear_scheduler_steps
        self.B = B

        #self.labels = torch.cat((torch.eye(B), torch.zeros(B, B)), 1)
        self.softmax = torch.nn.Softmax(dim=1)
        self.labels = torch.arange(B)

        # Monitor training steps for logging purpose
        self.steps = torch.tensor(0).type(torch.float32)

        # counts and step intervals for performance logging
        self.counts = np.zeros(4) # n_correct, n_total, n_correct_epoch, n_total_epoch
        self.loss_interval = torch.tensor(0).type(torch.float32)
        self.measure_steps = 1000

    def forward(self, x):
        """
        :param x: x['input_ids'] is the input to the BERT
        """

        #print(x['query_input_ids'].shape, x['query_attention_mask'].shape, x['positive_input_ids'].shape,
        #      x['positive_attention_mask'].shape, x['negative_input_ids'].shape, x['negative_attention_mask'].shape)

        query_vectors = self.query_encoder(
            input_ids=x['query_input_ids'],
            attention_mask=x['query_attention_mask']
        ).last_hidden_state    # shape: (B, longest_len_query, d)

        positive_vectors = self.passage_encoder(
            input_ids=x['positive_input_ids'],
            attention_mask=x['positive_attention_mask']
        ).last_hidden_state    # shape: (B, longest_len_passage_pos, d)

        negative_vectors = self.passage_encoder(
            input_ids=x['negative_input_ids'],
            attention_mask=x['negative_attention_mask']
        ).last_hidden_state    # shape: (B, longest_len_passage_neg, d)

        # similarity scores
        pos_scores = late_interaction(query_vectors, positive_vectors) # shape: (B, B)
        neg_scores = late_interaction(query_vectors, negative_vectors) # shape: (B, B) 
        scores = torch.cat((pos_scores, neg_vectors), 0) # shape: (2B, len_passage, d)
        return scores

    def late_interaction(self, query_vectors, passage_vectors):
        scores = torch.zeros(query_vectors.shape[0], passage_vectors.shape[0])

        for i in range(query_vectors.shape[0]):
            for j in range(passage_vectors.shape[0]):
                scores[i, j] = torch.matmul(query_vectors[i], passage_vectors[j].T).max(1).values.sum()

        return scores

    def loss(self, scores):
        scores /= 10  # Divided by 10 to avoid having 0.0000e+00 for some of the softmax result

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
        scores = scores.cpu()
        loss = self.loss(scores)

        self.add_counts(scores)
        self.loss_interval += loss

        # Save log
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('steps', self.steps, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if self.steps % self.measure_steps == 0:
            accuracy = self.counts[0] / self.counts[1]
            self.loss_interval /= self.measure_steps
            self.log('acc', accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log('loss_interval', self.loss_interval, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            print(accuracy, self.loss_interval)
            self.counts[:2] = 0
            self.loss_interval = torch.tensor(0).type(torch.float32)

        # Return loss for backpropagation
        return {'loss': loss}

    def train_epoch_end(self, outputs):
        accuracy = self.counts[0] / (self.counts[1] + 1e-10)
        accuracy_epoch = self.counts[2] / self.counts[3]
        self.log('acc', accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('acc_end', accuracy_epoch, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.counts = np.zeros(4)

    def validation_step(self, batch, batch_idx):

        # Compute the forward path
        scores = self.forward(batch)
        loss = self.loss(scores)

        self.add_counts(scores)

        # Save log
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if self.steps % self.measure_steps == 0:
            accuracy = self.counts[0] / self.counts[1]
            self.log('val_acc', accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.counts[:2] = 0

    def validation_epoch_end(self, outputs):
        accuracy = self.counts[0] / (self.counts[1] + 1e-10)
        accuracy_epoch = self.counts[2] / self.counts[3]
        self.log('val_acc', accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc_end', accuracy_epoch, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.counts = np.zeros(4)

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

    def add_counts(self, scores):

        prediction = torch.argmax(scores, axis=1)
        n_correct = torch.sum((prediction == self.labels).type(torch.int32)).tolist()
        n_total = scores.shape[0]

        self.counts[[0, 2]] += n_correct
        self.counts[[1, 3]] += n_total



