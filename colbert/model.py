import torch
from transformers import BertModel, get_linear_schedule_with_warmup
import pytorch_lightning as pl
import numpy as np

class ColBERT(pl.LightningModule):
    def __init__(self,
                 linear_scheduler_steps,
                 B,
                 tokenizer,
                 vector_size=128,
                 lr=3e-6,
                 measure_steps=250,
                 model_name='bert-base-uncased',
                 early_stop=True):

        super().__init__()
        self.save_hyperparameters()

        self.encoder = BertModel.from_pretrained(model_name)
        self.encoder.resize_token_embeddings(len(tokenizer))

        self.linear_scheduler_steps = linear_scheduler_steps
        self.B = B
        self.tokenizer = tokenizer
        self.lr = lr

        self.labels = torch.cat((torch.eye(B), torch.zeros(B, B)), 1)
        self.I = torch.eye(self.B)
        self.linear = torch.nn.Linear(in_features=768, out_features=vector_size, bias=True)

        self.softmax = torch.nn.Softmax(dim=1)
        self.labels = torch.arange(B)

        # Monitor training steps for logging purpose
        self.steps = torch.tensor(0).type(torch.float32)

        # counts and step intervals for performance logging
        self.counts = np.zeros(4) # n_correct, n_total, n_correct_epoch, n_total_epoch
        self.loss_interval = 0
        self.measure_steps = measure_steps

        # For early stop
        self.early_stop = early_stop
        self.prev_loss_interval = 0
        self.min_loss_interval = float('inf')
        self.early_stop_count = 0


    def forward(self, x):
        """
        :param x: x['input_ids'] is the input to the BERT
        """

        #print(x['query_input_ids'].shape, x['query_attention_mask'].shape, x['positive_input_ids'].shape,
        #      x['positive_attention_mask'].shape, x['negative_input_ids'].shape, x['negative_attention_mask'].shape)

        query_vectors = torch.nn.functional.normalize(self.linear(self.encoder(
            input_ids=x['query_input_ids'],
            attention_mask=x['query_attention_mask']
        ).last_hidden_state))    # shape: (B, q, d)

        positive_vectors = torch.nn.functional.normalize(self.linear(self.encoder(
            input_ids=x['positive_input_ids'],
            attention_mask=x['positive_attention_mask']
        ).last_hidden_state))   # shape: (B, p1, d)

        negative_vectors = torch.nn.functional.normalize(self.linear(self.encoder(
            input_ids=x['negative_input_ids'],
            attention_mask=x['negative_attention_mask']
        ).last_hidden_state))    # shape: (B, p2, d)

        # Similarity scores
        pos_scores = self.late_interaction(query_vectors, positive_vectors, x['positive_punc_mask']) # shape: (B, B)
        neg_scores = self.late_interaction(query_vectors, negative_vectors, x['negative_punc_mask']) # shape: (B, B)
        scores = torch.cat((pos_scores, neg_scores), 1) # shape: (B, 2B)
        return scores

    def late_interaction(self, query_vectors, passage_vectors, punc_mask):
        scores = torch.zeros(query_vectors.shape[0], passage_vectors.shape[0])

        #for i in range(query_vectors.shape[0]):
        #    for j in range(passage_vectors.shape[0]):
        #        scores[i, j] = torch.matmul(query_vectors[i], passage_vectors[j][punc_mask[j]].T).max(1).values.sum()

        for i in range(query_vectors.shape[0]):
            # Calculate cosine similarity scores between all query vectors and all passage vectors
            similarities = torch.matmul(query_vectors[i], torch.permute(passage_vectors, (0, 2, 1))).cpu()
            # shape: (B, q, p)
            for j in range(passage_vectors.shape[0]):
                # Filter punctuation vectors from passage vectors
                # Calculate the maximum similarity for each query vector and sum the results
                scores[i, j] = similarities[j][:, punc_mask[j]].max(1).values.sum()
                # max shape: (B, q)
                # shape: (B,)

        return scores

    def loss(self, scores):
        # Change data type to avoid having 0.0000e+00 for some of the softmax results
        scores = scores.type(torch.float64)

        # Calculate the probability of predicting correctly
        x = torch.sum(torch.softmax(scores, dim=1)[:, :self.B] * self.I, axis=1)

        # Calculate the negative log likelihood
        x = -1 * torch.log(x)

        # Calculate the mean loss
        loss = torch.mean(x)

        return loss


    def on_train_batch_start(self, batch, batch_idx):
        if self.early_stop and self.steps % self.measure_steps == 0:
            if self.prev_loss_interval > self.min_loss_interval * 1.2:
                self.early_stop_count += 1
                if self.early_stop_count == 3:
                    return -1
            else:
                self.early_stop_count = 0

    def training_step(self, batch, batch_idx):
        self.steps += 1

        # Compute the forward path
        scores = self.forward(batch)
        loss = self.loss(scores)

        # Add results to measurements
        self.add_counts(scores)
        self.loss_interval += loss.tolist()

        # Save log
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('steps', self.steps, on_step=True, on_epoch=False, prog_bar=False, logger=True)

        if self.steps % self.measure_steps == 0:
            # Log performance periodically
            accuracy = self.counts[0] / self.counts[1]
            self.loss_interval /= self.measure_steps
            self.prev_loss_interval = self.loss_interval
            self.min_loss_interval = min(self.min_loss_interval, self.loss_interval)


            self.log('acc', accuracy, on_step=True, on_epoch=False, prog_bar=True, logger=True)
            self.log('loss_interval', self.loss_interval, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            #print(accuracy, self.loss_interval)
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
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr,
                                     weight_decay=0.01)

        if not self.linear_scheduler_steps:
            return [optimizer]

        # learning rate warmup: ~self.linear_scheduler_steps[0] steps
        # learning rate linear decay: self.linear_scheduler_steps[1] steps
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

    def add_counts(self, scores):
        """
        Calculate the number of correct predictions and add to measurements
        """
        prediction = torch.argmax(scores, axis=1).cpu()
        n_correct = torch.sum((prediction == self.labels).type(torch.int32)).tolist()
        n_total = scores.shape[0]

        self.counts[[0, 2]] += n_correct
        self.counts[[1, 3]] += n_total


