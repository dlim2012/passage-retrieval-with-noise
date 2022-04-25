from transformers import BertModel
import torch

class DPR(torch.nn.Module):
    def __init__(self, B, device):
        super(DPR, self).__init__()
        self.query_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.passage_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.B = B
        self.I = torch.eye(self.B)

    def forward(self, batch):

        # Calculate the <CLS> vector of queries
        query_vectors = self.query_encoder(**batch['queries']).last_hidden_state[:, 0, :]    # shape: (B, d)

        # Calculate the <CLS> vectors of positive passages
        positive_vectors = self.passage_encoder(**batch['positives']).last_hidden_state[:, 0, :]    # shape: (B, d)

        # Calculate the <CLS> vectors of negative passages
        negative_vectors = self.passage_encoder(**batch['negatives']).last_hidden_state[:, 0, :]    # shape: (B, d)

        # Concatenate the passage vectors
        passage_vectors = torch.cat((positive_vectors, negative_vectors), 0) # shape: (2B, d)

        # Calculate the similarity scores
        scores = torch.matmul(query_vectors, passage_vectors.T) # shape: (B, 2B)

        return scores

    def loss(self, scores):

        # Change data type to avoid having 0.0000e+00 for some of the softmax result
        scores = scores.type(torch.float64)

        # Calculate the probability of predicting correctly
        x = torch.sum(torch.softmax(scores, dim=1)[:, :self.B] * self.I, axis=1)

        # Calculate the negative log likelihood
        x = -1 * torch.log(x)

        # Calculate the mean loss
        loss = torch.mean(x)

        return loss