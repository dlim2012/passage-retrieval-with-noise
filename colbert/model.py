from transformers import BertModel
import torch

class ColBERT(torch.nn.Module):
    def __init__(self, tokenizer, B, vector_size, device):
        super(ColBERT, self).__init__()
        self.tokenizer = tokenizer
        self.encoder = BertModel.from_pretrained('bert-base-uncased')
        self.encoder.resize_token_embeddings(len(tokenizer))

        self.linear = torch.nn.Linear(in_features=768, out_features=vector_size, bias=True)
        self.B = B
        self.I = torch.eye(self.B)


    def vector_representation(self, x):

        x = self.encoder(**x).last_hidden_state # Shape: (B, token_len, 768)

        x = self.linear(x) # Shape: (B, token_len, vector_size)

        x = torch.nn.functional.normalize(x) # Normalize to calculate the cosine simliarity

        return x

    def forward(self, batch):

        # Calculate the vector representations of queries
        query_vectors = self.vector_representation(batch['queries'])    # shape: (B, N_q + 3, d)

        # Calculate the vector representations of positive passages
        positive_vectors = self.vector_representation(batch['positives'])  # shape: (B, N_p + 3, d)

        # Calculate the vector representations of negative passages
        negative_vectors = self.vector_representation(batch['negatives'])   # shape: (B, N_p + 3, d)

        # Similarity scores
        pos_scores = self.late_interaction(query_vectors, positive_vectors, batch['positive_filter_indices']) # shape: (B, B)
        neg_scores = self.late_interaction(query_vectors, negative_vectors, batch['negative_filter_indices']) # shape: (B, B)
        scores = torch.cat((pos_scores, neg_scores), 1) # shape: (B, 2B)

        return scores

    def late_interaction(self, query_vectors, passage_vectors, filter_indices):
        scores = torch.zeros(query_vectors.shape[0], passage_vectors.shape[0])

        #for i in range(query_vectors.shape[0]):
        #    for j in range(passage_vectors.shape[0]):
        #        scores[i, j] = torch.matmul(query_vectors[i], passage_vectors[j][punc_mask[j]].T).max(1).values.sum()

        for i in range(query_vectors.shape[0]):
            # Calculate cosine similarity scores between all query vectors and all passage vectors
            similarities = torch.matmul(query_vectors[i], torch.permute(passage_vectors, (0, 2, 1))).cpu()
            # shape: (B, N_q, N_p)
            for j in range(passage_vectors.shape[0]):
                # Filter punctuation vectors from passage vectors
                # Calculate the maximum similarity for each query vector and sum the results
                scores[i, j] = similarities[j][:, filter_indices[j]].max(1).values.sum()

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
