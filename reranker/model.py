import torch
from transformers import BertForNextSentencePrediction

class Reranker(torch.nn.Module):
    def __init__(self):
        super(Reranker, self).__init__()
        self.encoder = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')

    def forward(self, inputs):
        outputs = self.encoder(**inputs)

        return outputs

