import torch
from transformers import BertModel, BertForNextSentencePrediction


#################### Model

class Reranker(torch.nn.Module):
    def __init__(self):
        super(Reranker, self).__init__()
        self.encoder = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')

    def forward(self, inputs):
        outputs = self.encoder(**inputs)

        return outputs

