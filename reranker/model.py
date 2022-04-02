import torch
from transformers import BertModel
import pytorch_lightning as pl

class Reranker(pl.LightningModule):
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.linear = torch.nn.Linear(in_features=768, out_features=2, bias=True)
        self.criterion = torch.nn.CrossEntropyLoss()
        
    def forward(self, x):
        
        input_ids = x['input_ids']
        output = self.bert(input_ids)
        output = output.last_hidden_state
        output = self.linear(output)
        return output
    
    def training_step(self, batch, batch_idx):
        output = self.forward(batch)
        loss = self.criterion(output, batch['labels'])
        
        
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return {'loss': loss}
        
    def validation_step(self, batch, batch_idx):
        output = self.forward(batch)
        loss = self.criterion(output, batch['labels'])
        
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return {'loss': loss}
    
    def configure_optimizers(self):
        
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=3e-6, weight_decay=0.01)
        
        return optimizer
