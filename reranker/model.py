import torch
from transformers import BertModel
import pytorch_lightning as pl

class Reranker(pl.LightningModule):
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__()

        # Download a pre-trained BERT model
        self.bert = BertModel.from_pretrained(model_name)

        # Linear layer that will use the <CLS> vector as its input
        self.linear = torch.nn.Linear(in_features=768, out_features=2, bias=True)

        # Cross-entropy loss
        self.criterion = torch.nn.CrossEntropyLoss()
        
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
        
        # Compute the forward path
        output = self.forward(batch)

        # Compute the cross-entropy loss
        loss = self.criterion(output, batch['labels'])
        
        # Log loss
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        # Return loss for backpropagation
        return {'loss': loss}
        
    def validation_step(self, batch, batch_idx):

        # Compute the forward path
        output = self.forward(batch)

        # Compute the cross-entropy loss
        loss = self.criterion(output, batch['labels'])
        
        # Log loss
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    
    def configure_optimizers(self):
        
        # Using Adam optimizer as shown in the paper (lr=3e-6, beta1=0.9, beta2=0.999, L2_weight_decay=0.01)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=3e-6, weight_decay=0.01)
        
        return optimizer
