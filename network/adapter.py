import torch

import torch.nn as nn
import torch.nn.functional as F   
    
    
class MultiLayerAdapter(nn.Module):
    def __init__(self, 
                 input_dim=1024,
                 hidden_dim=512,
                 batch_norm=True,
                 residual_connection=False,
                 queries=None,
                 values=None,
                 temperature=100.,
                 return_hidden_by_default=True,
                 num_encoder_layers=1,
                 activation='relu',
                 softmax_output=False):
        super().__init__()
        if activation == 'relu':
            activation = nn.ReLU(True)
        else:
            batch_norm = False
            activation = nn.GELU()
        self.BatchNorm1d = (nn.BatchNorm1d(hidden_dim) if batch_norm 
                            else nn.Identity())
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=not batch_norm),
            self.BatchNorm1d,
            activation,
            nn.Linear(hidden_dim, input_dim)
        )
        self.residual_connection = residual_connection
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = input_dim
        self.batch_norm = batch_norm
        
        # For classification
        self.queries = queries
        self.values = values
        self.temperature = temperature
        
        self.return_hidden_by_default = return_hidden_by_default
        self.num_encoder_layers = num_encoder_layers
        
        self.softmax_output = softmax_output
    
    def classifier(self, z):
        z = z / z.norm(dim=-1, keepdim=True)
        logits = (self.temperature * (z @ self.queries.T))
        if self.softmax_output:
            return logits.softmax(dim=-1)  # probs
        return logits
    
    def forward(self, z, return_hidden=False):
        z = self.encode(z)  
        y = self.classifier(z)
        if return_hidden or self.return_hidden_by_default:
            return y, z
        else:
            return y       
        
    def encode(self, z):
        for _ in range(self.num_encoder_layers):
            z = self.encode_single_layer(z)
        return z
    
    def encode_single_layer(self, z):
        if self.residual_connection:
            return self.encoder(z) + z
        return self.encoder(z)
    
    def to_device(self, device):
        self.to(device)
        self.queries = self.queries.to(device) 
        
    
class LinearProbe(nn.Module):
    def __init__(self, 
                 input_dim=1024,
                 num_classes=2,
                 return_hidden_by_default=False,
                 bias=True):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes,
                                    bias=bias)
        self.input_dim = input_dim
        self.output_dim = input_dim
        
        self.return_hidden_by_default = return_hidden_by_default
    
    def forward(self, z, return_hidden=False):
        y = self.classifier(z)
        if return_hidden or self.return_hidden_by_default:
            return y, z
        else:
            return y
        
    def to_device(self, device):
        self.to(device)
        
    def encode(self, z):
        return z
    
