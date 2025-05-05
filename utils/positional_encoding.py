import torch
import torch.nn as nn
import math 

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super(PositionalEncoding, self).__init__()
        # Model dimension = 512
        self.d_model = d_model
        # Sequence length based on vocab size
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Matrix of shape(seq_len, d_model)
        pe = torch.zeros(self.seq_len, d_model)
        # A Vetor of shape
        position = torch.arange(0, seq_len, dtype = torch.float32).unsqueeze(1) # Shape (seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()*(-math.log(10000.0)/d_model))

        # Sinosoid functions
        ## Even positions
        pe[:,0::2] = torch.sin(position * div_term) # Shape(seq_len, d_model/2)
        pe[:,1::2] = torch.cos(position * div_term) # Shape(seq_len, d_model/2)
 
        pe = pe.unsqueeze(0) # Shape(1, seq.len, d_model)
        self.register_buffer('pe', pe) # Buffer

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return self.dropout(x)
    
    






