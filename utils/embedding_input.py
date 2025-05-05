import torch.nn as nn
from utils.positional_encoding import PositionalEncoding

class InputLayer(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, seq_len: int, dropout: float):
        super(InputLayer, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim = embedding_dim,
            padding_idx = 0
        )
        self.positional_encoding = PositionalEncoding(embedding_dim, seq_len, dropout)

    def forward(self, x):
        embedded = self.embedding(x) # (batch_size, seq_len, embedding_dim)
        pos_encodings = self.positional_encoding(embedded) # (1, seq_len, embedding_dim)
        embedded = embedded + pos_encodings 
        return embedded