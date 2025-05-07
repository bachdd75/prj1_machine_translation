import torch.nn as nn
import torch
from utils.embedding_input import InputLayer
from .encoder import EncoderLayer, EncoderBlock
from .decoder import DecoderLayer, DecoderBlock

class Transformer(nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, embed_dim: int = 512, src_seq_len: int=100, tgt_seq_len: int=100,
                 num_encoder_layers: int=6, num_decoder_layers: int=6, num_heads: int=8, ff_dim: int=2048, dropout: float=0.1):
        super(Transformer, self).__init__()
        # Embedding and Encoder Input Layer
        self.encoder_input = InputLayer(
            vocab_size = src_vocab_size,
            embedding_dim=embed_dim,
            seq_len=src_seq_len,
            dropout=dropout
        )

        # Encoder Block
        encoder_layer = EncoderLayer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout
        )

        self.encoder = EncoderBlock(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers
        )

        # Embedding and Decoder Input Layer
        self.decoder_input = InputLayer(
            vocab_size=tgt_vocab_size,
            embedding_dim=embed_dim,
            seq_len=tgt_seq_len,
            dropout = dropout
        )

        # Decoder Blcok
        decoder_layer = DecoderLayer(
            embed_dim = embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout
        )

        self.decoder = DecoderBlock(
            decoder_layer = decoder_layer,
            num_layers = num_decoder_layers
        )

        # Final Output Linear Layer
        self.output_linear= nn.Linear(embed_dim, tgt_vocab_size)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: torch.Tensor = None, tgt_mask: torch.Tensor = None,
                src_key_padding_mask: torch.Tensor = None, tgt_key_padding_mask: torch.Tensor = None, memory_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        # Encoder
        encoder_embedded = self.encoder_input(src) # (batch_size, src_seq_len, embed_dim)
        memory = self.encoder(
            encoder_embedded,
            mask = src_mask,
            src_key_padding_mask = src_key_padding_mask,
        ) # (batch_size, src_seq_len, embed_dim)

        # Decoder
        decoder_embedded = self.decoder_input(tgt) # (batch_size, src_seq_len, embed_dim)
        decoder_output = self.decoder(
            decoder_embedded,
            memory,
            tgt_mask = tgt_mask,
            memory_mask = None,
            src_key_padding_mask = src_key_padding_mask,
        ) # (batch_size, src_seq_len, embed_dim)

        # Final Linear Layer
        logits = self.output_linear(decoder_output) # (batch_size, src_seq_len, embed_dim)
        return logits

def generate_square_subsequent_mask(sz: int, diagonal=1) -> torch.Tensor:
    mask = torch.triu(torch.ones((sz, sz), diagonal=1)).bool()
    return mask  # (sz, sz)


        
