import torch.nn as nn

class DecoderLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super(DecoderLayer, self).__init__()
        # Self-attention Layer
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Cross-attention Layer
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.layernorm_1 = nn.LayerNorm(embed_dim, eps = 1e-6)
        self.layernorm_2 = nn.LayerNorm(embed_dim, eps = 1e-6)
        self.layernorm_3 = nn.LayerNorm(embed_dim, eps = 1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask = None, memory_mask = None, tgt_key_padding_mask = None, memory_key_padding_mask = None):
        # Self-attention with skip connection and layer normalization
        attn_output1, _ = self.self_attn(tgt, tgt, tgt, attn_mask = tgt_mask, key_padding_mask = tgt_key_padding_mask
        )
        attn_output1 = self.dropout_1(attn_output1)
        out1 = self.layernorm_1(tgt + attn_output1)

        # Cross-attention with skip connection and layer normalization
        attn_output2, _ = self.cross_attn(out1, memory, memory, attn_mask = memory_mask, tgt_key_padding_mask = memory_key_padding_mask
        )
        attn_output2 = self.dropout_2(attn_output2)
        out2 = self.layernorm_2(tgt + attn_output2)

        # FFN with skip connection and layer normalization
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout_3(ffn_output)
        out3 = self.layernorm_3(out2 + ffn_output)

        return out3 # (batch_size, tgt_seq_len, embed_dim)
    
class DecoderBlock(nn.Module):
    def __init__(self, decoder_layer: nn.Module, num_layers: int, norm: nn.Module = None):
        super(DecoderBlock, self).__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.norm = norm
    
    def forward(self, tgt, memory, tgt_mask = None, memory_mask = None, tgt_key_padding_mask = None, memory_key_padding_mask = None):
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask = tgt_mask, memory_mask = memory_mask, tgt_key_padding_mask = tgt_key_padding_mask, memory_key_padding_mask = memory_key_padding_mask)
        if self.norm:
            tgt = self.norm(tgt)
        return tgt # (batch_size, tgt_seq_len, embed_dim)
    
          

