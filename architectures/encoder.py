import torch.nn as nn

class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim = embed_dim,
            num_heads = num_heads,
            batch_first= True
        )
        self.ffn = nn.Sequential(
            nn.Linear(in_features = embed_dim, out_features= ff_dim, bias = True),
            nn.ReLU(),
            nn.Linear(in_features = ff_dim, out_features=embed_dim, bias = True)
        )
        self.layernorm_1 = nn.LayerNorm(normalized_shape=embed_dim, eps=1e-6)
        self.layernorm_2 = nn.LayerNorm(normalized_shape=embed_dim, eps=1e-6)
        self.dropout_1 = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)

    def forward(self, src, src_mask = None, src_key_padding_mask = None, is_casual = None):
        attn_output, _ = self.self_attn(src, src, src, attn_mask = src_mask,
                                            key_padding_mask = src_key_padding_mask)
        attn_output = self.dropout_1(attn_output)
        out_1 = self.layernorm_1(src + attn_output)
        ffn_output = self.ffn(out_1)
        ffn_output = self.dropout_2(ffn_output)
        out_2 = self.layernorm_2(out_1 + ffn_output)
        return out_2
        
class EncoderBlock(nn.Module):
    def __init__(self, encoder_layer: nn.Module, num_layers: int, norm: nn.Module = None):
        super(EncoderBlock, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.norm = norm

    def forward(self, src, mask = None, src_key_padding_mask = None):
        for layer in self.layers:
            src = layer(src, mask, src_key_padding_mask)
        if self.norm:
            src = self.norm(src)
        return src # (batch_size, src_seq_len, embed_dim)
    