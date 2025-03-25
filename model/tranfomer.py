import torch
import torch.nn as nn

from mult_head import MultiHeadSelfAttention
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout=0.1, forward_expansion=4):
        super().__init__()
        self.attention = MultiHeadSelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attention = self.attention(x, mask)
        x = self.norm1(attention + x)
        forward = self.feed_forward(x)
        return self.norm2(forward + x)