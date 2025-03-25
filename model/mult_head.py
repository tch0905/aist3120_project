import torch
import torch.nn as nn
import numpy as np


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        assert embed_size % heads == 0, "Embedding size must be divisible by number of heads"

        self.heads = heads
        self.head_dim = embed_size // heads

        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x, mask=None):
        B, T, E = x.shape
        Q = self.query(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention, V).transpose(1, 2).contiguous().view(B, T, E)
        return self.fc_out(output)