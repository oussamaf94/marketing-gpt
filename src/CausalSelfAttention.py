import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config["n_embed"] % config["n_heads"] == 0

        self.c_attn = nn.Linear(config["n_embed"], 3 * config["n_embed"])
        self.c_proj = nn.Linear(config["n_embed"], config["n_embed"])
        self.attn_dropout = nn.Dropout(config["dropout"])
        self.resid_dropout = nn.Dropout(config["dropout"])

        # Assign config values to instance variables
        self.n_heads = config["n_heads"]
        self.n_embed = config["n_embed"]
        self.seg_length = config["seg_length"]
        self.message_length = config["message_length"]
        self.block_size = config["block_size"]

        # Fixed causal mask for training
        total_length = self.seg_length + self.message_length
        mask = torch.zeros(total_length, total_length)

        # Full 1s for the segment portion
        mask[:self.seg_length, :self.seg_length] = 1

        # Lower triangular 1s for the message portion
        for i in range(self.seg_length, total_length):
            mask[i, :i + 1] = 1

        # Reshape to 4D for attention (1, 1, total_length, total_length)
        mask = mask.view(1, 1, total_length, total_length)

        # Register the mask as a buffer
        self.register_buffer("fixed_causal_mask", mask)

    def forward(self, x):
        B, T, _ = x.size()

        # Linear projections for query, key, value
        q, k, v = self.c_attn(x).split(self.n_embed, dim=2)

        # Reshape projections for multi-head attention
        k = k.view(B, T, self.n_heads, self.n_embed // self.n_heads).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_heads, self.n_embed // self.n_heads).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_heads, self.n_embed // self.n_heads).transpose(1, 2)  # (B, nh, T, hs)

        # Compute attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # Apply the fixed causal mask during training or slice for generation
        mask = self.fixed_causal_mask[:, :, :T, :T]  # Dynamically slice the mask for the current sequence length
        att = att.masked_fill(mask == 0, float('-inf'))

        # Compute attention probabilities and apply dropout
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Weighted sum of values
        y = att @ v

        # Reshape and project back to original embedding size
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_embed)
        y = self.resid_dropout(self.c_proj(y))

        return y
