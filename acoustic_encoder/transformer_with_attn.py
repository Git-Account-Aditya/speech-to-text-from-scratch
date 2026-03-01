import numpy as np 
import torch
import torch.nn as nn 

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, maxlen=5_000):
        super().__init__()
        pe  = torch.zeros(maxlen, d_model)
        position = torch.arange(0, maxlen).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) 
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model=512, n_heads=8, ff_dim=2048, dropout=0.1):
        super().__init__()
        self.attn    = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff      = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model)
        )
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.dropout(attn_out) 
        
        ff_out = self.ff(self.norm2(x))
        x = x + self.dropout(ff_out)          
        return x