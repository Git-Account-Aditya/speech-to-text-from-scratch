import torch
import torch.nn as nn

class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model=512, n_heads=8, ff_dim=2048, dropout=0.1):
        super().__init__()
        
        # 1. Masked self-attention (over output tokens)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        # 2. Cross-attention (queries from decoder, keys/values from encoder)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        # 3. Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_out, tgt_mask=None, tgt_key_padding_mask=None):
        # x:           [batch, tgt_len, d_model]  — decoder tokens so far
        # encoder_out: [batch, src_len, d_model]  — H from encoder
        
        # 1. Masked self-attention (pre-norm)
        x_norm = self.norm1(x)
        self_attn_out, _ = self.self_attn(
            query=x_norm, key=x_norm, value=x_norm,
            attn_mask=tgt_mask,                        # causal mask
            key_padding_mask=tgt_key_padding_mask
        )
        x = x + self.dropout(self_attn_out)
        
        # 2. Cross-attention (pre-norm)
        x_norm = self.norm2(x)
        cross_attn_out, _ = self.cross_attn(
            query=x_norm,                              # from decoder
            key=encoder_out,                           # from encoder
            value=encoder_out                          # from encoder
        )
        x = x + self.dropout(cross_attn_out)
        
        # 3. Feed-forward (pre-norm)
        x = x + self.dropout(self.ff(self.norm3(x)))
        return x


def make_causal_mask(size, device):
    # Creates an [size x size] mask
    # Upper triangle (above diagonal) = -inf, rest = 0
    mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_heads=8,
                 ff_dim=2048, n_layers=6, max_len=448, dropout=0.1):
        super().__init__()
        
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        
        self.blocks = nn.ModuleList([
            TransformerDecoderBlock(d_model, n_heads, ff_dim, dropout)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights: embedding and output projection share the same matrix
        # (common trick — reduces parameters, often improves performance)
        self.proj.weight = self.token_emb.weight
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, tokens, encoder_out):        
        B, T = tokens.shape
        device = tokens.device
        
        # Embed tokens + positions
        positions = torch.arange(T, device=device)
        x = self.dropout(self.token_emb(tokens) + self.pos_emb(positions))
        
        # Causal mask
        tgt_mask = make_causal_mask(T, device)
        
        # Pass through N decoder blocks
        for block in self.blocks:
            x = block(x, encoder_out, tgt_mask=tgt_mask)
        
        x = self.norm(x)
        
        # Project to vocab — logits over all possible tokens
        logits = self.proj(x)   # [batch, tgt_len, vocab_size]
        return logits