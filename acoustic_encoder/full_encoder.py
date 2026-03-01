from .downsampler import DownSamplingNetwork, DownSamplerBlock
from .transformer_with_attn import SinusoidalPositionalEncoding, TransformerEncoderBlock

import torch
import torch.nn as nn 

class AcousticEncoder(nn.Module):
    def __init__(self,
                n_mels=80,
                d_model=512,
                n_heads=8, 
                ff_dim=2048,
                strides=[4, 4, 6],
                n_layers=6,
                max_len=1500,
                dropout=0.1):
        super().__init__()
        
        # create downsampler using convolutional layers
        self.downsampling_network = DownSamplingNetwork(
            embedding_dims=d_model,
            hidden_dims=d_model,
            in_channel=n_mels,
            strides=strides
        )

        # create positional sinusoidal encoding 
        self.sinusoidal_positional_encoding = SinusoidalPositionalEncoding(
            d_model = d_model,
            maxlen = max_len
        )

        self.layers = nn.ModuleList()
        
        for i in range(n_layers):
            self.layers.append(
                TransformerEncoderBlock(
                    d_model = d_model,
                    n_heads = n_heads,
                    ff_dim = ff_dim,
                    dropout = dropout
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, time, n_mels) or (batch, n_mels, time) depending on DownSamplingNetwork.
        Ensure shape matches downstream modules.
        """
        # basic shape check (adjust/remove if your downsampler expects channel-first)
        if x.dim() != 3:
            raise ValueError(f"Expected input tensor of dim 3, got {x.dim()}")

        x = self.downsampling_network(x)
        x = self.sinusoidal_positional_encoding(x)
        for layer in self.layers:
            x = layer(x)
        return x