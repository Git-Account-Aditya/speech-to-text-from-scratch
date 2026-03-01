import torch
import torch.nn as nn

class DownSamplerBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel_size=3):
        super(DownSamplerBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding="same")
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=stride,
                               padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.conv2(out) + self.residual(x)
        return out

class DownSamplingNetwork(nn.Module):
    def __init__(
        self,
        embedding_dims: int = 128,
        hidden_dims: int = 64, 
        in_channel: int = 1,
        initial_mean_pooling_kernel_size: int = 2,
        strides = [4, 4, 6, 6]
    ):
        super().__init__()
        self.layers = nn.ModuleList()

        self.mean_pooling_layer = nn.AvgPool1d(initial_mean_pooling_kernel_size)

        for i in range(len(strides)):
            self.layers.append(
                DownSamplerBlock(
                    in_channels=hidden_dims if i > 0 else in_channel,
                    out_channels=hidden_dims,
                    stride=strides[i],
                )
            )

        self.final_conv = nn.Conv1d(hidden_dims, embedding_dims, kernel_size=4, padding="same")
    
    def forward(self, x):
        x = self.mean_pooling_layer(x)
        for layer in self.layers:
            x = layer(x)
        x = self.final_conv(x)
        x = x.transpose(1, 2)
        return x

if __name__ == "__main__":
    batch_size = 2
    input_embedding_dims = 128
    seq_len = 16000
    x = torch.randn(batch_size, 1, seq_len)
    model = DownSamplingNetwork(embedding_dims=input_embedding_dims, strides=[2, 4])
    print(model(x).shape)