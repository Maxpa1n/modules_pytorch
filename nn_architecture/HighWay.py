import torch
import torch.nn as nn


class HighWay(nn.Module):
    def __init__(self, layer_num, hidden_size):
        super().__init__()
        self.n = layer_num
        self.linear = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(self.n)])
        self.gate = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(self.n)])

    def forward(self, x):
        # x[batch_size, seq_len, hidden_size]
        for i in range(self.n):
            gate = torch.sigmoid(self.gate[i](x))
            nonlinear = torch.relu(self.linear[i](x))
            x = gate * nonlinear + (1 - gate) * x
        return x


if __name__ == '__main__':
    highway = HighWay(3, 30)
    x = torch.randn(5, 15, 30)
    out = highway(x)
    print(out.shape)
