import torch
import torch.nn as nn


class Meta(nn.Module):
    def __init__(self, hid_dim):
        super(Meta, self).__init__()
        self.input_layer = nn.Linear(1, hid_dim)
        self.hid_layer = nn.Linear(hid_dim, hid_dim)
        self.output_layer = nn.Linear(hid_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, X):
        hid = self.relu(self.input_layer(X))
        hid = self.relu(self.hid_layer(hid))
        output = self.output_layer(hid)
        return output


if __name__ == '__main__':
    meta = Meta(64)
    X = torch.randn(21, 1)
    Y = meta(X)
    print(Y.data)
