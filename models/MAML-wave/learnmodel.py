import torch
import torch.nn as nn
from    torch.nn import functional as F


class Compute(nn.Module):
    def __init__(self, hid_dim):
        super(Compute, self).__init__()
        self.input_layer = nn.Linear(1, hid_dim)
        self.hid_layer = nn.Linear(hid_dim, hid_dim)
        self.output_layer = nn.Linear(hid_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, X):
        hid = self.relu(self.input_layer(X))
        hid = self.relu(self.hid_layer(hid))
        output = self.output_layer(hid)
        return output


class Learner(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.com = Compute(hid_dim)
        # self.com_temp = Compute(hid_dim)

    def forward(self, x, com=None):
        if com is not None:
            x = F.linear(x, com[0], com[1])
            x = F.linear(x, com[2], com[3])
            y = F.linear(x, com[4], com[5])
            return y
        else:
            y = self.com(x)
            return y


if __name__ == '__main__':
    x = torch.randn(25, 1)
    com = Compute(64)
    lea = Learner(64)
    para_dic = dict()
    for key, val in lea.named_parameters():
        para_dic[key] = val
    print(key, val.grad)
    y = lea(x, com)
    print('output shape {}'.format(y.shape))
