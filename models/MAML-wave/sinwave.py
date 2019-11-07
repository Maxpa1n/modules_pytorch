from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt
import numpy as np


# A [64]
# B [64]
# X [500]
class SinWave(Dataset):
    def __init__(self, A, B, X):
        assert len(A) == len(B), \
            'length of A and B should be equal'
        self.num_sample = X.shape[0]
        self.num_params = A.shape[0]
        A_t = torch.from_numpy(A).unsqueeze(1).repeat(1, self.num_sample)
        B_t = torch.from_numpy(B).unsqueeze(1).repeat(1, self.num_sample)
        self.X = torch.from_numpy(X).unsqueeze(0).repeat(self.num_params, 1)
        self.Y = torch.mul(A_t, torch.sin(self.X + B_t))
        self.Y = self.Y.numpy()
        self.X = self.X.numpy()
        self.A = A
        self.B = B


    def __getitem__(self, index):
        return self.X[index], self.Y[index], self.A[index], self.B[index]

    def __len__(self):
        return self.num_params


if __name__ == '__main__':
    x = np.arange(-5, 5, 0.02)
    A = np.arange(0.1, 5.0, 0.049)[:100]
    B = np.arange(0, np.pi, 0.031415926)[:100]

    sindataset = SinWave(A, B, x)
    x, y, a, b = sindataset[:50]
    print('INPUT SHAPE:{},\n'
          'OUTPUT:{},\n'
          'A:{},\nB:{}'
          .format(x.shape, y.shape, a, b))
    plt.xlim(-5, 5)
    plt.ylim(-9, 9)
    plt.plot(x[44].numpy(), y[44].numpy(), label='one sampler')
    plt.xlabel('input x')
    plt.ylabel('output y')
    plt.show()
