from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt


class SinWave(Dataset):
    def __init__(self, A, B, X):
        self.X = X
        self.Y = A * torch.sin(X + B)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.X)


if __name__ == '__main__':
    x = torch.arange(-5, 5, 0.02).float()
    plt.xlim(-5, 5)
    plt.ylim(-9, 9)
    sindataset = SinWave(4, 0.5, x)
    x, y = sindataset[:]
    print('INPUT:{},\nOUTPUT:{},\nLENGTH:{},'.format(x.numpy(), y.numpy(), len(sindataset)))
    plt.plot(x.numpy(), y.numpy(), label='one sampler')
    plt.xlabel('input x')
    plt.ylabel('output y')
    plt.show()
