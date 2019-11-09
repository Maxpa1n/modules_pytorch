import torch
import numpy as np
from sinwave import SinWave
import os


# [num_task, num_sample, 1]
class SinwaveNShot:
    def __init__(self, all_numbers_class, batch_size, n_way, k_shot, k_query, root):
        '''
        :param all_numbers_class: generate number class ,param A&B
        :param batch_size:task number
        :param n_way:
        :param k_shot:
        :param k_queue:
        '''

        self.root = root
        # amplitude varies A
        # phase varies B
        if not os.path.isfile(os.path.join(root, 'wave_data.npy')):
            self.A, self.B = self.get_func_param_set(A_low=0.1, A_high=5.0,
                                                     B_low=0.0, B_high=np.pi,
                                                     num_set=100, num_task=all_numbers_class)
            self.X = np.arange(-5, 5, 0.02)
            self.data = SinWave(self.A, self.B, self.X)

            self.data_dict = []
            for x, y, a, b in self.data:
                temp = {}
                temp['A_B'] = (a, b)
                temp['input'] = x
                temp['output'] = y
                self.data_dict.append(temp)
            np.save(os.path.join(root, 'wave_data.npy'), self.data)
            self.data_dict = np.load(os.path.join(root, 'wave_data.npy'))
            print('created wave data to {}'.format(os.path.join(root, 'wave_data.npy')))
        else:
            self.data_dict = np.load(os.path.join(root, 'wave_data.npy'))
            print('load wave data from {}'.format(os.path.join(root, 'wave_data.npy')))
        self.k_shot = k_shot
        self.n_way = n_way
        self.k_query = k_query
        self.batch_size = batch_size
        self.train, self.test = self.data_dict[:1600], self.data_dict[1600:]
        self.indexes = {"train": 0, "test": 0}
        self.datasets = {"train": self.train, "test": self.test}  # original data cached
        print("DB: train", self.train.shape, "test", self.test.shape)
        self.datasets_cache = {"train": self.load_cache_data(self.datasets["train"]),  # current epoch data cached
                               "test": self.load_cache_data(self.datasets["test"])}
        # train_data = self.load_cache_data(self.train)

    def get_func_param_set(self, A_low, A_high, B_low, B_high, num_set, num_task):
        A_step = (A_high - A_low) / num_set
        B_step = (B_high - B_low) / num_set
        A = np.arange(A_low, A_high, A_step)
        B = np.arange(B_low, B_high, B_step)
        return np.random.choice(A, num_task), np.random.choice(B, num_task)

    def next(self, mode='train'):
        if self.indexes[mode] >= len(self.datasets_cache[mode]):
            self.indexes[mode] = 0
            self.datasets_cache[mode] = self.load_data_cache(self.datasets[mode])

        next_batch = self.datasets_cache[mode][self.indexes[mode]]
        self.indexes[mode] += 1

        return next_batch

    def load_cache_data(self, dataset):
        # dataset [num_task, 4]  ->[batch,n_way,k_shot]
        data_cache = []
        all_task = dataset.shape[0]
        all_sample = dataset[0][0].shape[0]

        for sample in range(10):  # epoch
            x_spts, y_spts, x_qrys, y_qrys, param_spts, param_qrys = [], [], [], [], [], []
            for _ in range(self.batch_size):
                x_spt, y_spt, x_qry, y_qry = [], [], [], []
                select_way = np.random.choice(all_task, self.n_way, False)
                select_shot = np.random.choice(all_sample, self.k_shot, False)
                select_query = np.random.choice(all_sample, self.k_query, False)
                way = dataset[select_way]
                x = way[:, 0]
                y = way[:, 1]
                a = way[:, 2]
                b = way[:, 3]
                param = [(i, j) for i, j in zip(a, b)]
                param_spt = []
                param_qry = []
                for pa in param:
                    for _ in range(self.k_shot):
                        param_spt.append(pa)
                    for _ in range(self.k_query):
                        param_qry.append(pa)
                x_spt.append([list(i[select_shot]) for i in x])
                y_spt.append([list(i[select_shot]) for i in y])
                x_qry.append([list(i[select_query]) for i in x])
                y_qry.append([list(i[select_query]) for i in y])
                perm = np.random.permutation(self.n_way * self.k_shot)
                x_spt = np.array(x_spt).reshape(self.n_way * self.k_shot, 1)[perm]
                y_spt = np.array(y_spt).reshape(self.n_way * self.k_shot, 1)[perm]
                param_spt = np.array(param_spt).reshape(self.n_way * self.k_shot, 2)[perm]
                perm = np.random.permutation(self.n_way * self.k_query)
                x_qry = np.array(x_qry).reshape(self.n_way * self.k_query, 1)[perm]
                y_qry = np.array(y_qry).reshape(self.n_way * self.k_query, 1)[perm]
                param_qry = np.array(param_qry).reshape(self.n_way * self.k_query, 2)[perm]

                x_spts.append(x_spt)
                y_spts.append(y_spt)
                x_qrys.append(x_qry)
                y_qrys.append(y_qry)
                param_spts.append(param_spt)
                param_qrys.append(param_qry)

            x_spts = np.array(x_spts).astype(np.float32).reshape(self.batch_size, self.n_way * self.k_shot, 1)
            y_spts = np.array(y_spts).astype(np.float32).reshape(self.batch_size, self.n_way * self.k_shot, 1)
            param_spts = np.array(param_spts).astype(np.float32).reshape(self.batch_size, self.n_way * self.k_shot, 2)

            x_qrys = np.array(x_qrys).astype(np.float32).reshape(self.batch_size, self.n_way * self.k_query, 1)
            y_qrys = np.array(y_qrys).astype(np.float32).reshape(self.batch_size, self.n_way * self.k_query, 1)
            param_qrys = np.array(param_qrys).astype(np.float32).reshape(self.batch_size, self.n_way * self.k_query, 2)

            data_cache.append([x_spts, y_spts, x_qrys, y_qrys, param_spts, param_qrys])

        return data_cache


if __name__ == '__main__':
    nshot = SinwaveNShot(2000, 20, 5, 5, 15, 'data')
    x_spt, y_spt, x_qry, y_qry, param_spt, param_qry = nshot.next('train')
    print('x_spt shape: {}'.format(x_spt.shape))
    print('y_spt shape: {}'.format(y_spt.shape))
    print('param_spt shape {}'.format(param_spt.shape))
    print('x_qry shape: {}'.format(x_qry.shape))
