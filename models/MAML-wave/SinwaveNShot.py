import torch
import numpy as np
from sinwave import SinWave


# [num_task, num_sample, 1]
class SinwaveNShot:
    def __init__(self, all_numbers_class, batch_size, n_way, k_shot, k_queue):
        '''
        :param all_numbers_class: generate number class ,param A&B
        :param batch_size:task number
        :param n_way:
        :param k_shot:
        :param k_queue:

        '''
        # amplitude varies A
        # phase varies B
        self.A, self.B = self.get_func_param_set(A_low=0.1, A_high=5.0,
                                                 B_low=0.0, B_high=np.pi,
                                                 num_set=100, num_task=all_numbers_class)
        self.X = np.arange(-5, 5, 0.02)
        self.data = SinWave(self.A, self.B, self.X)

        # print('A length:{}\n B length:{}'.format(len(self.A), len(self.B)))
        # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        # print('DATA :{}'.format(self.data[45]))
        # print('len data:{}'.format(len(self.data)))

    def get_func_param_set(self, A_low, A_high, B_low, B_high, num_set, num_task):
        A_step = (A_high - A_low) / num_set
        B_step = (B_high - B_low) / num_set
        A = np.arange(A_low, A_high, A_step)
        B = np.arange(B_low, B_high, B_step)
        return np.random.choice(A, num_task), np.random.choice(B, num_task)


if __name__ == '__main__':
    nshot = SinwaveNShot(2000, 1, 1, 1, 1)
