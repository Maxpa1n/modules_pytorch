from copy import deepcopy
import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
import numpy as np
from learnmodel import Learner, Compute
from SinwaveNShot import SinwaveNShot


class Meta(nn.Module):
    def __init__(self, hid_dim, meta_lr, update_lr):
        super(Meta, self).__init__()
        self.update_step = 5
        self.update_step_test = 5
        self.inner_com = Compute(hid_dim)
        self.outer_net = Learner(hid_dim)
        self.meta_lr = meta_lr
        self.update_lr = update_lr
        self.outer_optim = optim.Adam(self.outer_net.parameters(), lr=self.update_lr)

    def forward(self, x_spt, y_spt, x_qry, y_qry):
        task_num, setsz, _ = x_spt.size()
        querysz = x_qry.size(1)

        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i

        for i in range(task_num):

            # 1. run the i-th task and compute loss for k=0
            logits = self.outer_net(x_spt[i], com=None)
            loss = F.mse_loss(logits, y_spt[i])
            grad = torch.autograd.grad(loss, self.outer_net.parameters(), allow_unused=True, retain_graph=True)
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.outer_net.parameters())))

            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.outer_net(x_qry[i], com=None)
                loss_q = F.mse_loss(logits_q, y_qry[i])

                losses_q[0] += loss_q

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.outer_net(x_qry[i], fast_weights)
                loss_q = F.mse_loss(logits_q, y_qry[i])
                losses_q[1] += loss_q

            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = self.outer_net(x_spt[i], fast_weights)

                loss = F.mse_loss(logits, y_spt[i])
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                logits_q = self.outer_net(x_qry[i], fast_weights)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = F.mse_loss(logits_q, y_qry[i])
                losses_q[k + 1] += loss_q

        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num
        # optimize theta parameters
        self.outer_optim.zero_grad()
        loss_q.backward()
        # print('meta update')
        # for p in self.net.parameters()[:5]:
        # 	print(torch.norm(p).item())
        self.outer_optim.step()
        loss_np = [i.item() for i in losses_q]
        loss_avg = np.array(loss_np) / (querysz * task_num)
        return loss_avg

    def finetunning(self, x_spt, y_spt, x_qry, y_qry, param_spt, param_qry):
        """
        :param x_spt:   [task_num, setsz, 1]
        :param y_spt:   [task_num, setsz, 1]
        :param x_qry:   [task_num, querysz, 1]
        :param y_qry:   [task_num, querysz, 1]
        :param param_spt: [task_num, setsz, 2]
        :param param_qry: [task_num, querysz, 2]
        :return:
        """
        assert len(x_spt.shape) == 2

        querysz = x_qry.size(0)

        losses_q = [0 for _ in range(self.update_step_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.outer_net)

        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt)
        loss = F.mse_loss(logits, y_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, com=None)
            loss_q = F.mse_loss(logits_q, y_qry)

            losses_q[0] += loss_q

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, fast_weights)
            loss_q = F.mse_loss(logits_q, y_qry)
            losses_q[1] += loss_q

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(x_spt, fast_weights)

            loss = F.mse_loss(logits, y_spt)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            logits_q = net(x_qry, fast_weights)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = F.mse_loss(logits_q, y_qry)
            losses_q[k + 1] += loss_q

        del net

        loss_avg = np.array(losses_q) / querysz

        return loss_avg


if __name__ == '__main__':
    nshot = SinwaveNShot(2000, 20, 5, 5, 15, 'data')
    x_spt, y_spt, x_qry, y_qry, param_spt, param_qry = nshot.next('train')
    x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt), torch.from_numpy(y_spt), torch.from_numpy(
        x_qry), torch.from_numpy(y_qry)
    meta = Meta(hid_dim=64, meta_lr=1e-3, update_lr=2e-1)
    X = torch.randn(21, 1)
    Y = meta(x_spt, y_spt, x_qry, y_qry)
    print('loss is {}'.format(Y))
