import torch
import os
import torch.nn as nn
from SinwaveNShot import SinwaveNShot
from meta import Meta
import numpy as np


def main():
    torch.manual_seed(121)
    torch.cuda.manual_seed_all(121)
    np.random.seed(121)

    nshot = SinwaveNShot(all_numbers_class=2000, batch_size=20,
                         n_way=5, k_shot=5, k_query=15, root='data')
    maml = Meta(hid_dim=64, meta_lr=1e-3, update_lr=0.004)

    for step in range(10000):
        x_spt, y_spt, x_qry, y_qry, param_spt, param_qry = nshot.next('train')
        x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt), torch.from_numpy(y_spt), torch.from_numpy(
            x_qry), torch.from_numpy(y_qry)

        loss = maml(x_spt, y_spt, x_qry, y_qry)
        if step % 20 == 0:
            print('step:', step, '\ttraining loss:', loss)

        if step % 500 == 0:
            loss = []
            for _ in range(1000//20):
                # test
                x_spt, y_spt, x_qry, y_qry, param_spt, param_qry = nshot.next('test')
                x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt), torch.from_numpy(y_spt), \
                                             torch.from_numpy(x_qry), torch.from_numpy(y_qry)

                # split to single task each time
                for x_spt_one, y_spt_one, x_qry_one, y_qry_one, param_spt_one, param_qry_onein in \
                        zip(x_spt, y_spt, x_qry, y_qry, param_spt, param_qry):
                    test_acc = maml.finetunning(x_spt_one, y_spt_one, x_qry_one, y_qry_one, param_spt_one, param_qry_onein)
                    loss.append(test_acc)

            # [b, update_step+1]
            loss = np.array(loss).mean(axis=0).astype(np.float16)
            print('Test loss:', loss)


if __name__ == '__main__':
    main()