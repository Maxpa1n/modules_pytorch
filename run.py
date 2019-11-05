import torch
import torch.nn as nn
import argparse
from torch.optim import SGD
from torch.utils.data import DataLoader
import time
from torch.utils.data.dataset import random_split
from torchtext.datasets import text_classification
import torch.utils.data.distributed
import os

from tqdm import tqdm


def metric_average(args, val, name):
    tensor = torch.tensor(val)
    avg_tensor = args.hvd.allreduce(tensor, name=name)
    return avg_tensor.item()


class TextSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)


def generate_batch(batch):
    label = torch.tensor([entry[0] for entry in batch])
    text = [entry[1] for entry in batch]
    offsets = [0] + [len(entry) for entry in text]
    # torch.Tensor.cumsum returns the cumulative sum
    # of elements in the dimension dim.
    # torch.Tensor([1.0, 2.0, 3.0]).cumsum(dim=0)

    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
    return text, offsets, label


def train_func(args, sub_train_, model, optimizer, criterion, scheduler):
    # Train the model
    train_loss = 0
    train_acc = 0
    if args.horovod:
        args.hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        args.hvd.broadcast_optimizer_state(optimizer, root_rank=0)
        compression = args.hvd.Compression.fp16 if args.fp16_allreduce else args.hvd.Compression.none
        optimizer = args.hvd.DistributedOptimizer(optimizer,
                                                  named_parameters=model.named_parameters(),
                                                  compression=compression)

    if args.horovod:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            sub_train_, num_replicas=args.hvd_size, rank=args.rank)
        data = torch.utils.data.DataLoader(
            sub_train_, args.batch_size, sampler=train_sampler, collate_fn=generate_batch, **args.kwargs)
    else:
        data = DataLoader(sub_train_, batch_size=args.batch_size, shuffle=True,
                          collate_fn=generate_batch)

    for i, (text, offsets, cls) in enumerate(tqdm(data, desc='data train')):
        optimizer.zero_grad()
        if args.cuda:
            text, offsets, cls = text.cuda(), offsets.cuda(), cls.cuda()
        output = model(text, offsets)
        loss = criterion(output, cls)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        train_acc += (output.argmax(1) == cls).sum().item()

    # Adjust the learning rate
    scheduler.step()

    return train_loss / len(sub_train_), train_acc / len(sub_train_)


def test(args, data_, model, criterion):
    loss = 0
    acc = 0
    if args.horovod:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            data_, num_replicas=args.hvd_size, rank=args.rank)
        data = torch.utils.data.DataLoader(
            data_, args.batch_size, sampler=train_sampler, collate_fn=generate_batch, **args.kwargs)
    else:
        data = DataLoader(data_, batch_size=args.batch_size, shuffle=True,
                          collate_fn=generate_batch)
    for text, offsets, cls in data:
        if args.cuda:
            text, offsets, cls = text.cuda(), offsets.cuda(), cls.cuda()
        with torch.no_grad():
            output = model(text, offsets)
            loss = criterion(output, cls)
            loss += loss.item()
            acc += (output.argmax(1) == cls).sum().item()

    return loss / len(data_), acc / len(data_)


def main():
    parser = argparse.ArgumentParser(description='PyTorch NMIST example')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='batch size (default: 32)')
    parser.add_argument('--num_epochs', type=int, default=5, metavar='N',
                        help='number of epochs (default: 5)')
    parser.add_argument('--embedding_dim', type=int, default=128, metavar='N',
                        help='dimension of embedding (default: 128)')
    parser.add_argument('--ngrams', type=int, default=2, metavar='N',
                        help='size of ngrams (default: 2)')
    parser.add_argument('--horovod', action='store_true', default=False,
                        help='use horovod')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                        help='use fp16 compression during allreduce')
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if not os.path.isdir('data'):
        os.mkdir('data')
    train_dataset, test_dataset = text_classification.DATASETS['AG_NEWS'](
        root='data', ngrams=args.ngrams, vocab=None)

    args.vocab_size = len(train_dataset.get_vocab())
    args.num_class = len(train_dataset.get_labels())
    model = TextSentiment(args.vocab_size, args.embedding_dim, args.num_class)

    # horovod training
    if args.horovod:
        try:
            import horovod.torch as hvd
        except ImportError:
            raise ImportError("Please install horovod from https://github.com/horovod to use horovod training.")
        hvd.init()
        args.kwargs = {'num_workers': 1, 'pin_memory': True}
        args.hvd_size = hvd.size()
        args.rank = hvd.rank()
        args.hvd = hvd

    if args.cuda:
        # Horovod: pin GPU to local rank.
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(args.seed)

    torch.set_num_threads(1)
    args.kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    min_valid_loss = float('inf')

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
    if args.cuda:
        criterion.cuda()
        model.cuda()

    train_len = int(len(train_dataset) * 0.95)
    sub_train_, sub_valid_ = \
        random_split(train_dataset, [train_len, len(train_dataset) - train_len])

    for epoch in range(args.num_epochs):
        start_time = time.time()
        train_loss, train_acc = train_func(args, sub_train_, model, optimizer, criterion, scheduler)
        valid_loss, valid_acc = test(args, sub_valid_, model, criterion)

        secs = int(time.time() - start_time)
        mins = secs / 60
        secs = secs % 60

        if args.horovod:
            valid_loss = metric_average(valid_loss, 'avg_loss')
            valid_acc = metric_average(valid_acc, 'avg_accuracy')
            if hvd.rank() == 0:
                print('Epoch: %d' % (epoch + 1), " | time in %d minutes, %d seconds" % (mins, secs))
                print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
                print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')
        else:
            print('Epoch: %d' % (epoch + 1), " | time in %d minutes, %d seconds" % (mins, secs))
            print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
            print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')


if __name__ == '__main__':
    main()