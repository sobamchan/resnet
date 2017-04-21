import fire
import numpy as np
from tqdm import tqdm
from model import PlainCNN
import chainer
from chainer import computational_graph
from chainer import optimizers
import chainer.functions as F

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from sobamchan.sobamchan_iterator import Iterator
from sobamchan.sobamchan_log import Log


model = PlainCNN()

def train(model=model, gpu=None, epoch=10, batch_size=128):
    train, test = chainer.datasets.cifar.get_cifar10()
    train_x = np.array([x[0] for x in train])
    train_t = np.array([x[1] for x in train])
    test_x = np.array([x[0] for x in test])
    test_t = np.array([x[1] for x in test])
    train_n = len(train_x)
    test_n = len(test_x)
    train_x = train_x.reshape(train_n, 3, 32, 32)
    test_x = test_x.reshape(test_n, 3, 32, 32)

    train_x = np.subtract(train_x, np.mean(train_x, axis=0))
    test_x = np.subtract(test_x, np.mean(test_x, axis=0))

    optimizer = optimizers.SGD()
    if gpu:
        chainer.cuda.get_device(gpu).use()
        model.to_gpu()
        xp = chainer.cuda.cupy
    else:
        xp = np

    optimizer.setup(model)
    optimizer.lr = 0.1
    optimizer.add_hook(chainer.optimizer.WeightDecay(.05))
    optimizer.add_hook(chainer.optimizer.GradientClipping(.0005))

    train_log = Log()
    test_loss_log = Log()
    test_acc_log = Log()
    for i in tqdm(range(epoch)):
        order = np.random.permutation(train_n)
        train_iter_x = Iterator(train_x, batch_size, order=order)
        train_iter_t = Iterator(train_t, batch_size, order=order)
        sum_loss = 0
        if i % 10 == 0:
            optimizer.lr /= 2
        for x, t in tqdm(zip(train_iter_x, train_iter_t), total=train_n/batch_size):
            x = model.prepare_input(x, dtype=xp.float32, xp=xp)
            t = model.prepare_input(t, dtype=xp.int32, xp=xp)
            model.cleargrads()
            loss, _ = model(x, t)
            loss.backward()
            optimizer.update()
            loss.to_cpu()
            sum_loss += loss.data * len(x)
            del x
            del t
        if i == 0:
            with open('graph.dot', 'w') as o:
                g = computational_graph.build_computational_graph(
                    (loss, ), remove_split=True)
                o.write(g.dump())
            print('graph generated')

        del loss
        train_log.add(sum_loss/train_n)

        order = np.random.permutation(test_n)
        test_iter_x = Iterator(test_x, batch_size, order=order)
        test_iter_t = Iterator(test_t, batch_size, order=order)
        sum_loss = 0
        sum_acc = 0
        for x, t in tqdm(zip(test_iter_x, test_iter_t), total=test_n/batch_size):
            x = model.prepare_input(x, dtype=xp.float32, xp=xp)
            t = model.prepare_input(t, dtype=xp.int32, xp=xp)
            model.cleargrads()
            loss, acc = model(x, t, train=False)
            loss.to_cpu()
            acc.to_cpu()
            sum_loss += loss.data * len(x)
            sum_acc = acc.data * len(x)
        test_loss_log.add(sum_loss/test_n)
        test_acc_log.add(sum_acc/test_n)

    train_log.save('train.log')
    test_loss_log.save('test_loss.log')
    test_acc_log.save('test_acc.log')

fire.Fire()
