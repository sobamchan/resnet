import numpy as np
import chainer
from sobamchan import sobamchan_chainer
import chainer.links as L
import chainer.functions as F
from chainer import Variable

class MLP(sobamchan_chainer.Model):

    def __init__(self):
        super(MLP, self).__init__(
            h1=L.Linear(None, 1000),
            h2=L.Linear(1000, 1280),
            h3=L.Linear(1280, 10),
            bn1=L.BatchNormalization(1000),
            bn2=L.BatchNormalization(1280),
            bn3=L.BatchNormalization(10),
        )

    def __call__(self, x, t, train=True):
        y = self.fwd(x, train)
        return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

    def fwd(self, x, train=True):
        x = F.relu(self.bn1(self.h1(x)))
        x = F.relu(self.bn2(self.h2(x)))
        x = self.bn3(self.h3(x))
        return x
