import numpy as np
import chainer
from sobamchan import sobamchan_chainer
import chainer.links as L
import chainer.functions as F
from chainer import Variable

class PlainCNN(sobamchan_chainer.Model):

    def __init__(self):
        super(PlainCNN, self).__init__()
        layer_i = 1
        n = 2
        modules = []
        input_channel = 3
        # 32 layer
        for i in range(layer_i, layer_i+n*2+1):
            modules += [('conv_{}'.format(layer_i), L.Convolution2D(input_channel, 32, ksize=(3,3), stride=2, pad=4))]
            modules += [('bnorm_{}'.format(layer_i), L.BatchNormalization(32))]
            input_channel = 32
            layer_i += 1
        # 16 layer
        for i in range(layer_i, layer_i+n*2):
            modules += [('conv_{}'.format(layer_i), L.Convolution2D(input_channel, 16, ksize=(3,3), stride=2, pad=4))]
            modules += [('bnorm_{}'.format(layer_i), L.BatchNormalization(16))]
            input_channel = 16
            layer_i += 1
        # 8 layer
        for i in range(layer_i, layer_i+n*2):
            modules += [('conv_{}'.format(layer_i), L.Convolution2D(input_channel, 8, ksize=(3,3), stride=2, pad=4))]
            modules += [('bnorm_{}'.format(layer_i), L.BatchNormalization(8))]
            input_channel = 8
            layer_i += 1

        modules += [('fc', L.Linear(None, 10))]
        
        # register
        [ self.add_link(*link) for link in modules ]
        self.modules = modules
        self.layer_n = layer_i

    def __call__(self, x, t, train=True):
        y = self.fwd(x, train)
        return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

    def fwd(self, x, train=True):
        # convs and bns
        for i in range(1, self.layer_n-1):
            x = self['conv_{}'.format(i)](x)
            x = F.relu(x)
            x = self['bnorm_{}'.format(i)](x)
            if i == 1:
                x = F.average_pooling_2d(x, (2,2))
        # fc
        x = self['fc'](x)

        return x
