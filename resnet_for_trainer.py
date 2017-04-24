import numpy as np
import chainer
from sobamchan import sobamchan_chainer
import chainer.links as L
import chainer.functions as F
from chainer import Variable

class ResBlock(sobamchan_chainer.Model):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__(
            conv1=L.Convolution2D(in_channels, out_channels, ksize=(3,3), stride=1, pad=1),
            conv2=L.Convolution2D(out_channels, out_channels, ksize=(3,3), stride=1, pad=1),
            bn1=L.BatchNormalization(out_channels),
            bn2=L.BatchNormalization(out_channels),
        )

    def __call__(self, x):
        h = self.fwd(x)
        return h

    def fwd(self, x):
        h = F.relu(self.bn1((self.conv1(x))))
        h = F.relu(self.bn2((self.conv2(h))))
        _, x_channels, _, _ = x.shape
        h_batch_size, h_channels, h_h, h_w = h.shape
        if x_channels != h_channels:
            pad = chainer.Variable(np.zeros((h_batch_size, h_channels - x_channels, h_h, h_w)).astype(np.float32))
            if np.ndarray is not type(h.data):
                pad.to_gpu()
            return h + F.concat((x, pad))
        return h + x

class ResNet(sobamchan_chainer.Model):

    def __init__(self):
        super(ResNet, self).__init__()
        layer_i = 1
        n = 2
        modules = []
        input_channel = 3
        # 16 layer, 32 * 32 output map size
        for i in range(layer_i, layer_i+n*2+1):
            modules += [('resblock_{}'.format(layer_i), ResBlock(input_channel, 16, stride=1))]
            input_channel = 16
            layer_i += 1
        # 32 layer
        for i in range(layer_i, layer_i+n*2):
            if i+layer_i != layer_i+n*2:
                modules += [('resblock_{}'.format(layer_i), ResBlock(input_channel, 32, stride=1))]
            else:
                modules += [('resblock_{}'.format(layer_i), ResBlock(input_channel, 32, stride=2))]
            input_channel = 16
            layer_i += 1
        # 64 layer
        for i in range(layer_i, layer_i+n*2):
            if i+layer_i != layer_i+n*2:
                modules += [('resblock_{}'.format(layer_i), ResBlock(input_channel, 64, stride=1))]
            else:
                modules += [('resblock_{}'.format(layer_i), ResBlock(input_channel, 64, stride=2))]
            input_channel = 8
            layer_i += 1

        modules += [('fc', L.Linear(None, 10))]
        
        # register
        [ self.add_link(*link) for link in modules ]
        self.modules = modules
        self.layer_n = layer_i

    def __call__(self, x):
        y = self.fwd(x)
        return y

    def fwd(self, x):
        # convs and bns
        for i in range(1, self.layer_n-1):
            x = self['resblock_{}'.format(i)](x)
            if i == 1:
                x = F.max_pooling_2d(x, (2,2), stride=1)

        x = F.average_pooling_2d(x, (2,2), stride=1)
        # fc
        x = self['fc'](x)

        return x
