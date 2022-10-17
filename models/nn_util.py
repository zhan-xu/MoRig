"""
This file belongs to the MultiBodySync code repository and is distributed for free.
Author: Jiahui Huang <huang-jh18@mails.tsinghua.edu.cn>
"""


import numpy as np
import torch
import torch.nn as nn


class GroupNorm(nn.Sequential):
    def __init__(self, in_size, num_groups, name=""):
        super(GroupNorm, self).__init__()
        self.add_module(name + "gn", nn.GroupNorm(num_groups, in_size))
        nn.init.constant_(self[0].weight, 1.0)
        nn.init.constant_(self[0].bias, 0.0)


class _BNBase(nn.Sequential):
    def __init__(self, in_size, batch_norm=None, name=""):
        super(_BNBase, self).__init__()
        self.add_module(name + "bn", batch_norm(in_size))
        nn.init.constant_(self[0].weight, 1.0)
        nn.init.constant_(self[0].bias, 0)


class BatchNorm1d(_BNBase):
    def __init__(self, in_size, name=""):
        super(BatchNorm1d, self).__init__(in_size, batch_norm=nn.BatchNorm1d, name=name)


class BatchNorm2d(_BNBase):
    def __init__(self, in_size, name=""):
        super(BatchNorm2d, self).__init__(in_size, batch_norm=nn.BatchNorm2d, name=name)


class _INBase(nn.Sequential):
    def __init__(self, in_size, instance_norm=None, name=""):
        super(_INBase, self).__init__()
        self.add_module(name + "in", instance_norm(in_size))


class InstanceNorm1d(_INBase):
    def __init__(self, in_size, name=""):
        super(InstanceNorm1d, self).__init__(in_size, instance_norm=nn.InstanceNorm1d, name=name)


class InstanceNorm2d(_INBase):
    def __init__(self, in_size, name=""):
        super(InstanceNorm2d, self).__init__(in_size, instance_norm=nn.InstanceNorm2d, name=name)


def get_norm_layer(layer_def, dimension, **kwargs):
    if layer_def is None:
        return nn.Identity()
    class_name = layer_def["class"]
    kwargs.update(layer_def)
    del kwargs["class"]
    return {
        "GroupNorm": GroupNorm,
        "BatchNorm": [BatchNorm1d, BatchNorm2d][dimension - 1],
        "InstanceNorm": [InstanceNorm1d, InstanceNorm2d][dimension - 1]
    }[class_name](**kwargs)


class _ConvBase(nn.Sequential):

    def __init__(self, in_size, out_size, kernel_size, stride, padding, dilation,
                 activation, bn, bn_dim, init, conv=None,
                 bias=True, preact=False, name=""):
        super(_ConvBase, self).__init__()

        bias = bias and (bn is None)
        conv_unit = conv(
            in_size,
            out_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias)
        init(conv_unit.weight)
        if bias:
            nn.init.constant_(conv_unit.bias, 0)

        if bn is not None:
            if not preact:
                bn_unit = get_norm_layer(bn, bn_dim, in_size=out_size)
            else:
                bn_unit = get_norm_layer(bn, bn_dim, in_size=in_size)

        if preact:
            if bn is not None:
                self.add_module(name + 'normlayer', bn_unit)

            if activation is not None:
                self.add_module(name + 'activation', activation)

        self.add_module(name + 'conv', conv_unit)

        if not preact:
            if bn is not None:
                self.add_module(name + 'normlayer', bn_unit)

            if activation is not None:
                self.add_module(name + 'activation', activation)


class Conv1d(_ConvBase):
    def __init__(self, in_size, out_size, kernel_size=1, stride=1, padding=0, dilation=1,
                 activation=nn.ReLU(inplace=True), bn=None, init=nn.init.kaiming_normal_,
                 bias=True, preact=False, name=""):
        super(Conv1d, self).__init__(
            in_size, out_size, kernel_size, stride, padding, dilation,
            activation, bn, 1,
            init, conv=nn.Conv1d,
            bias=bias, preact=preact, name=name)


class Conv2d(_ConvBase):
    def __init__(self, in_size, out_size, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1),
                 activation=nn.ReLU(inplace=True), bn=None, init=nn.init.kaiming_normal_,
                 bias=True, preact=False, name=""):
        super(Conv2d, self).__init__(
            in_size, out_size, kernel_size, stride, padding, dilation,
            activation, bn, 2,
            init, conv=nn.Conv2d,
            bias=bias, preact=preact, name=name)


class FC(nn.Sequential):

    def __init__(self,
                 in_size,
                 out_size,
                 activation=nn.ReLU(inplace=True),
                 bn=None,
                 init=None,
                 preact=False,
                 name=""):
        super(FC, self).__init__()

        fc = nn.Linear(in_size, out_size, bias=bn is None)
        if init is not None:
            init(fc.weight)
        if bn is None:
            nn.init.constant_(fc.bias, 0)

        if bn is not None:
            if not preact:
                bn_unit = get_norm_layer(bn, 1, in_size=out_size)
            else:
                bn_unit = get_norm_layer(bn, 1, in_size=in_size)

        if preact:
            if bn is not None:
                self.add_module(name + 'normlayer', bn_unit)

            if activation is not None:
                self.add_module(name + 'activation', activation)

        self.add_module(name + 'fc', fc)

        if not preact:
            if bn is not None:
                self.add_module(name + 'normlayer', bn_unit)

            if activation is not None:
                self.add_module(name + 'activation', activation)


class SharedMLP(nn.Sequential):

    def __init__(self, args,
                 bn=None, activation=nn.ReLU(inplace=True),
                 preact=False, first=False, name=""):
        super(SharedMLP, self).__init__()

        for i in range(len(args) - 1):
            self.add_module(
                name + 'layer{}'.format(i),
                Conv2d(
                    args[i],
                    args[i + 1],
                    bn=bn if (not first or not preact or (i != 0)) else None,
                    activation=activation if (not first or not preact or
                                              (i != 0)) else None,
                    preact=preact
                ))


def knead_leading_dims(n_dim: int, data: torch.Tensor):
    if data is None:
        return None
    data_dim = list(data.size())
    knead_size = np.prod(data_dim[:n_dim])
    new_size = tuple([knead_size, *data_dim[n_dim:]])
    return data.view(new_size)


def break_leading_dim(dim_size: list, data: torch.Tensor):
    if data is None:
        return None
    data_dim = list(data.size())
    new_size = tuple([*dim_size, *data_dim[1:]])
    return data.view(new_size)


class Seq(nn.Sequential):

    def __init__(self, input_channels):
        super(Seq, self).__init__()
        self.count = 0
        self.current_channels = input_channels

    def conv1d(self, out_size, kernel_size=1, stride=1, padding=0, dilation=1, activation=nn.ReLU(inplace=True), leaky=False,
               bn=None, init=nn.init.kaiming_normal_, bias=True, preact=False, name=""):
        if leaky:
            activation = nn.LeakyReLU(0.1, inplace=True)

        self.add_module(
            str(self.count),
            Conv1d(
                self.current_channels, out_size, kernel_size=kernel_size, stride=stride,
                padding=padding, dilation=dilation, activation=activation, bn=bn, init=init,
                bias=bias, preact=preact, name=name))
        self.count += 1
        self.current_channels = out_size

        return self

    def conv2d(self, out_size, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1),
               activation=nn.ReLU(inplace=True), bn=None, init=nn.init.kaiming_normal_, bias=True, preact=False, name=""):
        self.add_module(
            str(self.count),
            Conv2d(
                self.current_channels, out_size, kernel_size=kernel_size, stride=stride,
                padding=padding, dilation=dilation, activation=activation, bn=bn, init=init,
                bias=bias, preact=preact, name=name))
        self.count += 1
        self.current_channels = out_size

        return self

    def fc(self, out_size, activation=nn.ReLU(inplace=True), bn=None, init=None, preact=False, name=""):
        self.add_module(
            str(self.count),
            FC(self.current_channels,
               out_size, activation=activation, bn=bn, init=init, preact=preact, name=name))
        self.count += 1
        self.current_channels = out_size

        return self

    def dropout(self, p=0.5):
        self.add_module(str(self.count), nn.Dropout(p=p))
        self.count += 1
        return self

    def maxpool2d(self, kernel_size, stride=None, padding=0, dilation=1,
                  return_indices=False, ceil_mode=False):
        self.add_module(
            str(self.count),
            nn.MaxPool2d(
                kernel_size=kernel_size, stride=stride,
                padding=padding, dilation=dilation,
                return_indices=return_indices, ceil_mode=ceil_mode))
        self.count += 1

        return self