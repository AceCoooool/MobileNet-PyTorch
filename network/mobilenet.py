# reference:
# 1. https://github.com/apache/incubator-mxnet/blob/master/python/mxnet/gluon/model_zoo/vision/mobilenet.py

from torch import nn
from torch.nn import functional as F


def _add_conv(out, in_c=1, out_c=1, kernel=1, stride=1, pad=0,
              num_group=1, active=True, relu6=False):
    out.append(nn.Conv2d(in_c, out_c, kernel, stride, pad, groups=num_group, bias=False))
    out.append(nn.BatchNorm2d(out_c))
    if active:
        out.append(nn.ReLU6(inplace=True) if relu6 else nn.ReLU(inplace=True))


def _add_conv_dw(out, dw_channels, channels, stride, relu6=False):
    _add_conv(out, dw_channels, dw_channels, kernel=3, stride=stride,
              pad=1, num_group=dw_channels, relu6=relu6)
    _add_conv(out, dw_channels, channels, relu6=relu6)


class LinearBottleneck(nn.Module):
    """LinearBottleneck used in MobileNetV2 model from the
    `"Inverted Residuals and Linear Bottlenecks:
      Mobile Networks for Classification, Detection and Segmentation"
    <https://arxiv.org/abs/1801.04381>`_ paper.
    Parameters
    ----------
    in_channels : int
        Number of input channels.
    channels : int
        Number of output channels.
    t : int
        Layer expansion ratio.
    stride : int
        stride
    """

    def __init__(self, in_c, c, t, stride):
        super(LinearBottleneck, self).__init__()
        layers = list()
        self.use_shortcut = stride == 1 and in_c == c
        _add_conv(layers, in_c, in_c * t, relu6=True)
        _add_conv(layers, in_c * t, in_c * t, 3, stride, 1, in_c * t, relu6=True)
        _add_conv(layers, in_c * t, c, active=False, relu6=True)
        self.out = nn.Sequential(*layers)

    def forward(self, x):
        out = self.out(x)
        if self.use_shortcut:
            out = out + x
        return out


class MobileNet(nn.Module):
    """MobileNet model from the
    `"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
    <https://arxiv.org/abs/1704.04861>`_ paper.
    Parameters
    ----------
    multiplier : float, default 1.0
        The width multiplier for controling the model size. Only multipliers that are no
        less than 0.25 are supported. The actual number of channels is equal to the original
        channel size multiplied by this multiplier.
    classes : int, default 1000
        Number of classes for the output layer.
    """

    def __init__(self, multiplier=1.0, classes=1000):
        super(MobileNet, self).__init__()
        layers = list()
        dw_channels = [int(x * multiplier) for x in [32, 64] + [128] * 2
                       + [256] * 2 + [512] * 6 + [1024]]
        channels = [int(x * multiplier) for x in [64] + [128] * 2 + [256] * 2
                    + [512] * 6 + [1024] * 2]
        strides = [1, 2] * 3 + [1] * 5 + [2, 1]
        _add_conv(layers, 3, int(32 * multiplier), 3, 2, 1)
        for dwc, c, s in zip(dw_channels, channels, strides):
            _add_conv_dw(layers, dwc, c, s)
        self.feature = nn.Sequential(*layers)
        self.linear = nn.Linear(1024, classes)

    def forward(self, x):
        n, _, h, _ = x.size()
        x = self.feature(x)
        x = F.avg_pool2d(x, h // 32).view(n, -1)
        x = self.linear(x)
        return x


class MobileNetV2(nn.Module):
    """MobileNetV2 model from the
    `"Inverted Residuals and Linear Bottlenecks:
      Mobile Networks for Classification, Detection and Segmentation"
    <https://arxiv.org/abs/1801.04381>`_ paper.
    Parameters
    ----------
    multiplier : float, default 1.0
        The width multiplier for controling the model size. The actual number of channels
        is equal to the original channel size multiplied by this multiplier.
    classes : int, default 1000
        Number of classes for the output layer.
    """

    def __init__(self, multiplier=1.0, classes=1000):
        super(MobileNetV2, self).__init__()
        layers = list()
        _add_conv(layers, 3, int(32 * multiplier), 3, 2, 1, relu6=True)
        in_channels_group = [int(x * multiplier) for x in [32] + [16] + [24] * 2
                             + [32] * 3 + [64] * 4 + [96] * 3 + [160] * 3]
        channels_group = [int(x * multiplier) for x in [16] + [24] * 2 + [32] * 3
                          + [64] * 4 + [96] * 3 + [160] * 3 + [320]]
        ts = [1] + [6] * 16
        strides = [1, 2] * 2 + [1, 1, 2] + [1] * 6 + [2] + [1] * 3
        for in_c, c, t, s in zip(in_channels_group, channels_group, ts, strides):
            layers.append(LinearBottleneck(in_c, c, t, s))
        last_c = int(1280 * multiplier) if multiplier > 1.0 else 1280
        _add_conv(layers, int(320 * multiplier), last_c, relu6=True)
        self.feat = nn.Sequential(*layers)
        self.output = nn.Sequential(nn.Conv2d(last_c, classes, 1, bias=False))

    def forward(self, x):
        n, c, h, w = x.size()
        x = self.feat(x)
        x = F.avg_pool2d(x, h // 32)
        x = self.output(x)
        return x.view(n, -1)


def get_mobilenet(multiplier, classes=1000):
    return MobileNet(multiplier, classes)


def get_mobilenet_v2(multiplier, classes=1000):
    return MobileNetV2(multiplier, classes)


if __name__ == '__main__':
    import torch

    net = MobileNetV2(1.0, 10)
    x = torch.randn(2, 3, 224, 224)
    print(net(x).size())
