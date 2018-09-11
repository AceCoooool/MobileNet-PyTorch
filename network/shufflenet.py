# reference: nearly copy
# 1. https://github.com/kuangliu/pytorch-cifar/blob/master/models/shufflenet.py

import torch
from torch import nn
from torch.nn import functional as F


class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        n, c, h, w = x.size()
        assert c % self.groups == 0
        x = x.view(n, self.groups, c // self.groups, h, w).permute(0, 2, 1, 3, 4)
        x = x.contiguous().view(n, c, h, w)
        return x


class Bottleneck(nn.Module):
    def __init__(self, in_c, out_c, stride, groups):
        super(Bottleneck, self).__init__()
        self.stride = stride
        mid_c = out_c // 4
        g = 1 if in_c == 24 else groups
        self.gconv1 = nn.Sequential(nn.Conv2d(in_c, mid_c, 1, groups=g, bias=False),
                                    nn.BatchNorm2d(mid_c), nn.ReLU(inplace=True))
        self.shuffle1 = ShuffleBlock(g)
        self.dwconv = nn.Sequential(nn.Conv2d(mid_c, mid_c, 3, stride, 1, groups=mid_c, bias=False),
                                    nn.BatchNorm2d(mid_c), nn.ReLU(inplace=True))
        self.gconv2 = nn.Sequential(nn.Conv2d(mid_c, out_c, 1, groups=g, bias=False),
                                    nn.BatchNorm2d(out_c))

        if stride == 2:
            self.shortcut = nn.Sequential(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, x):
        out = self.gconv1(x)
        out = self.shuffle1(out)
        out = self.dwconv(out)
        out = self.gconv2(out)
        out = F.relu(torch.cat([out, self.shortcut(x)], 1)) if self.stride == 2 else F.relu(out + x)
        return out


class ShuffleNet(nn.Module):
    def __init__(self, cfg, classes=1000):
        super(ShuffleNet, self).__init__()
        out_planes = cfg['out_planes']
        num_blocks = cfg['num_blocks']
        groups = cfg['groups']
        self.in_planes, self.len = 24, len(out_planes)
        self.head = nn.Sequential(nn.Conv2d(3, 24, 3, 2, 1, bias=False), nn.BatchNorm2d(24),
                                  nn.ReLU(inplace=True), nn.MaxPool2d(3, 2, 1))
        self.layer0 = self._make_layer(out_planes[0], num_blocks[0], groups)
        self.layer1 = self._make_layer(out_planes[1], num_blocks[1], groups)
        self.layer2 = self._make_layer(out_planes[2], num_blocks[2], groups)
        self.linear = nn.Linear(out_planes[-1], classes)

    def _make_layer(self, out_planes, num_blocks, groups):
        layers = list()
        for i in range(num_blocks):
            stride = 2 if i == 0 else 1
            # due to concat operation
            cat_planes = self.in_planes if i == 0 else 0
            layers.append(Bottleneck(self.in_planes, out_planes - cat_planes, stride=stride, groups=groups))
            self.in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        n, c, h, w = x.size()
        out = self.head(x)
        out = self.layer0(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = F.avg_pool2d(out, h // 32).view(n, -1)
        out = self.linear(out)
        return out


def get_shufflenet(groups=2, classes=1000):
    cfg = [{'out_planes': [144, 288, 576], 'num_blocks': [4, 8, 4], 'groups': 1},
           {'out_planes': [200, 400, 800], 'num_blocks': [4, 8, 4], 'groups': 2},
           {'out_planes': [240, 480, 960], 'num_blocks': [4, 8, 4], 'groups': 3},
           {'out_planes': [272, 544, 1088], 'num_blocks': [4, 8, 4], 'groups': 4},
           {'out_planes': [384, 768, 1536], 'num_blocks': [4, 8, 4], 'groups': 8}]
    select = {1: 0, 2: 1, 3: 2, 4: 3, 8: 4}
    net = ShuffleNet(cfg[select[groups]], classes)
    return net


if __name__ == '__main__':
    net = get_shufflenet(2, 10)
    x = torch.randn(1, 3, 64, 64)
    print(net(x))
