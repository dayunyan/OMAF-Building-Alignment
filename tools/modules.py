import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F


def init_weight(module: nn.Module, a=0, mode="fan_in", nonlinearity="relu"):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
