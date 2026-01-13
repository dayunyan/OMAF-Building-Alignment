from typing import Dict, Type, Optional, Union

from .register import LayerRegistry
from .sep_conv import *
from .conv_module import *
from .nonlocal2d import *

# 1. 卷积层注册器（支持普通Conv2d和深度可分离卷积）
conv_registry = LayerRegistry()
# 注册默认卷积模块
conv_registry.register("Conv2d", nn.Conv2d)
conv_registry.register("DepthwiseSeparableConv", DepthwiseSeparableConv)

# 2. 归一化层注册器（支持LN、BN、SyncBN）
norm_registry = LayerRegistry()
# 注册默认归一化模块
norm_registry.register("LN", nn.LayerNorm)
norm_registry.register("BN1d", nn.BatchNorm1d)
norm_registry.register("BN2d", nn.BatchNorm2d)
norm_registry.register("SyncBatchNorm", nn.SyncBatchNorm)


def build_conv_layer(
    conv_cfg: Union[str, Dict],
    in_channels: int,
    out_channels: int,
    kernel_size: Union[int, tuple],
    stride: Union[int, tuple] = 1,
    padding: Union[int, tuple] = 0,
    dilation: Union[int, tuple] = 1,
    bias: bool = True,
    **kwargs
) -> nn.Module:
    """
    构建卷积层（支持普通Conv2d和深度可分离卷积）
    Args:
        conv_cfg: 卷积配置，支持字符串（如"Conv2d"）或字典（如{"type": "DepthwiseSeparableConv", "BN_op": "SyncBatchNorm"}）
        其余参数：与nn.Conv2d一致
    """
    # 合并通用参数和额外参数
    common_params = {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "bias": bias,
        **kwargs,
    }
    return conv_registry.build(conv_cfg, **common_params)


def build_norm_layer(
    norm_cfg: Union[str, Dict], num_features: int, **kwargs
) -> nn.Module:
    """
    构建归一化层（支持LN、BN2d、BN1d、SyncBN2d、SyncBN1d）
    Args:
        norm_cfg: 归一化配置，支持字符串（如"BN2d"）或字典（如{"type": "SyncBN2d", "eps": 1e-5}）
        num_features: 特征维度（必填）
        其余参数：归一化层额外参数（如eps、momentum等）
    """
    common_params = {"normalized_shape": num_features, **kwargs}
    return norm_registry.build(norm_cfg, **common_params)
