import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Union, Tuple


class ConvModule(nn.Module):
    """
    组合模块：Conv2d + Norm + Activation + (Dropout)
    完全对齐 mmcv.cnn.ConvModule 接口，支持 SyncBN、自定义卷积/归一化/激活
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int], str] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: Optional[bool] = None,
        conv_cfg: Optional[Dict] = None,
        norm_cfg: Optional[Dict] = None,
        act_cfg: Optional[Dict] = dict(type="ReLU"),
        dropout_cfg: Optional[Dict] = None,
        padding_mode: str = "zeros",
        order: Tuple[str, ...] = ("conv", "norm", "act", "dropout"),
    ):
        super().__init__()
        self.order = order
        self.conv_cfg = conv_cfg or dict(type="Conv2d")
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.dropout_cfg = dropout_cfg

        # 1. 处理卷积层
        conv_type = self.conv_cfg.get("type", "Conv2d")
        assert conv_type == "Conv2d", f"仅支持 Conv2d，当前传入 {conv_type}"
        # 当使用归一化时，默认禁用卷积层bias
        if bias is None:
            bias = norm_cfg is None
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding if isinstance(padding, (int, tuple)) else 0,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

        # 2. 处理归一化层（支持 SyncBN/BN2d/LN 等）
        self.norm = None
        if norm_cfg is not None:
            norm_type = norm_cfg.get("type")
            norm_kwargs = {k: v for k, v in norm_cfg.items() if k != "type"}
            if norm_type in ["BN2d", "SyncBN"]:
                NormLayer = nn.BatchNorm2d if norm_type == "BN2d" else nn.SyncBatchNorm
                self.norm = NormLayer(out_channels, **norm_kwargs)
            elif norm_type == "LN":
                self.norm = nn.LayerNorm(out_channels, **norm_kwargs)
            else:
                raise ValueError(f"不支持的归一化类型：{norm_type}")

        # 3. 处理激活层
        self.act = None
        if act_cfg is not None and act_cfg.get("type") is not None:
            act_type = act_cfg.get("type")
            act_kwargs = {k: v for k, v in act_cfg.items() if k != "type"}
            if act_type == "ReLU":
                self.act = nn.ReLU(**act_kwargs)
            elif act_type == "LeakyReLU":
                self.act = nn.LeakyReLU(**act_kwargs)
            elif act_type == "GELU":
                self.act = nn.GELU(**act_kwargs)
            elif act_type == "SiLU":
                self.act = nn.SiLU(**act_kwargs)
            else:
                raise ValueError(f"不支持的激活类型：{act_type}")

        # 4. 处理 Dropout 层
        self.dropout = None
        if dropout_cfg is not None:
            dropout_type = dropout_cfg.get("type", "Dropout")
            dropout_kwargs = {k: v for k, v in dropout_cfg.items() if k != "type"}
            if dropout_type == "Dropout":
                self.dropout = nn.Dropout(**dropout_kwargs)
            elif dropout_type == "Dropout2d":
                self.dropout = nn.Dropout2d(**dropout_kwargs)
            else:
                raise ValueError(f"不支持的Dropout类型：{dropout_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer_name in self.order:
            if layer_name == "conv":
                x = self.conv(x)
            elif layer_name == "norm" and self.norm is not None:
                x = self.norm(x)
            elif layer_name == "act" and self.act is not None:
                x = self.act(x)
            elif layer_name == "dropout" and self.dropout is not None:
                x = self.dropout(x)
        return x
