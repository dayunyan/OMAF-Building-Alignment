import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Union, Tuple

from geoseg.models.convs import ConvModule


class NonLocal2d(nn.Module):
    """
    非局部模块，完全对齐 mmcv.cnn.NonLocal2d 接口
    支持模式：gaussian/embedded_gaussian/dot_product/concatenation
    支持下采样、尺度缩放、自定义卷积/归一化
    """

    def __init__(
        self,
        in_channels: int,
        reduction: int = 2,
        use_scale: bool = True,
        conv_cfg: Optional[Dict] = None,
        norm_cfg: Optional[Dict] = None,
        mode: str = "embedded_gaussian",
        sub_sample: bool = True,
        dropout_cfg: Optional[Dict] = None,
    ):
        super().__init__()
        assert mode in [
            "gaussian",
            "embedded_gaussian",
            "dot_product",
            "concatenation",
        ], f"不支持的模式：{mode}，可选模式：gaussian/embedded_gaussian/dot_product/concatenation"

        self.in_channels = in_channels
        self.inter_channels = in_channels // reduction  # 中间通道数
        self.use_scale = use_scale
        self.mode = mode
        self.sub_sample = sub_sample

        # 构建核心卷积层（theta/phi/g/conv_out）
        conv_cfg = conv_cfg or dict(type="Conv2d")
        norm_cfg = norm_cfg or None
        dropout_cfg = dropout_cfg or None

        # theta：查询投影（query）
        self.theta = ConvModule(
            in_channels=in_channels,
            out_channels=self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=None,  # theta 不使用归一化（遵循mmcv默认）
            act_cfg=None,  # theta 不使用激活（遵循mmcv默认）
        )

        # phi：键投影（key）
        self.phi = ConvModule(
            in_channels=in_channels,
            out_channels=self.inter_channels,
            kernel_size=1,
            stride=1 if not sub_sample else 2,  # 下采样时stride=2
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=None,
            act_cfg=None,
        )

        # g：值投影（value）
        self.g = ConvModule(
            in_channels=in_channels,
            out_channels=self.inter_channels,
            kernel_size=1,
            stride=1 if not sub_sample else 2,  # 下采样时stride=2
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=None,
            act_cfg=None,
        )

        # 下采样层（如果需要）
        self.sub_sample_layer = (
            nn.MaxPool2d(kernel_size=2, stride=2) if sub_sample else None
        )

        # 输出投影
        self.conv_out = ConvModule(
            in_channels=self.inter_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None,  # 输出层不使用激活（遵循mmcv默认）
            dropout_cfg=dropout_cfg,
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化卷积层权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm, nn.LayerNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def gaussian(self, theta_x: torch.Tensor, phi_x: torch.Tensor) -> torch.Tensor:
        """gaussian模式：theta_x * phi_x^T"""
        # theta_x: (N, H1W1, C)，phi_x: (N, C, H2W2)
        pairwise_weight = torch.matmul(theta_x, phi_x)
        if self.use_scale:
            pairwise_weight /= theta_x.shape[-1] ** 0.5
        pairwise_weight = F.softmax(pairwise_weight, dim=-1)
        return pairwise_weight

    def embedded_gaussian(
        self, theta_x: torch.Tensor, phi_x: torch.Tensor
    ) -> torch.Tensor:
        """embedded_gaussian模式（默认）：theta_x * phi_x^T，带softmax"""
        return self.gaussian(theta_x, phi_x)

    def dot_product(self, theta_x: torch.Tensor, phi_x: torch.Tensor) -> torch.Tensor:
        """dot_product模式：theta_x * phi_x^T，不带softmax"""
        pairwise_weight = torch.matmul(theta_x, phi_x)
        if self.use_scale:
            pairwise_weight /= theta_x.shape[-1] ** 0.5
        return pairwise_weight

    def concatenation(self, theta_x: torch.Tensor, phi_x: torch.Tensor) -> torch.Tensor:
        """concatenation模式：拼接后通过MLP计算相似度"""
        # theta_x: (N, C, H1W1, 1)，phi_x: (N, C, 1, H2W2)
        N, C, H1W1, _ = theta_x.shape
        _, _, _, H2W2 = phi_x.shape

        # 拼接：(N, 2C, H1W1, H2W2)
        concat = torch.cat(
            [theta_x.expand(-1, -1, -1, H2W2), phi_x.expand(-1, -1, H1W1, -1)], dim=1
        )
        # 投影到1维：(N, 1, H1W1, H2W2)
        pairwise_weight = self.conv_concat(concat)
        pairwise_weight = pairwise_weight.squeeze(1)  # (N, H1W1, H2W2)

        if self.use_scale:
            pairwise_weight /= concat.shape[1] ** 0.5
        pairwise_weight = F.softmax(pairwise_weight, dim=-1)
        return pairwise_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        原始forward逻辑（mmcv默认），VWA类重写了该方法，此处保持兼容
        Args:
            x: (N, C, H, W)
        Returns:
            out: (N, C, H, W)
        """
        N, C, H, W = x.shape

        # 1. 投影变换
        theta_x = (
            self.theta(x).view(N, self.inter_channels, -1).permute(0, 2, 1)
        )  # (N, HW, C)
        phi_x = self.phi(x).view(N, self.inter_channels, -1)  # (N, C, HW)
        g_x = self.g(x).view(N, self.inter_channels, -1).permute(0, 2, 1)  # (N, HW, C)

        # 2. 下采样（如果需要）
        if self.sub_sample_layer is not None:
            phi_x = self.sub_sample_layer(phi_x)
            g_x = self.sub_sample_layer(g_x)

        # 3. 计算相似度权重
        pairwise_func = getattr(self, self.mode)
        pairwise_weight = pairwise_func(theta_x, phi_x)  # (N, H1W1, H2W2)

        # 4. 加权求和
        out = torch.matmul(pairwise_weight, g_x)  # (N, H1W1, C)
        out = (
            out.permute(0, 2, 1).contiguous().view(N, self.inter_channels, H, W)
        )  # (N, C, H, W)

        # 5. 输出投影
        out = self.conv_out(out)
        # 残差连接
        out = out + x
        return out
