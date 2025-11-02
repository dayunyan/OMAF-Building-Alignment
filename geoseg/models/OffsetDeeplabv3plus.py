from typing import Dict

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

from tools.modules import init_weight
from geoseg.models.backbones.resnet import get_convnet
from geoseg.models.decoders.sep_aspp import SepASPP
from geoseg.models.convs import DepthwiseSeparableConv


class OffsetDeepLabV3Plus(nn.Module):
    """
    DeepLab v3+: Dilated ResNet with multi-grid + improved ASPP + decode
    Deeplabv3plus implememts
    This module has five components:

    self.backbone
    self.aspp
    self.projector: an 1x1 conv for lowlevel feature projection
    self.preclassifier: an 3x3 conv for feature mixing, before final classification
    self.classifier: last 1x1 conv for output classification results

    Args:
        backbone: Dict, configs for backbone
        decoder: Dict, configs for decoder

    NOTE: The bottleneck has only one 3x3 conv by default, some implements stack
        two 3x3 convs
    """

    def __init__(self, backbone: Dict, decoder: Dict) -> None:
        super(OffsetDeepLabV3Plus, self).__init__()

        self.align_corners = decoder["align_corners"]
        BN_op = getattr(nn, decoder["norm_layer"])
        channels = decoder["channels"]
        # self.backbone = modules.backbones.get_backbone(backbone)
        self.backbone = get_convnet(**backbone)

        # self.aspp = modules.decoders.get_decoder(decoder)
        decoder.pop("type")
        self.aspp = SepASPP(**decoder)
        self.projector = nn.Sequential(
            nn.Conv2d(
                decoder["lowlevel_in_channels"],
                decoder["lowlevel_channels"],
                kernel_size=1,
                bias=False,
            ),
            BN_op(decoder["lowlevel_channels"]),
            nn.ReLU(inplace=True),
        )
        self.pre_classifier = DepthwiseSeparableConv(
            decoder["norm_layer"],
            channels + decoder["lowlevel_channels"],
            channels,
            3,
            padding=1,
        )

        self.classifier = nn.Conv2d(channels, decoder["num_classes"], 1, 1)

        # 替换输入层为4通道
        first_conv = "conv1"
        old_conv = getattr(self.backbone.stem_layer, first_conv)
        new_conv = nn.Conv2d(
            in_channels=4,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        # 拷贝原有权重前三通道，第四通道初始化为均值
        with torch.no_grad():
            new_conv.weight[:, :3] = old_conv.weight
            new_conv.weight[:, 3] = old_conv.weight.mean(dim=1)
            if old_conv.bias is not None:
                new_conv.bias = old_conv.bias
        setattr(self.backbone.stem_layer, first_conv, new_conv)

        init_weight(self.projector)
        init_weight(self.pre_classifier)
        init_weight(self.classifier)

    def forward(self, x: Tensor) -> Tensor:
        size = (x.shape[2], x.shape[3])
        out = self.backbone(x)
        lowlevel_feature = self.projector(out[1])
        output = self.aspp(out[4])
        output = F.interpolate(
            output,
            size=lowlevel_feature.shape[-2:],
            mode="bilinear",
            align_corners=self.align_corners,
        )
        output = torch.cat([lowlevel_feature, output], dim=1)
        output = self.pre_classifier(output)
        out["embeddings"] = output
        output = self.classifier(output)
        out["pre_logits"] = output
        out["logits"] = F.interpolate(
            output, size=size, mode="bilinear", align_corners=self.align_corners
        )

        return out
