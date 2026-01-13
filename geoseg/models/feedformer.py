from typing import Dict

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

from geoseg.models.backbones.mit import get_mit
from geoseg.models.decoders.feedformer_head import FeedFormerHead
from geoseg.models.convs import DepthwiseSeparableConv
from tools.modules import init_weight


class Feedformer(nn.Module):
    """
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
        super(Feedformer, self).__init__()

        self.align_corners = decoder["align_corners"]
        self.backbone = get_mit(**backbone)
        self.decoder = FeedFormerHead(**decoder)

    def forward(self, x: Tensor) -> Tensor:
        size = (x.shape[2], x.shape[3])
        output = self.backbone(x)
        output = self.decoder(output)

        out = {}
        out["logits"] = F.interpolate(
            output, size=size, mode="bilinear", align_corners=self.align_corners
        )
        return out
