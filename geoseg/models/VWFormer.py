from typing import Dict

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

from geoseg.models.backbones.swin import get_swin
from geoseg.models.decoders.vm_head import VWHead


class VWformer(nn.Module):
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
        super(VWformer, self).__init__()

        self.align_corners = decoder["align_corners"]
        self.backbone = get_swin(**backbone)
        self.decoder = VWHead(**decoder)

    def forward(self, x: Tensor) -> Tensor:
        size = (x.shape[2], x.shape[3])
        output = self.backbone(x)
        output = self.decoder(output)

        out = {}
        out["logits"] = F.interpolate(
            output, size=size, mode="bilinear", align_corners=self.align_corners
        )
        return out
