from typing import Dict

from . import resnet
from . import mit
from . import swin
from . import vmamba

#def get_backbone(config: Dict):
#    backbone_obj = MODULES_REG.BACKBONES.get(config.type)
#    return backbone_obj(config.pretrain, **config.settings)
