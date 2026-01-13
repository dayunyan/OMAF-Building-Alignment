from typing import Dict

from . import base
from . import aspp
from . import sep_aspp
from . import segformer_head
from . import feedformer_head
from . import vm_head
from . import psp_head
from . import uper_head


# def get_decoder(config: Dict):
#    decoder_obj = MODULES_REG.DECODERS.get(config.type)
#    return decoder_obj(**config.settings)
