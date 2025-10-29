import torch
from tools.cfg import py2cfg
from geoseg.models.UNetFormerOutDict import UNetFormerOutDict
from train_offset_v3 import Supervision_Train

# load config
cfg = py2cfg("config/teq/v5_offset.py")
# override net to avoid downloading pretrained weights during smoke test
cfg.net = UNetFormerOutDict(
    decode_channels=64,
    dropout=0.1,
    backbone_name=cfg.backbone_name,
    pretrained=False,
    window_size=8,
    num_classes=cfg.num_classes,
)
# ensure loss is present
# cfg.loss already set in config

# instantiate training module
model = Supervision_Train(cfg)
model.train()

# create synthetic single-sample batch
H = 256
W = 256
img = torch.randn(1, 3, H, W)
mask = torch.zeros(1, H, W, dtype=torch.long)
# draw a simple rectangle instance
mask[0, 32:96, 40:90] = 1
pred_mask = torch.zeros_like(mask)

batch = {
    "img": img,
    "gt_semantic_seg": mask,
    "pred_semantic_seg": pred_mask,
    "img_id": ["smoke_test"],
}

# run one training step
loss = model.training_step(batch, 0)
print("Smoke test loss output:", loss)
