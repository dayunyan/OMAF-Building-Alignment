from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.teq_dataset import *
from geoseg.models.UNetFormerOutDict import UNetFormerOutDict
from tools.utils import Lookahead
from tools.utils import process_model_params

# training hparam
max_epoch = 40
ignore_index = 255
train_batch_size = 64
val_batch_size = 64
lr = 6e-4
weight_decay = 0.01
backbone_lr = 6e-5
backbone_weight_decay = 0.01
num_classes = len(CLASSES)
classes = CLASSES

weights_name = "unetformer-w-pred_offsets"
weights_path = "model_weights/offset/{}".format(weights_name)
test_weights_name = "unetformer-w-pred_offsets"
log_name = "offset/{}".format(weights_name)
visualize_name = "vis_logs/unetformer-w-pred_offsets/{}".format(weights_name)
monitor = "val_F1"
monitor_mode = "max"
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1
pretrained_ckpt_path = "model_weights/xbd/unetformer-r18-512-crop-ms-e105/unetformer-r18-512-crop-ms-e105.ckpt"  # "pretrained_weights/stseg_base.pth"  # the path for the pretrained model weight
gpus = [
    1,
]  # default or gpu ids:[0] or gpu nums: 2, more setting can refer to pytorch_lightning
resume_ckpt_path = None  # whether continue training with the checkpoint, default None
backbone_name = "swsl_resnet18"
backbone_pretrained_cfg_overlay = {
    "file": r"pretrained_weights/timm/resnet18.fb_swsl_ig1b_ft_in1k/pytorch_model.bin"
}

#  define the network
net = UNetFormerOutDict(
    backbone_name=backbone_name,
    num_classes=num_classes,
    pretrained_cfg_overlay=backbone_pretrained_cfg_overlay,
)

# define the loss
alpha = 0.1
loss = UnetFormerLoss(ignore_index=ignore_index)
use_aux_loss = True

# define the dataloader

train_dataset = TeqDataset(
    data_root="../data/segmentation/Turkey/Islahiye/pre/train",
    mode="train",
    mask_dir="pred_offsets",
    mosaic_ratio=0.25,
    transform=train_aug,
)

val_dataset = TeqDataset(
    data_root="../data/segmentation/Turkey/Islahiye/pre/test",
    mode="test",
    transform=val_aug,
)
test_dataset = TeqDataset(
    data_root="../data/segmentation/Turkey/Islahiye/pre/test",
    mode="test",
    transform=val_aug,
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=train_batch_size,
    num_workers=0,
    pin_memory=True,
    shuffle=True,
    drop_last=True,
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=val_batch_size,
    num_workers=0,
    shuffle=False,
    pin_memory=True,
    drop_last=False,
)

# define the optimizer
layerwise_params = {
    "backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)
}
net_params = process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=5, T_mult=2
)
