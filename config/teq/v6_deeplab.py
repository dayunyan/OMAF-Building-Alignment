from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.losses.useful_loss import UnetFormerLoss
from geoseg.datasets.teq_dataset import *
from geoseg.models.Deeplabv3plus import DeepLabV3Plus
from geoseg.models.discriminator import get_fc_discriminator
from tools.utils import Lookahead
from tools.utils import process_model_params
from tools.alignment import InstanceWiseAlignmentOptimizer

# training hparam
max_epoch = 50
ignore_index = len(CLASSES)
train_batch_size = 64
val_batch_size = 16
lr = 6e-4
weight_decay = 0.01
backbone_lr = 1e-5
backbone_weight_decay = 0.001
num_classes = len(CLASSES)
classes = CLASSES
warmup_epoch = 10

weights_name = "offset-v6-deeplab-w-align-mask"
weights_path = "model_weights/offset/{}".format(weights_name)
test_weights_name = "last-v1"  # "offset-v5-pretrain-xbd-RSB-predict-object-v1"
log_name = "offset-v6-w-align-mask/{}".format(weights_name)
visualize_name = "vis_logs/offset-v6-w-align-mask/{}".format(weights_name)
monitor = "val_F1"
monitor_mode = "max"
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1
pretrained_ckpt_path = None  # "model_weights/xbd/unetformer-r18-512-crop-ms-e105/unetformer-r18-512-crop-ms-e105.ckpt"  # "pretrained_weights/stseg_base.pth"  # the path for the pretrained model weight
gpus = [
    1
]  # default or gpu ids:[0] or gpu nums: 2, more setting can refer to pytorch_lightning
resume_ckpt_path = None  # whether continue training with the checkpoint, default None
backbone_name = "swsl_resnet18"
backbone_pretrained_cfg_overlay = {
    "file": r"pretrained_weights/timm/resnet18.fb_swsl_ig1b_ft_in1k/pytorch_model.bin"
}

# define network: use UNetFormerOutDict and UnetFormerLoss
net = DeepLabV3Plus(
    backbone={
        "pretrain": "pretrained_weights/resnetv1d101_mmcv.pth",
        "variety": "resnet-D",
        "depth": 101,
        "out_indices": [1, 4],
        "output_stride": 16,
        "contract_dilation": False,
        "multi_grid": True,
        "norm_layer": "SyncBatchNorm",
    },
    decoder={
        "type": "SepASPP",
        "in_channels": 2048,
        "channels": 256,
        "lowlevel_in_channels": 256,
        "lowlevel_channels": 48,
        "atrous_rates": [6, 12, 18],
        "dropout_ratio": 0.1,
        "num_classes": 2,
        "norm_layer": "SyncBatchNorm",
        "align_corners": False,
    },
)

# define the loss
loss = ConfidenceWeightedCrossEntropyLoss(distance_threshold=5)

# define the dataloader

train_dataset = TeqDataset(
    data_root="../data/segmentation/Turkey/Islahiye/pre/train",
    mode="train",
    mosaic_ratio=0.25,
    transform=train_aug,
    align_dir="align",
    align_suffix=".pt",
)

val_dataset = TeqDataset(
    data_root="../data/segmentation/Turkey/Islahiye/pre/test",
    mode="test",
    transform=val_aug,
    test_gt_dir="gt",
    align_dir="align",
    align_suffix=".pt",
)
test_dataset = TeqDataset(
    data_root="../data/segmentation/Turkey/Islahiye/pre/test",
    mode="test",
    transform=val_aug,
    test_gt_dir="gt",
    align_dir="align",
    align_suffix=".pt",
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
    optimizer, T_0=2, T_mult=2
)
