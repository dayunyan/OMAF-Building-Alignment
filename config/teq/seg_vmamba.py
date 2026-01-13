from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.losses.useful_loss import Deeplabv3PlusLoss
from geoseg.datasets.teq_dataset import *
from geoseg.models.VMamba import VMamba
from tools.utils import Lookahead
from tools.utils import process_model_params

# training hparam
max_epoch = 40
ignore_index = len(CLASSES)
train_batch_size = 16
val_batch_size = 16
lr = 1e-3
weight_decay = 0.01
backbone_lr = 1e-4
backbone_weight_decay = 0.001
num_classes = len(CLASSES)
classes = CLASSES
warmup_epoch = 10

weights_name = "vmamba-base-w-pred_offsets"  # deeplab-w-pred_offsets
weights_path = "model_weights/offset/{}".format(weights_name)
test_weights_name = "vmamba-base-w-pred_offsets"
log_name = "vmamba-base-w-pred_offsets/{}".format(weights_name)
visualize_name = "vis_logs/vmamba-base-w-pred_offsets/{}".format(weights_name)
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

# define network: use UNetFormerOutDict and UnetFormerLoss
net = VMamba(
    backbone=dict(
        depths=[
            2,
            2,
            15,
            2,
        ],
        dims=128,
        drop_path_rate=0.6,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        ssm_d_state=1,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",
        ssm_act_layer="silu",
        ssm_conv=3,
        ssm_conv_bias=False,
        ssm_drop_rate=0.0,
        ssm_init="v0",
        forward_type="v05_noz",
        mlp_ratio=4.0,
        mlp_act_layer="gelu",
        mlp_drop_rate=0.0,
        gmlp=False,
        patch_norm=True,
        norm_layer="ln2d",
        downsample_version="v3",
        patchembed_version="v2",
        use_checkpoint=False,
        posembed=False,
        imgsize=512,
    ),
    decoder=dict(
        in_channels=[128, 256, 512, 1024],
        in_index=[0, 1, 2, 3],
        pool_scales=(
            1,
            2,
            3,
            6,
        ),
        channels=512,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type="SyncBN", requires_grad=True),
        align_corners=False,
    ),
    pretrained="pretrained_weights/upernet_vssm_4xb4-160k_ade20k-512x512_base_iter_160000.pth",
)
# define the loss
loss = Deeplabv3PlusLoss(ignore_index=255)

# define the dataloader

train_dataset = TeqDataset(
    data_root="../data/segmentation/Turkey/Islahiye/pre/train",
    mode="train",
    mask_dir="pred_offsets",  # pred_offsets
    mosaic_ratio=0.25,
    transform=train_aug,
)

val_dataset = TeqDataset(
    data_root="../data/segmentation/Turkey/Islahiye/pre/test",
    mode="test",
    transform=val_aug,
    test_gt_dir="gt",
)
test_dataset = TeqDataset(
    data_root="../data/segmentation/Turkey/Islahiye/pre/test",
    mode="test",
    transform=val_aug,
    test_gt_dir="gt",
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=train_batch_size,
    num_workers=0,
    pin_memory=True,
    shuffle=True,
    drop_last=False,
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
