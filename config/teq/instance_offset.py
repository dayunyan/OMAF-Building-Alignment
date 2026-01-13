from torch.utils.data import DataLoader
from geoseg.losses.InstanceLoss import ConfidenceWeightedL1Loss
from geoseg.models.InstanceOffsetNet import InstanceOffsetNet
from geoseg.datasets.teq_instance_dataset import *
from tools.utils import Lookahead
from tools.utils import process_model_params

# training hparam
max_epoch = 20
train_batch_size = 64
val_batch_size = 16
lr = 1e-4
weight_decay = 0.01
backbone_lr = 1e-5
backbone_weight_decay = 0.001
classes = CLASSES
warmup_epoch = 10

weights_name = "offset-instance-bbox-w-align-conf"
weights_path = "model_weights/Islahiye/offset-instance/{}".format(weights_name)
test_weights_name = "offset-instance-bbox-w-align-conf"  # "offset-v5-pretrain-xbd-RSB-predict-object-v1"
log_name = "offset-instance-bbox-w-align-conf/{}".format(weights_name)
visualize_name = "vis_logs/Islahiye/offset-instance-bbox-w-align-conf/{}".format(
    weights_name
)
monitor = "val_MAE"  # ！！！监控新指标
monitor_mode = "min"  # ！！！我们希望 MAE 最小化
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1
pretrained_ckpt_path = None  # "model_weights/xbd/unetformer-r18-512-crop-ms-e105/unetformer-r18-512-crop-ms-e105.ckpt"  # "pretrained_weights/stseg_base.pth"  # the path for the pretrained model weight
gpus = [
    1
]  # default or gpu ids:[0] or gpu nums: 2, more setting can refer to pytorch_lightning
resume_ckpt_path = None  # whether continue training with the checkpoint, default None
backbone_cfg = {
    "pretrain": "pretrained_weights/resnetv1d101_mmcv.pth",
    "variety": "resnet-D",
    "depth": 101,
    "out_indices": [1, 4],
    "output_stride": 16,
    "contract_dilation": False,
    "multi_grid": True,
    "norm_layer": "SyncBatchNorm",
}
decoder_cfg = {
    "type": "SepASPP",
    "in_channels": 2048,
    "channels": 256,
    "lowlevel_in_channels": 256,
    "lowlevel_channels": 48,
    "atrous_rates": [6, 12, 18],
    "dropout_ratio": 0.1,
    "num_classes": 2,  # 虽然没用，但 SepASPP 可能需要
    "norm_layer": "SyncBatchNorm",
    "align_corners": False,
}

# define network: use UNetFormerOutDict and UnetFormerLoss
net = InstanceOffsetNet(
    backbone=backbone_cfg, decoder=decoder_cfg, roi_output_size=7, head_features=256
)

# define the loss
loss = ConfidenceWeightedL1Loss(reduction="mean")

# define the dataloader

train_dataset = TeqInstanceDataset(
    data_root="../data/segmentation/Turkey/Islahiye/pre/train",
    mode="train",
    transform=get_training_transform(),  # 使用新的 aug
    instance_dir="instances",  # 指定实例数据目录
)

val_dataset = TeqInstanceDataset(
    data_root="../data/segmentation/Turkey/Islahiye/pre/test",
    mode="test",
    transform=get_val_transform(),  # 使用新的 aug
    instance_dir="instances",  # 指定实例数据目录
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=train_batch_size,
    num_workers=0,
    pin_memory=True,
    shuffle=True,
    drop_last=True,
    collate_fn=instance_collate_fn,  # ！！！使用自定义 collate_fn！！！
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=val_batch_size,
    num_workers=0,
    shuffle=False,
    pin_memory=True,
    drop_last=False,
    collate_fn=instance_collate_fn,  # ！！！使用自定义 collate_fn！！！
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
