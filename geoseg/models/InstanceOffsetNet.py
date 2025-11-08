# 创建新文件 geoseg/models/InstanceOffsetNet.py

from typing import Dict, List

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from torchvision.ops import roi_align

from geoseg.models.backbones.resnet import get_convnet
from geoseg.models.decoders.sep_aspp import SepASPP
from geoseg.models.convs import DepthwiseSeparableConv
from tools.modules import init_weight


class InstanceOffsetNet(nn.Module):
    """
    实例偏移回归网络

    复用 DeepLabV3+ 的 Backbone 和 Neck (ASPP + Decoder) 来提取
    融合了高低层特征的丰富特征图 (Issue 4)。

    然后使用 RoIAlign 和一个回归头来预测每个实例的偏移量。
    """

    def __init__(
        self,
        backbone: Dict,
        decoder: Dict,
        roi_output_size: int = 7,
        head_features: int = 256,
    ):
        super().__init__()

        # -----------------------------------------------------------------
        # 1. 构建 DeepLabV3+ 特征提取器 (复用 v6_deeplab.py 和 Deeplabv3plus.py 的逻辑)
        # -----------------------------------------------------------------
        self.align_corners = decoder["align_corners"]
        BN_op = getattr(nn, decoder["norm_layer"])
        channels = decoder["channels"]  # 256

        self.backbone = get_convnet(**backbone)

        decoder_cfg = decoder.copy()
        decoder_cfg.pop("type")  # SepASPP 不接受 'type' 参数
        self.aspp = SepASPP(**decoder_cfg)

        self.projector = nn.Sequential(
            nn.Conv2d(
                decoder["lowlevel_in_channels"],  # 256
                decoder["lowlevel_channels"],  # 48
                kernel_size=1,
                bias=False,
            ),
            BN_op(decoder["lowlevel_channels"]),
            nn.ReLU(inplace=True),
        )

        self.pre_classifier = DepthwiseSeparableConv(
            decoder["norm_layer"],
            channels + decoder["lowlevel_channels"],  # 256 + 48
            channels,  # 256
            3,
            padding=1,
        )

        # ！！！我们复用 DeepLab 的所有权重初始化！！！
        init_weight(self.projector)
        init_weight(self.pre_classifier)

        # ！！！特征图在融合后 (pre_classifier 之后) 的步幅是 4 ！！！
        self.feature_map_stride = 4

        # -----------------------------------------------------------------
        # 2. 构建 RoI 回归头
        # -----------------------------------------------------------------
        self.roi_output_size = roi_output_size

        # (Channels * H * W) + (2 for centroid)
        # 融合后的特征通道数为 'channels' (256)
        mlp_input_dims = (channels * roi_output_size * roi_output_size) + 2

        self.head_mlp = nn.Sequential(
            nn.Linear(mlp_input_dims, head_features),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(head_features, head_features),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        )

        # 3. 最终的回归层，输出 (dx_ratio, dy_ratio)
        self.offset_regressor = nn.Linear(head_features, 2)

        init_weight(self.head_mlp)
        init_weight(self.offset_regressor)

    def forward(
        self, image: Tensor, bboxes_list: List[Tensor], centroids_list: List[Tensor]
    ) -> Tensor:

        # 1. 提取融合特征图 (Issue 4)
        # (复用 Deeplabv3plus.py 的 forward)
        out = self.backbone(image)
        lowlevel_feature = self.projector(out[1])  # [B, 48, H/4, W/4]

        output = self.aspp(out[4])  # [B, 256, H/16, W/16]
        output = F.interpolate(
            output,
            size=lowlevel_feature.shape[-2:],
            mode="bilinear",
            align_corners=self.align_corners,
        )  # [B, 256, H/4, W/4]

        output = torch.cat([lowlevel_feature, output], dim=1)  # [B, 304, H/4, W/4]
        fused_feature_map = self.pre_classifier(output)  # [B, 256, H/4, W/4]

        # 2. RoIAlign
        spatial_scale = 1.0 / self.feature_map_stride

        pooled_features = roi_align(
            fused_feature_map,
            bboxes_list,
            output_size=(self.roi_output_size, self.roi_output_size),
            spatial_scale=spatial_scale,
            aligned=True,
        )
        # pooled_features.shape = [N_total, 256, 7, 7]

        # 3. 准备质心特征 (Issue 5)
        # (这个逻辑仍然有价值，作为一种位置编码)
        centroids_batch = torch.cat(centroids_list, dim=0)  # [N_total, 2]
        bboxes_batch = torch.cat(bboxes_list, dim=0)  # [N_total, 4]

        x1, y1, x2, y2 = (
            bboxes_batch[:, 0],
            bboxes_batch[:, 1],
            bboxes_batch[:, 2],
            bboxes_batch[:, 3],
        )
        cx, cy = centroids_batch[:, 0], centroids_batch[:, 1]

        bw = x2 - x1
        bh = y2 - y1

        bw[bw == 0] = 1e-6
        bh[bh == 0] = 1e-6

        # 质心在BBox内的归一化坐标
        norm_cx = (cx - x1) / bw
        norm_cy = (cy - y1) / bh

        centroid_features = torch.stack([norm_cx, norm_cy], dim=1)  # [N_total, 2]

        # 4. 展平 RoI 特征
        pooled_features_flat = pooled_features.flatten(start_dim=1)

        # 5. 连接特征
        combined_features = torch.cat([pooled_features_flat, centroid_features], dim=1)

        # 6. 通过头部进行预测
        x = self.head_mlp(combined_features)
        predicted_offsets = self.offset_regressor(x)  # [N_total, 2]

        return predicted_offsets
