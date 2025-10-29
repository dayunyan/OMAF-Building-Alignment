import itertools
import os
from typing import List
import cv2
import numpy as np
from scipy import ndimage
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .soft_ce import SoftCrossEntropyLoss
from .joint_loss import JointLoss
from .dice import DiceLoss
from .emd import diff_num_emdloss, soft_emd_loss
from .variance import IntraClassConsistencyLoss
from tools.offset import offset_tensor_v3, compute_distance_transform
from tools.visual import save_tensor_as_png, visualize_masks


# class OffsetDistributionLoss(nn.Module):
#     def __init__(self, mean_x=0.0, std_x=1.0, mean_y=0.0, std_y=1.0, bins=100):
#         """
#         初始化损失函数，设置 x 和 y 轴的正态分布的均值和标准差。

#         参数:
#         mean_x (float): x 轴正态分布的均值，默认为 0.0
#         std_x (float): x 轴正态分布的标准差，默认为 1.0
#         mean_y (float): y 轴正态分布的均值，默认为 0.0
#         std_y (float): y 轴正态分布的标准差，默认为 1.0
#         bins (int): 直方图的 bins 数量，默认为 100
#         """
#         super(OffsetDistributionLoss, self).__init__()
#         self.mean_x = torch.tensor(mean_x)
#         self.std_x = torch.tensor(std_x)
#         self.mean_y = torch.tensor(mean_y)
#         self.std_y = torch.tensor(std_y)
#         self.bins = bins
#         self.kldiv_loss = nn.KLDivLoss(reduction="batchmean")
#         self.eps = torch.tensor(1e-10)

#     def forward(self, pred: Tensor, offset: Tensor):
#         """
#         计算输入的 offset 张量与设定的正态分布之间的 KL 散度。

#         参数:
#         offset (torch.Tensor): 形状为 [B, 2, H, W] 的张量，其中两个通道分别表示在 x 和 y 方向上的偏移量

#         返回:
#         torch.Tensor: 计算得到的 KL 散度损失
#         """
#         B, C, H, W = offset.shape
#         assert C == 2, "Offset 张量的通道数应为 2"
#         # 分离 x 和 y 方向的偏移量
#         offset_x = offset[:, 0:1, :, :]
#         offset_y = offset[:, 1:2, :, :]
#         # 计算前景区域offset的分布损失
#         foreground_indices = torch.nonzero(pred)
#         if foreground_indices.numel() == 0:  # 处理没有前景区域的情况
#             return torch.tensor(0.0).to(offset.device)
#         else:
#             offset_x = offset_x[pred == 1]
#             offset_y = offset_y[pred == 1]
#             # 计算 offset 的直方图
#             hist_x = torch.histc(
#                 torch.round(W * offset_x), bins=self.bins, min=-W, max=W
#             )
#             hist_y = torch.histc(
#                 torch.round(H * offset_y), bins=self.bins, min=-H, max=H
#             )
#             # 归一化直方图为概率分布
#             prob_hist_x = hist_x / torch.sum(hist_x)
#             prob_hist_y = hist_y / torch.sum(hist_y)
#             # 创建模拟的正态分布概率分布
#             x_range = torch.linspace(-W, W, self.bins)
#             y_range = torch.linspace(-H, H, self.bins)
#             normal_dist_x = torch.distributions.Normal(self.mean_x, self.std_x)
#             normal_dist_y = torch.distributions.Normal(self.mean_y, self.std_y)
#             prob_normal_x = torch.exp(normal_dist_x.log_prob(x_range))
#             prob_normal_y = torch.exp(normal_dist_y.log_prob(y_range))
#             # 归一化正态分布概率分布
#             prob_normal_x = prob_normal_x / torch.sum(prob_normal_x)
#             prob_normal_y = prob_normal_y / torch.sum(prob_normal_y)
#             # 计算 KL 散度
#             kl_divergence_x = self.kldiv_loss(
#                 torch.log(prob_hist_x + self.eps), prob_normal_x.to(offset.device)
#             )
#             kl_divergence_y = self.kldiv_loss(
#                 torch.log(prob_hist_y + self.eps), prob_normal_y.to(offset.device)
#             )
#             # 平均 KL 散度损失
#             kl_divergence = (kl_divergence_x + kl_divergence_y) / 2.0
#             return kl_divergence


class OffsetDistributionLoss(nn.Module):
    def __init__(self, mean_x=0.0, std_x=1.0, mean_y=0.0, std_y=1.0):
        """
        初始化损失函数，设置 x 和 y 轴的正态分布的均值和标准差。

        参数:
        mean_x (float): x 轴正态分布的均值，默认为 0.0
        std_x (float): x 轴正态分布的标准差，默认为 1.0
        mean_y (float): y 轴正态分布的均值，默认为 0.0
        std_y (float): y 轴正态分布的标准差，默认为 1.0
        """
        super(OffsetDistributionLoss, self).__init__()
        self.mean_x = torch.tensor(mean_x)
        self.std_x = torch.tensor(std_x)
        self.mean_y = torch.tensor(mean_y)
        self.std_y = torch.tensor(std_y)
        self.eps = torch.tensor(1e-10)

    def forward(self, pred: Tensor, offset: Tensor):
        """
        计算输入的 offset 张量与设定的正态分布之间的 KL 散度。

        参数:
        offset (torch.Tensor): 形状为 [B, 2, H, W] 的张量，其中两个通道分别表示在 x 和 y 方向上的偏移量

        返回:
        torch.Tensor: 计算得到的 KL 散度损失
        """
        B, C, H, W = offset.shape
        assert C == 2, "Offset 张量的通道数应为 2"
        # 分离 x 和 y 方向的偏移量
        offset_x = offset[:, 0:1, :, :]
        offset_y = offset[:, 1:2, :, :]
        # 计算前景区域offset的分布损失
        foreground_indices = torch.nonzero(pred)
        if foreground_indices.numel() == 0:  # 处理没有前景区域的情况
            return torch.tensor(0.0).to(offset.device)
        else:
            offset_x = offset_x[pred == 1]
            offset_y = offset_y[pred == 1]
            # 计算输入的 offset 分布的均值和标准差
            mean_offset_x = torch.mean(offset_x)
            std_offset_x = torch.std(offset_x)
            mean_offset_y = torch.mean(offset_y)
            std_offset_y = torch.std(offset_y)
            # 计算 KL 散度，KL(p || q) = log(std_q / std_p) + (std_p^2 + (mean_p - mean_q)^2) / (2 * std_q^2) - 0.5
            kl_divergence_x = (
                torch.log(self.std_x / (std_offset_x + self.eps))
                + (
                    (std_offset_x**2 + (mean_offset_x - self.mean_x) ** 2)
                    / (2 * self.std_x**2)
                )
                - 0.5
            )
            kl_divergence_y = (
                torch.log(self.std_y / (std_offset_y + self.eps))
                + (
                    (std_offset_y**2 + (mean_offset_y - self.mean_y) ** 2)
                    / (2 * self.std_y**2)
                )
                - 0.5
            )
            # 平均 KL 散度损失
            kl_divergence = (kl_divergence_x + kl_divergence_y) / 2.0
            return kl_divergence


class OffsetUsefulLoss(nn.Module):
    def __init__(self, mean_x=0.0, std_x=1.0, mean_y=0.0, std_y=1.0, ignore_index=255):
        super().__init__()
        self.main_loss = JointLoss(
            SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
            DiceLoss(smooth=0.05, ignore_index=ignore_index),
            1.0,
            1.0,
        )
        self.aux_loss = SoftCrossEntropyLoss(
            smooth_factor=0.05, ignore_index=ignore_index
        )
        self.kl_loss = OffsetDistributionLoss(mean_x, std_x, mean_y, std_y)

    def forward(self, logits: Tensor, offset: Tensor, labels: Tensor):
        """计算offset损失

        Args:
            logits (Tensor): 模型输出(batch_size, 4, h, w) | Tuple[(batch_size, 4, h, w),(batch_size, 4, h, w)]
            offsets (Tensor): 原始图像(batch_size, 2, h, w)
            offset_labels (Tensor): 偏移的标签(batch_size, h, w)
        """
        if self.training and len(logits) == 2:
            logit_main, logit_aux = logits

            # 判断logits中判别为前景的区域
            pred = ((logit_main + logit_aux) / 2).argmax(dim=1, keepdim=True)
            logit_main = _offset_logits(logit_main, offset)
            logit_aux = _offset_logits(logit_aux, offset)

            offset = torch.where(pred.expand(-1, 2, -1, -1) == 1, offset, 0)
            loss_kl = self.kl_loss(pred, offset)
            loss_main = self.main_loss(logit_main, labels) + 0.4 * self.aux_loss(
                logit_aux, labels
            )

            return {
                "loss": loss_main + 0.5 * loss_kl,
                "loss_main": loss_main,
                "loss_kl": loss_kl,
            }
        else:
            loss = self.main_loss(logits, labels)

            return {"loss": loss}


def _offset_logits(logits: torch.Tensor, offset: torch.Tensor):
    """
    此函数使用 offset 张量对 logits 张量中的像素位置进行移动。

    参数:
    logits (torch.Tensor): 形状为 [B, C, H, W] 的张量，表示模型输出的 logits
    offset (torch.Tensor): 形状为 [B, 2, H, W] 的张量，其中两个通道分别表示在 x 和 y 方向上的偏移量

    返回:
    torch.Tensor: 移动后的 logits 张量
    """
    if logits.dim() != 4:
        # 则传入的是labels
        B, H, W = logits.shape
        C = len(torch.unique(logits))
        logits = logits.unsqueeze(1).expand(-1, 2, -1, -1)
        pred = logits
    else:
        B, C, H, W = logits.shape
        pred = logits.argmax(dim=1, keepdim=True).expand(-1, 2, -1, -1)
    # 检查输入张量的形状是否符合预期
    assert offset.shape == (B, 2, H, W), "Offset 张量的形状应为 [B, 2, H, W]"
    new_offset = torch.where(pred == 1, offset, 0)
    # shifted_logits = torch.zeros_like(logits)
    # for idx in torch.nonzero(new_offset):
    #     if (
    #         idx[3] - torch.round(W * new_offset[idx[0], 0, idx[2], idx[3]]) >= 0
    #         and idx[3] - torch.round(W * new_offset[idx[0], 0, idx[2], idx[3]]) < W
    #         and idx[2] - torch.round(H * new_offset[idx[0], 1, idx[2], idx[3]]) >= 0
    #         and idx[2] - torch.round(H * new_offset[idx[0], 1, idx[2], idx[3]]) < H
    #     ):
    #         shifted_logits[
    #             idx[0],
    #             :,
    #             idx[2]
    #             - torch.round(H * new_offset[idx[0], 1, idx[2], idx[3]]).to(
    #                 torch.int32
    #             ),
    #             idx[3]
    #             - torch.round(W * new_offset[idx[0], 0, idx[2], idx[3]]).to(
    #                 torch.int32
    #             ),
    #         ] = logits[idx[0], :, idx[2], idx[3]]
    # return shifted_logits
    # new_offset = torch.where(pred == 1, offset, torch.zeros_like(offset))

    # 生成坐标网格
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    y = y.unsqueeze(0).expand(B, -1, -1).to(logits.device)
    x = x.unsqueeze(0).expand(B, -1, -1).to(logits.device)
    # 计算新的坐标
    new_y = (y - torch.round(H * new_offset[:, 1, :, :])).to(torch.int64)
    new_x = (x - torch.round(W * new_offset[:, 0, :, :])).to(torch.int64)
    # 确保新坐标在合法范围内
    new_y = torch.clamp(new_y, 0, H - 1).unsqueeze(1).expand(-1, C, -1, -1)
    new_x = torch.clamp(new_x, 0, W - 1).unsqueeze(1).expand(-1, C, -1, -1)
    # print(f"new_y: {new_y},\n new_x: {new_x}")
    # 初始化 shifted_logits
    shifted_logits_y = logits.clone()
    # 使用广播机制更新 shifted_logits
    shifted_logits_y.scatter_(
        2,
        new_y,
        logits,  # .gather(2, new_y.unsqueeze(1).expand(-1, C, -1, -1)),
    )
    # print(f"shifted_logits: {shifted_logits_y}")
    new_x_shifted = torch.zeros_like(new_x)
    new_x_shifted.scatter_(2, new_y, new_x)
    # print(f"new_x_shifted: {new_x_shifted}")
    shifted_logits_xy = logits.clone()
    shifted_logits_xy.scatter_(
        3,
        new_x_shifted,
        shifted_logits_y,  # .gather(3, new_x.unsqueeze(1).expand(-1, C, -1, -1)),
    )
    # print(f"shifted_logits: {shifted_logits_xy}")
    return shifted_logits_xy


def entropy(tensor: Tensor):
    """
    计算输入张量的熵。

    参数:
    tensor (torch.Tensor): 输入张量，形状为 [B, 3, H, W]

    返回:
    torch.Tensor: 熵张量，形状为 [B]
    """
    B, C, H, W = tensor.shape
    # 将张量展平为 [B, C, H * W]
    tensor = tensor.view(B, C, H * W)
    # 计算每个像素值的概率分布，假设像素值范围为 [0, 255]
    histograms = torch.histc(tensor.float(), bins=256, min=0, max=255)
    # 归一化直方图以得到概率分布
    probabilities = histograms / (H * W)
    # 避免 log(0) 的情况
    probabilities = torch.where(probabilities > 0, probabilities, torch.tensor(1e-10))
    # 计算熵
    entropy = -torch.sum(probabilities * torch.log2(probabilities), dim=-1)
    # 对三个通道求平均熵
    entropy = torch.mean(entropy, dim=1)
    return entropy


class OffsetUsefulLoss2(nn.Module):
    def __init__(self, lambda_l1=0.01, lambda_l2=0.001, ignore_index=255):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.main_loss = JointLoss(
            SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
            DiceLoss(smooth=0.05, ignore_index=ignore_index),
            1.0,
            1.0,
        )
        self.aux_loss = SoftCrossEntropyLoss(
            smooth_factor=0.05, ignore_index=ignore_index
        )

    def forward(self, logits: Tensor, offset: Tensor, labels: Tensor):
        """计算offset损失

        Args:
            logits (Tensor): 模型输出(batch_size, 4, h, w) | Tuple[(batch_size, 4, h, w),(batch_size, 4, h, w)]
            offsets (Tensor): 原始图像(batch_size, 2, h, w)
            offset_labels (Tensor): 偏移的标签(batch_size, h, w)
        """
        if self.training and len(logits) == 2:
            logit_main, logit_aux = logits

            # 判断logits中判别为前景的区域
            pred = ((logit_main + logit_aux) / 2).argmax(dim=1, keepdim=True)
            logit_main = _offset_logits(logit_main, offset)
            logit_aux = _offset_logits(logit_aux, offset)

            offset = torch.where(pred.expand(-1, 2, -1, -1) == 1, offset, 0)
            loss_l1 = torch.norm(offset, p=1) * self.lambda_l1
            loss_l2 = torch.norm(offset, p=2) * self.lambda_l2
            loss_main = self.main_loss(logit_main, labels) + 0.4 * self.aux_loss(
                logit_aux, labels
            )

            return {
                "loss": loss_main + loss_l1 + loss_l2,
                "loss_main": loss_main,
                "loss_l1": loss_l1,
                "loss_l2": loss_l2,
            }
        else:
            loss = self.main_loss(logits, labels)

            return {"loss": loss}


class OffsetUsefulLossWithL1(OffsetUsefulLoss2):
    def __init__(self, lambda_l1=0.01, ignore_index=255):
        super().__init__(lambda_l1=lambda_l1, lambda_l2=0.0, ignore_index=ignore_index)

    def forward(self, logits: Tensor, offset: Tensor, labels: Tensor):
        """计算offset损失
        Args:
            logits (Tensor): 模型输出(batch_size, 4, h, w) | Tuple[(batch_size, 4, h, w),(batch_size, 4, h, w)]
            offsets (Tensor): 原始图像(batch_size, 2, h, w)
            offset_labels (Tensor): 偏移的标签(batch_size, h, w)
        """
        if self.training and len(logits) == 2:
            logit_main, logit_aux = logits
            B, C, H, W = logit_main.shape

            pred = ((logit_main + logit_aux) / 2).argmax(dim=1, keepdim=True)
            logit_main = _offset_logits(logit_main, offset)
            logit_aux = _offset_logits(logit_aux, offset)

            offset = torch.where(pred.expand(-1, 2, -1, -1) == 1, offset, 0)
            loss_l1 = torch.norm(offset, p=1) * self.lambda_l1
            loss_l2 = torch.norm(offset, p=2) * self.lambda_l2
            loss_main = self.main_loss(logit_main, labels) + 0.4 * self.aux_loss(
                logit_aux, labels
            )
            # 将标签转换为 one-hot 编码 [B, H, W] -> [B, H, W, C] -> [B, C, H, W]
            labels_one_hot = (
                torch.nn.functional.one_hot(labels, num_classes=C)
                .permute(0, 3, 1, 2)
                .float()
            )
            # 计算 L1 损失（绝对差均值）
            l1_loss = torch.nn.functional.l1_loss(logit_main, labels_one_hot)
            # 判断logits中判别为前景的区域

            return {
                "loss": loss_main + loss_l1 + loss_l2 + l1_loss,
                "loss_main": loss_main,
                "loss_l1": loss_l1,
                "loss_l2": loss_l2,
                "l1_loss": l1_loss,
            }
        else:
            loss = self.main_loss(logits, labels)

            return {"loss": loss}


class Offset_v2(nn.Module):
    def __init__(self, lambda1=0.01, lambda2=0.001, beta=-0.5, ignore_index=255):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.beta = beta
        self.main_loss = JointLoss(
            SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
            DiceLoss(smooth=0.05, ignore_index=ignore_index),
            1.0,
            1.0,
        )
        self.aux_loss = SoftCrossEntropyLoss(
            smooth_factor=0.05, ignore_index=ignore_index
        )

    def forward(self, logits: Tensor, offset: Tensor, labels: Tensor):
        """计算offset损失

        Args:
            logits (Tensor): 模型输出(batch_size, 4, h, w) | Tuple[(batch_size, 4, h, w),(batch_size, 4, h, w)]
            offsets (Tensor): 原始图像(batch_size, 2, h, w)
            offset_labels (Tensor): 偏移的标签(batch_size, h, w)
        """
        labels_32 = labels.to(torch.float32)
        if self.training and len(logits) == 2:
            logit_main, logit_aux = logits

            # 先让labels偏移
            labels_offset = offset_tensor_v3(
                labels_32.unsqueeze(1), offset, sample_mode="bilinear"
            ).squeeze(
                1
            )  # probability map

            # 计算EMD损失
            h, w = labels_offset.shape[-2:]
            loss_emd = soft_emd_loss(
                labels_offset,
                labels_32,
                scale_factor=0.125,
            )

            # 计算偏移损失
            # loss_offset_x = torch.abs(offset[:, 0, :, :]).sum()
            # loss_offset_y = torch.abs(offset[:, 1, :, :]).sum()
            # loss_offset_mean = ((1 + self.beta) ** 2) / (1 + offset.mean()) + (
            #     (1 - self.beta) ** 2
            # ) / (1 - offset.mean())

            # loss_offset_x = self.dynamic_range_loss(
            #     offset[:, 0, :, :],
            #     labels_offset,
            #     margin_low=0.005,
            #     margin_high=0.1,
            #     alpha=1.0,
            #     beta=1.0,
            # )
            # loss_offset_y = self.dynamic_range_loss(
            #     offset[:, 1, :, :],
            #     labels_offset,
            #     margin_low=0.005,
            #     margin_high=0.1,
            #     alpha=1.0,
            #     beta=1.0,
            # )

            # 计算交叉熵损失
            loss_main = self.main_loss(
                logit_main, labels_offset.to(torch.int64)
            ) + 0.4 * self.aux_loss(logit_aux, labels_offset.to(torch.int64))

            return {
                "loss": loss_main + 1e6 * loss_emd,
                # + self.lambda1 * loss_offset_x
                # + self.lambda2 * loss_offset_y
                # + 0.1 * loss_offset_mean,
                "loss_main": loss_main,
                "loss_emd": 1e6 * loss_emd,
                # "loss_x": self.lambda1 * loss_offset_x,
                # "loss_y": self.lambda2 * loss_offset_y,
                # "loss_off_m": 0.1 * loss_offset_mean,
            }
        else:
            labels_offset = offset_tensor_v3(
                labels_32.unsqueeze(1), offset, sample_mode="nearest"
            ).squeeze(1)
            # print(
            #     f"labels_offset: {labels_offset.shape},\n logits: {logits.shape},\n labels: {labels.shape}"
            # )
            loss = self.main_loss(logits, labels_offset.to(torch.int64))

            return {"loss": loss}

    def dynamic_range_loss(
        self, A, B, margin_low=0.03, margin_high=0.7, alpha=1.0, beta=1.0, eps=1e-6
    ):
        """
        A : 预测张量 [B,H,W] ∈ [-1,1]
        B : 二值标签 [B,H,W] ∈ {0,1}
        margin_low : 最小允许绝对值（建议0.01-0.05）
        margin_high : 最大允许绝对值（建议0.5-0.8）
        """

        # 区域划分
        mask_0 = B == 0
        mask_1 = B == 1

        # B=0区域损失（强制趋近0）
        loss_0 = torch.abs(A[mask_0]).mean()

        # B=1区域动态约束
        a_abs = torch.abs(A[mask_1].clamp(min=eps))  # 防止除零

        # 构造双曲排斥场
        tanh_scale = 10.0  # 控制梯度陡峭度
        low_repulsion = torch.tanh(tanh_scale * (margin_low - a_abs))  # 下界排斥
        high_repulsion = torch.tanh(tanh_scale * (a_abs - margin_high))  # 上界排斥

        # 组合动态损失
        loss_1 = (low_repulsion + high_repulsion).mean()

        # 总损失加权
        total_loss = alpha * loss_0 + beta * loss_1
        return total_loss


class DifferentiableOffsetEstimator(torch.nn.Module):
    def __init__(
        self, max_displacement=5, patch_size=5, stride=4, temperature=1.0, mode="avg"
    ):
        super().__init__()
        self.max_d = max_displacement
        self.patch_size = patch_size
        self.stride = stride
        self.temperature = temperature
        self.mode = mode

        # 创建所有可能的位移向量
        displacements = list(
            itertools.product(
                range(-max_displacement, max_displacement + 1),
                range(-max_displacement, max_displacement + 1),
            )
        )
        self.displacements = torch.tensor(displacements, dtype=torch.float32)
        self.n_candidates = len(displacements)

        # 预计算索引映射
        self.register_buffer("center_indices", None)
        self.register_buffer("candidate_indices", None)

    def precompute_indices(self, search_size):
        """预计算中心区域和候选区域的索引"""
        # 中心区域索引 (在搜索窗口中)
        center_indices = []
        for i in range(self.max_d, self.max_d + self.patch_size):
            for j in range(self.max_d, self.max_d + self.patch_size):
                center_indices.append(i * search_size + j)

        # 所有候选位移的索引
        candidate_indices_list = []
        for dx, dy in self.displacements:
            # 计算候选区域起始位置
            start_x = self.max_d - int(dy)
            start_y = self.max_d - int(dx)

            # 计算候选区域索引
            indices = []
            for i in range(start_x, start_x + self.patch_size):
                for j in range(start_y, start_y + self.patch_size):
                    indices.append(i * search_size + j)
            candidate_indices_list.append(indices)

        return torch.tensor(center_indices, dtype=torch.long), candidate_indices_list

    def forward(self, gt_mask, offset_mask):
        B, C, H, W = gt_mask.shape
        device = gt_mask.device

        # 计算搜索区域大小和填充
        search_size = self.patch_size + 2 * self.max_d
        padding = self.max_d + self.patch_size // 2

        # 计算展开后的输出尺寸
        H_out = (H + 2 * padding - search_size) // self.stride + 1
        W_out = (W + 2 * padding - search_size) // self.stride + 1
        total_positions = H_out * W_out

        # 1. 使用相同参数展开两个掩码
        unfold_kwargs = {
            "kernel_size": search_size,
            "padding": padding,
            "stride": self.stride,
        }

        # 统一展开操作
        gt_unfolded = F.unfold(
            gt_mask, **unfold_kwargs
        )  # [B, search_size², H_out*W_out]
        offset_unfolded = F.unfold(
            offset_mask, **unfold_kwargs
        )  # [B, search_size², H_out*W_out]

        # 2. 确保维度正确
        assert gt_unfolded.size(2) == total_positions
        assert offset_unfolded.size(2) == total_positions

        # 3. 预计算索引（如果是第一次或尺寸变化）
        if (
            self.center_indices is None
            or self.candidate_indices is None
            or self.center_indices.numel() != self.patch_size**2
        ):
            self.center_indices, candidate_indices_list = self.precompute_indices(
                search_size
            )
            self.candidate_indices = torch.tensor(
                candidate_indices_list,
                dtype=torch.long,
                device=device,
            )

        # 4. 提取中心区域特征
        center_patches = torch.index_select(
            gt_unfolded, 1, self.center_indices.to(device)
        )  # [B, patch_size², total_positions]

        # 5. 为每个候选位移计算相似度
        similarity_scores = torch.zeros(
            B, self.n_candidates, total_positions, device=device
        )

        for k, indices in enumerate(self.candidate_indices):
            # 确保索引设备一致
            indices = indices.to(device)

            # 提取候选区域特征
            candidate_patches = torch.index_select(
                offset_unfolded, 1, indices
            )  # [B, patch_size², total_positions]

            # 检查维度一致性
            assert center_patches.shape == candidate_patches.shape

            # 计算相似度（负的L1距离）
            patch_similarity = -torch.abs(center_patches - candidate_patches).mean(
                dim=1
            )
            similarity_scores[:, k] = patch_similarity

        # 6. 使用softmax计算候选位移的概率分布
        probs = F.softmax(similarity_scores / self.temperature, dim=1)

        if self.mode == "avg":
            # 计算每个位置的加权位移
            displacements_x = self.displacements[:, 0].to(device)[
                None, :, None
            ]  # [1, n_candidates, 1]
            displacements_y = self.displacements[:, 1].to(device)[
                None, :, None
            ]  # [1, n_candidates, 1]
            dx_vectors = (probs * displacements_x).sum(dim=1)  # [B, total_positions]
            dy_vectors = (probs * displacements_y).sum(dim=1)  # [B, total_positions]
        elif self.mode == "max":
            # 选择概率最大的位移作为输出
            max_indices = probs.argmax(dim=1).cpu()  # [B, total_positions]
            dx_vectors = self.displacements[max_indices, 0].to(
                device
            )  # [B, total_positions]
            dy_vectors = self.displacements[max_indices, 1].to(
                device
            )  # [B, total_positions]
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        # 计算每个位置的置信度（作为权重）
        confidences = F.softmax(probs.max(dim=1)[0], dim=1)  # [B, total_positions]

        # 重构位移场 (考虑重叠区域)
        # 创建网格映射每个输出位置到原始图像的像素
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij",
        )  # [H, W]

        # 初始化位移和权重累加图
        dx_map = torch.zeros(B, H, W, device=device)
        dy_map = torch.zeros(B, H, W, device=device)
        weight_map = torch.zeros(B, H, W, device=device)

        # 为每个滑动窗口位置
        for i in range(H_out):
            for j in range(W_out):
                # 计算窗口在原始图像中的位置
                h_start = i * self.stride
                w_start = j * self.stride

                # 中心区域的坐标范围
                h_center = h_start + self.max_d
                w_center = w_start + self.max_d

                # 创建中心区域掩码
                h_min = h_center
                h_max = min(h_center + self.patch_size, H)
                w_min = w_center
                w_max = min(w_center + self.patch_size, W)

                # 跳过空区域
                if h_min >= h_max or w_min >= w_max:
                    continue

                # 当前窗口索引
                pos_idx = i * W_out + j

                # 中心区域的位移和置信度
                dx_val = dx_vectors[:, pos_idx]  # [B]
                dy_val = dy_vectors[:, pos_idx]  # [B]
                conf_val = confidences[:, pos_idx]  # [B]

                # 更新位移图 (加权累加)
                for b in range(B):
                    dx_map[b, h_min:h_max, w_min:w_max] += dx_val[b] * conf_val[b]
                    dy_map[b, h_min:h_max, w_min:w_max] += dy_val[b] * conf_val[b]
                    weight_map[b, h_min:h_max, w_min:w_max] += conf_val[b]

        # 加权平均处理重叠区域
        # valid_mask = weight_map > 1e-8
        # dx_map = torch.where(valid_mask, dx_map / weight_map, 0)
        # dy_map = torch.where(valid_mask, dy_map / weight_map, 0)
        dx_map = dx_map / (weight_map + 1e-8)
        dy_map = dy_map / (weight_map + 1e-8)

        return dx_map.unsqueeze(1) / W, dy_map.unsqueeze(1) / H


class ObjectAwareOffsetEstimator(nn.Module):
    def __init__(
        self,
        max_displacement=5,
        patch_size=5,
        stride=1,
        temperature=1.0,
        max_d_x=0,
        min_d_x=-20,
        max_d_y=0,
        min_d_y=-60,
    ):
        super().__init__()
        self.max_d = max_displacement
        self.patch_size = patch_size
        self.stride = stride
        self.temperature = temperature
        self.max_d_x = max_d_x
        self.min_d_x = min_d_x
        self.max_d_y = max_d_y
        self.min_d_y = min_d_y

        # 位移向量
        displacements = list(
            itertools.product(
                range(-max_displacement, max_displacement + 1),
                range(-max_displacement, max_displacement + 1),
            )
        )
        self.displacements = torch.tensor(displacements, dtype=torch.float32)
        self.n_candidates = len(displacements)

        # 预计算索引映射
        self.register_buffer("center_indices", None)
        self.register_buffer("candidate_indices", None)

    def detect_objects(self, mask):
        """检测分割图中的对象实例"""
        # 使用连通组件分析检测对象
        with torch.no_grad():
            # 二值化分割图
            binary_mask = (mask > 0.5).float()

            # 转换为CPU numpy数组进行连通组件分析
            np_mask = binary_mask.squeeze(1).cpu().numpy()
            instance_maps = []
            for i in range(np_mask.shape[0]):
                labeled, n_objects = ndimage.label(np_mask[i])
                instance_maps.append(torch.from_numpy(labeled).to(mask.device))

            instance_map = torch.stack(instance_maps, dim=0).unsqueeze(1)

        return instance_map

    def match_objects(self, gt_objects, offset_objects, gt_features, offset_features):
        """改进的对象匹配算法"""
        B, _, H, W = gt_objects.shape
        matched_displacements = torch.zeros(B, H, W, 2, device=gt_objects.device)
        match_confidence = torch.zeros(B, H, W, device=gt_objects.device)
        object_displacements = {}

        for b in range(B):
            # 获取对象ID并移除背景
            gt_ids = torch.unique(gt_objects[b][gt_objects[b] > 0])
            offset_ids = torch.unique(offset_objects[b][offset_objects[b] > 0])

            # 步骤1：建立潜在匹配池（基于空间重叠）
            potential_matches = {}
            for gt_id in gt_ids:
                gt_mask = gt_objects[b] == gt_id
                gt_area = gt_mask.sum().item()

                # 仅考虑有重叠的对象
                overlapping_objs = []
                for offset_id in offset_ids:
                    offset_mask = offset_objects[b] == offset_id
                    overlap = (gt_mask & offset_mask).sum().item()

                    if overlap > 0:
                        overlap_ratio = overlap / (gt_area + 1e-5)
                        overlapping_objs.append((offset_id, overlap_ratio))

                # 按重叠比例排序并保留前N个候选
                overlapping_objs.sort(key=lambda x: x[1], reverse=True)
                potential_matches[gt_id] = overlapping_objs[:3]  # 最多3个候选

            # 步骤2：多特征融合的匹配决策
            matched_pairs = {}
            for gt_id, candidates in potential_matches.items():
                if not candidates:
                    continue

                gt_mask = (gt_objects[b] == gt_id).float()
                gt_feat = gt_features[b] * gt_mask

                best_match_id = -1
                best_similarity = 0

                # 评估所有候选
                for offset_id, overlap_ratio in candidates:
                    offset_mask = (offset_objects[b] == offset_id).float()
                    offset_feat = offset_features[b] * offset_mask

                    # 特征相似度
                    feat_sim = torch.sum(gt_feat * offset_feat) / (
                        torch.norm(gt_feat) * torch.norm(offset_feat) + 1e-8
                    )

                    # 位移一致性
                    gt_centroid = self.compute_centroid(gt_mask)
                    offset_centroid = self.compute_centroid(offset_mask)
                    displacement = (
                        offset_centroid[1] - gt_centroid[1],
                        offset_centroid[0] - gt_centroid[0],
                    )

                    # 位移在合理范围内才考虑
                    if not (
                        self.min_d_x <= displacement[0] <= self.max_d_x
                        and self.min_d_y <= displacement[1] <= self.max_d_y
                    ):
                        continue

                    # 形状相似度（IoU）
                    union = ((gt_mask != 0) | (offset_mask != 0)).sum()
                    intersection = ((gt_mask != 0) & (offset_mask != 0)).sum()
                    iou = intersection / (union + 1e-8)

                    # 综合评分（权重可调整）
                    similarity_score = 0.4 * feat_sim + 0.4 * overlap_ratio + 0.2 * iou

                    if similarity_score > best_similarity:
                        best_similarity = similarity_score
                        best_match_id = offset_id
                        best_displacement = displacement

                # 有效匹配
                if best_similarity > 0.3 and best_match_id != -1:
                    matched_pairs[(gt_id, best_match_id)] = best_displacement
                    object_displacements[(b, gt_id.item())] = torch.tensor(
                        best_displacement, device=gt_objects.device
                    )
                    object_displacements[(b, best_match_id.item())] = torch.tensor(
                        best_displacement, device=gt_objects.device
                    )

            # 步骤3：处理分裂/融合的对象
            # 寻找在GT中是单一对象但在offset中被分割的情况
            for offset_id in offset_ids:
                if any(offset_id.item() in pair for pair in matched_pairs):
                    continue  # 已匹配

                offset_mask = offset_objects[b] == offset_id

                # 查找与该偏移对象重叠的GT对象
                overlapping_gts = []
                for gt_id in gt_ids:
                    gt_mask = gt_objects[b] == gt_id
                    if (gt_mask & offset_mask).sum() > 0:
                        overlapping_gts.append(gt_id)

                # 如果与多个GT对象重叠（可能是对象分裂）
                if len(overlapping_gts) > 1:
                    # 计算联合质心位移
                    union_mask = torch.zeros_like(gt_objects[b])
                    for gt_id in overlapping_gts:
                        gt_mask = gt_objects[b] == gt_id
                        union_mask = union_mask | gt_mask

                    # 计算联合质心
                    union_centroid = self.compute_centroid(union_mask)
                    offset_centroid = self.compute_centroid(offset_mask)
                    displacement = (
                        offset_centroid[1] - union_centroid[1],
                        offset_centroid[0] - union_centroid[0],
                    )

                    # 如果位移在合理范围内，应用位移
                    if (
                        self.min_d_x <= displacement[0] <= self.max_d_x
                        and self.min_d_y <= displacement[1] <= self.max_d_y
                    ):
                        object_displacements[(b, offset_id.item())] = torch.tensor(
                            displacement, device=gt_objects.device
                        )

            # 步骤4：应用位移到对象区域
            for (gt_id, offset_id), displacement in matched_pairs.items():
                gt_mask = gt_objects[b] == gt_id
                offset_mask = offset_objects[b] == offset_id

                # 使用联合区域应用位移
                union_mask = gt_mask | offset_mask
                matched_displacements[b, ..., 0][union_mask.squeeze()] = displacement[0]
                matched_displacements[b, ..., 1][union_mask.squeeze()] = displacement[1]
                match_confidence[b, union_mask.squeeze()] = 0.8  # 高置信度

        return matched_displacements, match_confidence, object_displacements

    def compute_centroid(self, mask):
        """鲁棒的质心计算方法"""
        device = mask.device
        if mask.sum() < 10:  # 小对象使用简单平均
            y_indices, x_indices = torch.where(mask.squeeze())
            if len(y_indices) == 0:
                return 0.0, 0.0
            return y_indices.float().mean().item(), x_indices.float().mean().item()
        else:  # 大对象使用形态学中心
            # 创建距离变换图
            mask_np = mask.squeeze().cpu().numpy().astype(np.uint8)
            dist = cv2.distanceTransform(mask_np, cv2.DIST_L2, 3)
            max_loc = np.unravel_index(np.argmax(dist), dist.shape)
            return max_loc[0], max_loc[1]  # (y, x)

    def propagate_displacements(
        self, matched_displacements, match_confidence, gt_objects, offset_objects
    ):
        """传播位移到未匹配的对象"""
        B, H, W, _ = matched_displacements.shape
        object_displacements = {}

        for b in range(B):
            # 获取未匹配的对象
            matched_mask = match_confidence[b] > 0
            unmatched_mask = ~matched_mask & torch.logical_or(
                gt_objects[b].squeeze() > 0, offset_objects[b].squeeze() > 0
            )

            if not unmatched_mask.any():
                continue

            # 为未匹配对象分配平均位移
            avg_dx = matched_displacements[b, ..., 0][matched_mask].mean()
            avg_dy = matched_displacements[b, ..., 1][matched_mask].mean()

            # 如果没有匹配对象，使用随机位移
            if torch.isnan(avg_dx) or torch.isnan(avg_dy):
                avg_dx = (
                    torch.rand(1, device=matched_displacements.device)
                    * (self.max_d_x - self.min_d_x)
                    + self.min_d_x
                )
                avg_dy = (
                    torch.rand(1, device=matched_displacements.device)
                    * (self.max_d_y - self.min_d_y)
                    + self.min_d_y
                )

            # 收集未匹配对象的位移
            gt_ids = torch.unique(gt_objects[b])
            offset_ids = torch.unique(offset_objects[b])
            gt_ids = gt_ids[gt_ids != 0]
            offset_ids = offset_ids[offset_ids != 0]

            # 对于未匹配的GT对象
            for obj_id in gt_ids:
                obj_mask = gt_objects[b].squeeze() == obj_id
                if not match_confidence[b, obj_mask].any():
                    object_displacements[(b, obj_id.item())] = torch.tensor(
                        [avg_dx, avg_dy], device=obj_mask.device
                    )

            # 对于未匹配的offset对象
            for obj_id in offset_ids:
                obj_mask = offset_objects[b].squeeze() == obj_id
                if not match_confidence[b, obj_mask].any():
                    object_displacements[(b, obj_id.item())] = torch.tensor(
                        [avg_dx, avg_dy], device=obj_mask.device
                    )

            # 为未匹配对象分配平均位移
            matched_displacements[b, unmatched_mask] = torch.tensor(
                [avg_dx, avg_dy], device=matched_displacements.device
            )
            match_confidence[b, unmatched_mask] = 0.5  # 中等置信度

        return matched_displacements, match_confidence, object_displacements

    def generate_object_offset_map(self, offset_objects, object_displacements):
        """生成基于对象的位移图"""
        B, _, H, W = offset_objects.shape
        device = offset_objects.device
        dx_map = torch.zeros(B, 1, H, W, device=device)
        dy_map = torch.zeros(B, 1, H, W, device=device)
        # print(f"{object_displacements.keys()=}, {object_displacements.values()}")

        for b in range(B):
            # 获取当前batch的对象ID
            obj_ids = torch.unique(offset_objects[b])
            obj_ids = obj_ids[obj_ids != 0]

            for obj_id in obj_ids:
                # 只处理有位移值的对象
                key = (b, obj_id.item())
                if key not in object_displacements:
                    continue

                displacement = object_displacements[key]
                dx = displacement[0].item()
                dy = displacement[1].item()
                # print(f"{key=}, {dx=}, {dy=}")

                # 创建对象掩码
                obj_mask = (offset_objects[b] == obj_id).squeeze(0)  # [H, W]

                # 如果没有有效像素则跳过
                if not torch.any(obj_mask):
                    continue

                # 1. 创建原始位置掩码
                original_mask = obj_mask.clone()

                # 2. 创建偏移后位置掩码
                # 计算新位置坐标（整数坐标）
                y_indices, x_indices = torch.where(obj_mask)
                new_y = torch.clamp((y_indices.float() - dy).round().long(), 0, H - 1)
                new_x = torch.clamp((x_indices.float() - dx).round().long(), 0, W - 1)

                # 创建新位置的掩码
                shifted_mask = torch.zeros_like(obj_mask)
                shifted_mask[new_y, new_x] = True

                # 3. 计算并集区域
                union_mask = original_mask | shifted_mask

                # 4. 在并集区域应用位移
                dx_map[b, 0, union_mask] = dx
                dy_map[b, 0, union_mask] = dy

        return dx_map, dy_map

    def forward(self, gt_mask, offset_mask):
        # 检测对象实例
        gt_objects = self.detect_objects(gt_mask)  # [B,1,H,W]
        offset_objects = self.detect_objects(offset_mask)  # [B,1,H,W]

        # 匹配对象并计算位移
        features = torch.ones_like(gt_objects)
        matched_displacements, match_confidence, object_displacements = (
            self.match_objects(gt_objects, offset_objects, features, features)
        )

        # 传播位移到未匹配对象
        matched_displacements, match_confidence, all_displacements = (
            self.propagate_displacements(
                matched_displacements, match_confidence, gt_objects, offset_objects
            )
        )

        # 合并位移字典
        combined_displacements = {**object_displacements, **all_displacements}

        # 生成最终的基于对象的位移图
        dx_map, dy_map = self.generate_object_offset_map(
            offset_objects, combined_displacements
        )

        return dx_map, dy_map, match_confidence.unsqueeze(1)


import datetime


class Offset_v3(nn.Module):
    def __init__(self, alpha=0.01, beta=0.01, gamma=0.1, margin=0.01, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.margin = margin
        self.main_loss = JointLoss(
            SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
            DiceLoss(smooth=0.05, ignore_index=ignore_index),
            1.0,
            1.0,
        )
        self.aux_loss = SoftCrossEntropyLoss(
            smooth_factor=0.05, ignore_index=ignore_index
        )
        # self.intra_class_loss = IntraClassConsistencyLoss(alpha=0.5, beta=0.5)
        # self.offset_estimator = DifferentiableOffsetEstimator(
        #     max_displacement=30, patch_size=64, stride=16, temperature=0.5, mode="max"
        # )
        self.offset_estimator = ObjectAwareOffsetEstimator(
            max_displacement=5,
            patch_size=5,
            stride=1,
            temperature=1.0,
            max_d_x=20,
            min_d_x=-40,
            max_d_y=0,
            min_d_y=-60,
        )

    def forward(
        self,
        logits: Tensor,
        offset: Tensor,
        labels: Tensor,
        pred_mask: Tensor = None,
        features: Tensor = None,
    ):
        """计算offset损失

        Args:
            logits (Tensor): 模型输出(batch_size, 4, h, w) | Tuple[(batch_size, 4, h, w),(batch_size, 4, h, w)]
            offsets (Tensor): 原始图像(batch_size, 2, h, w)
            offset_labels (Tensor): 偏移的标签(batch_size, h, w)
        """
        B, C, H, W = offset.shape
        labels_32 = labels.to(torch.float32)
        # 先让labels偏移
        labels_offset = offset_tensor_v3(
            labels_32.unsqueeze(1), offset, sample_mode="nearest"
        ).squeeze(
            1
        )  # probability map
        labels_offset = (labels_offset > 0).to(torch.int64)

        if self.training and len(logits) == 2:
            logit_main, logit_aux = logits
            pred = ((logit_main + logit_aux) / 2).argmax(dim=1)

            # 计算交叉熵损失
            loss_main = self.main_loss(
                logit_main, labels_offset.to(torch.int64)
            ) + 0.4 * self.aux_loss(logit_aux, labels_offset.to(torch.int64))

            offset_est_x, offset_est_y, confidence = self.offset_estimator(
                pred.unsqueeze(1).to(dtype=torch.float32), labels_32.unsqueeze(1)
            )
            # 计算offset损失
            offset_est = torch.cat([offset_est_x / W, offset_est_y / H], dim=1)
            base_loss_offset = torch.mean(
                nn.SmoothL1Loss(reduction="none")(offset_est, offset)
            )

            # 1. 位移方向一致性损失（减少静态区域偏差）
            pred_direction = F.normalize(offset, p=2, dim=1)
            gt_direction = F.normalize(offset_est, p=2, dim=1)
            cos_sim = (pred_direction * gt_direction).sum(dim=1, keepdim=True)
            direction_loss = torch.mean((1 - cos_sim) * confidence)

            # 2. 位移量级损失（防止零位移陷阱）
            pred_magnitude = torch.norm(offset, p=2, dim=1, keepdim=True)
            gt_magnitude = torch.norm(offset_est, p=2, dim=1, keepdim=True)

            # 仅对需要较大位移的区域施加惩罚
            active_mask = (gt_magnitude > self.margin).float()
            magnitude_loss = F.l1_loss(pred_magnitude, gt_magnitude, reduction="none")
            magnitude_loss = torch.mean(magnitude_loss * active_mask * confidence)

            # 3. 位移梯度平滑损失（优化边缘区域）
            # dx_pred = torch.abs(offset[:, 0, :, 1:] - offset[:, 0, :, :-1])
            # dy_pred = torch.abs(offset[:, 1, 1:, :] - offset[:, 1, :-1, :])
            # smooth_loss = (dx_pred.mean() + dy_pred.mean()) * 0.5

            total_offset_loss = (
                self.alpha * base_loss_offset
                + self.beta * direction_loss
                + self.gamma * magnitude_loss
                # + smooth_loss
            )
        else:
            pred = (logits / 2).argmax(dim=1)

            loss_main = self.main_loss(logits, labels)

            # offset_est_x, offset_est_y, confidence = self.offset_estimator(
            #     pred.unsqueeze(1).to(dtype=torch.float32), labels_32.unsqueeze(1)
            # )
            # offset_est_x[offset_est_x != 0] = 255
            # offset_est_y[offset_est_y != 0] = 255
            # for b in range(B):
            #     visualize_masks(
            #         pred_mask[b],
            #         offset_est_x[b].squeeze(),
            #         os.path.join(
            #             "/root/workspace/zjj/xjd/GeoSeg-main", "DEBUG", "visual"
            #         ),
            #         f"offset_est_x_{b}",
            #     )
            #     visualize_masks(
            #         pred_mask[b],
            #         offset_est_y[b].squeeze(),
            #         os.path.join(
            #             "/root/workspace/zjj/xjd/GeoSeg-main", "DEBUG", "visual"
            #         ),
            #         f"offset_est_y_{b}",
            #     )
            total_offset_loss = base_loss_offset = direction_loss = magnitude_loss = (
                smooth_loss
            ) = 0

        return {
            "loss": loss_main + total_offset_loss,
            "loss_main": loss_main,
            "total_offset_loss": total_offset_loss,
            "base_loss_offset": base_loss_offset,
            "direction_loss": direction_loss,
            "magnitude_loss": magnitude_loss,
            # "smooth_loss": smooth_loss,
        }

    def intra_class_loss_with_blocks(
        self, logits: Tensor, masks: Tensor, block_size=16
    ):
        """分块计算类内一致性损失

        Args:
            logits (Tensor): 模型输出(batch_size, 4, h, w)
            masks (Tensor): 标签(batch_size, h, w)
            block_size (int): 分块大小
        """
        B, C, H, W = logits.shape
        masks = masks.unsqueeze(1)
        losses = []
        for i in range(0, H, block_size):
            for j in range(0, W, block_size):
                block_logits_x = logits[:, :1, i : i + block_size, j : j + block_size]
                block_logits_y = logits[:, 1:, i : i + block_size, j : j + block_size]
                block_labels = masks[:, :, i : i + block_size, j : j + block_size]
                losses.append(
                    self.intra_class_loss(block_logits_x, block_labels)
                    + self.intra_class_loss(block_logits_y, block_labels)
                )
        return sum(losses) / len(losses)


class Offset_v4(nn.Module):
    def __init__(self, alpha=0.01, beta=0.01, gamma=0.1, margin=0.01, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.margin = margin
        self.main_loss = JointLoss(
            SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
            DiceLoss(smooth=0.05, ignore_index=ignore_index),
            1.0,
            1.0,
        )
        self.aux_loss = SoftCrossEntropyLoss(
            smooth_factor=0.05, ignore_index=ignore_index
        )
        # self.intra_class_loss = IntraClassConsistencyLoss(alpha=0.5, beta=0.5)
        # self.offset_estimator = DifferentiableOffsetEstimator(
        #     max_displacement=30, patch_size=64, stride=16, temperature=0.5, mode="max"
        # )
        self.offset_estimator = ObjectAwareOffsetEstimator(
            max_displacement=5,
            patch_size=5,
            stride=1,
            temperature=1.0,
            max_d_x=20,
            min_d_x=-40,
            max_d_y=0,
            min_d_y=-60,
        )

    def forward(
        self,
        logits: Tensor,
        offset: Tensor,
        labels: Tensor,
        pred_mask: Tensor = None,
        features: Tensor = None,
    ):
        """计算offset损失

        Args:
            logits (Tensor): 模型输出(batch_size, 4, h, w) | Tuple[(batch_size, 4, h, w),(batch_size, 4, h, w)]
            offsets (Tensor): 原始图像(batch_size, 2, h, w)
            offset_labels (Tensor): 偏移的标签(batch_size, h, w)
        """

    def get_instances(label_tensor):
        """
        输入: label_tensor (torch.Tensor), 形状 [H, W], 值 0 或 1
        输出: inst_map (torch.Tensor), 形状 [H, W], 值 0 , 1...K
                masks (list of torch.Tensor), 每个实例的二进制掩码
        """
        label_np = label_tensor.cpu().numpy().astype(int)
        labeled, num_objects = ndimage.label(label_np)
        inst_map = torch.from_numpy(labeled).to(label_tensor.device)

        masks = []
        for k in range(1, num_objects + 1):
            mask = inst_map == k
            masks.append(mask)

        return inst_map, masks, num_objects


if __name__ == "__main__":
    # 示例输入
    B, C, H, W = 2, 3, 4, 4
    logits = torch.randn(B, C, H, W)
    offset = torch.randn(B, 2, H, W)
    print(f"logits: {logits},\n offset: {offset}")
    OFF = OffsetUsefulLoss()
    shifted_logits = OFF._offset_logits(logits, offset)
    print(f"shifted_logits: {shifted_logits}")
