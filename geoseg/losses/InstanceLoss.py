import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfidenceWeightedL1Loss(nn.Module):
    """
    计算 L1 损失 (MAE)，并按实例置信度加权。
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        predicted_offsets: torch.Tensor,  # [N_total, 2]
        gt_offsets: torch.Tensor,  # [N_total, 2]
        gt_confidences: torch.Tensor,  # [N_total, 1]
    ) -> torch.Tensor:

        # 1. 计算未加权的 L1 损失
        # reduction='none' 保留每个元素（dx, dy）的损失
        loss_unweighted = F.l1_loss(predicted_offsets, gt_offsets, reduction="none")

        # 2. 按实例求和 (dx_loss + dy_loss)
        loss_per_instance = loss_unweighted.sum(dim=1)  # [N_total]

        # 3. 按置信度加权
        confidences = gt_confidences.squeeze()  # [N_total]
        weighted_loss_per_instance = loss_per_instance * confidences

        # 4. 聚合
        if self.reduction == "mean":
            return weighted_loss_per_instance.sum() / (gt_confidences.sum() + 1e-8)
        elif self.reduction == "sum":
            return weighted_loss_per_instance.sum()
        else:
            return weighted_loss_per_instance
