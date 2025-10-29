import torch
import torch.nn as nn
import torch.nn.functional as F


class IntraClassConsistencyLoss(nn.Module):
    def __init__(self, alpha=0.1, beta=0.01):
        """
        alpha: 前景方差损失权重
        beta: 背景方差损失权重
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, logits, mask):
        """
        logits: 网络输出的置信度图 [B, C, H, W]
        mask: 二值分割标签 [B, 1, H, W] (0=背景, 1=前景)
        """
        # 分离前景和背景区域
        fg_mask = (mask > 0.5).float()
        bg_mask = 1 - fg_mask

        # 计算前景区域均值
        fg_pixels = logits * fg_mask
        fg_count = torch.sum(fg_mask, dim=[2, 3], keepdim=True)

        # 初始化前景损失为0
        fg_loss = torch.tensor(0.0, device=logits.device)

        # 仅在前景区域存在像素时计算前景方差
        if torch.any(fg_count > 0):
            fg_sum = torch.sum(fg_pixels, dim=[2, 3], keepdim=True)
            fg_mean = fg_sum / (fg_count + 1e-8)

            # 计算前景方差损失
            fg_var = torch.sum(fg_mask * (fg_pixels - fg_mean) ** 2, dim=[2, 3])
            fg_var = fg_var / (fg_count.squeeze(2, 3) + 1e-8)
            fg_loss = torch.mean(fg_var)

        # 计算背景区域均值
        bg_pixels = logits * bg_mask
        bg_count = torch.sum(bg_mask, dim=[2, 3], keepdim=True)

        # 初始化背景损失为0
        bg_loss = torch.tensor(0.0, device=logits.device)

        # 仅在背景区域存在像素时计算背景方差
        if torch.any(bg_count > 0):
            bg_sum = torch.sum(bg_pixels, dim=[2, 3], keepdim=True)
            bg_mean = bg_sum / (bg_count + 1e-8)

            # 计算背景方差损失
            bg_var = torch.sum(bg_mask * (bg_pixels - bg_mean) ** 2, dim=[2, 3])
            bg_var = bg_var / (bg_count.squeeze(2, 3) + 1e-8)
            bg_loss = torch.mean(bg_var)

        return self.alpha * fg_loss + self.beta * bg_loss
