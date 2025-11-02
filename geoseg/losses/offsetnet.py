from typing import Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfidenceWeightedCrossEntropyLoss(nn.Module):
    """
    置信度加权交叉熵损失（二分类场景：前景/背景）
    特性：
    1. 置信度图的值 ∈ [0,1]，值越高，损失惩罚越大
    2. 置信度=0的区域（背景）赋予最小权重，避免梯度消失
    3. 支持1通道（sigmoid）或2通道（softmax）logits输入
    4. 可选权重平滑，提升训练稳定性
    """

    def __init__(
        self,
        distance_threshold: int = 30,
        weight_smooth: bool = True,  # 是否启用权重平滑
        smooth_epsilon: float = 0.05,  # 平滑系数（使权重远离0和1）
        reduction: str = "mean",  # 损失聚合方式：mean/sum/none
        min_confidence: float = 0.01,
        distance_type: str = "l2",  # 距离类型：l1（曼哈顿）/l2（欧氏）
    ):
        super().__init__()
        self.distance_threshold = distance_threshold
        self.weight_smooth = weight_smooth
        self.smooth_epsilon = smooth_epsilon
        self.reduction = reduction
        self.min_confidence = min_confidence

        assert distance_type in ["l1", "l2"], "距离类型仅支持 l1/l2"
        self.dist_type = cv2.DIST_L2 if distance_type == "l2" else cv2.DIST_L1

        # 验证参数合法性
        assert distance_threshold > 0, "距离阈值必须为正整数"
        assert 0 < smooth_epsilon < 0.5, "smooth_epsilon必须在(0,0.5)之间"
        assert reduction in ["mean", "sum", "none"], "reduction只能是mean/sum/none"
        assert 0 < min_confidence < 1, "min_confidence必须在(0,1)之间"

    def _compute_min_distance(
        self, confidence_map: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算每个零置信像素到最近非零置信像素的「最小距离」和「最近非零置信度」
        输入：confidence_map (B, H, W) - 单样本置信度图
        输出：
            min_distances (B, H, W) - 每个像素到最近非零置信像素的最小距离（非零像素距离=0）
            nearest_confidences (B, H, W) - 每个像素对应的最近非零置信度（非零像素=自身置信度）
        """
        B, H, W = confidence_map.shape
        device = confidence_map.device
        min_distances = torch.zeros((B, H, W), device=device)
        nearest_confidences = torch.zeros((B, H, W), device=device)

        for b in range(B):
            # 单样本置信度图（CPU 计算：cv2 更高效，且避免 GPU 显存峰值）
            conf = confidence_map[b].cpu().numpy()  # (H, W)
            # 二值掩码：非零置信像素=255（前景），零置信=0（背景）
            fg_mask = (conf > self.min_confidence).astype(np.uint8) * 255

            if np.sum(fg_mask) == 0:
                # 全零置信图：距离设为阈值+1，置信度=0
                min_distances[b] = self.distance_threshold + 1
                nearest_confidences[b] = 0.0
                continue

            # ---------------------- 1. 距离变换计算最近距离（显存友好）----------------------
            # cv2.distanceTransform：仅计算背景像素到最近前景像素的距离，输出 (H,W)
            dist_map = cv2.distanceTransform(fg_mask, self.dist_type, 5)  # 5=掩码尺寸
            # 转换为 torch 张量并移回 GPU
            dist_tensor = torch.from_numpy(dist_map).float().to(device)
            min_distances[b] = dist_tensor

            # ---------------------- 2. 分层匹配获取最近非零置信度 ----------------------
            # 核心思路：对每个零置信像素，在「距离范围内」找最近的非零置信像素
            # 避免全量索引，而是按距离分层匹配，减少计算量
            zero_mask = conf <= self.min_confidence  # 零置信像素掩码
            if not np.any(zero_mask):
                # 全是非零置信像素
                nearest_confidences[b] = torch.from_numpy(conf).to(device)
                continue

            # 非零置信像素的坐标和置信度（仅存储一次，减少内存）
            fg_y, fg_x = np.nonzero(fg_mask)
            fg_conf_vals = conf[fg_y, fg_x]  # (N,)

            # 零置信像素的坐标
            zero_y, zero_x = np.nonzero(zero_mask)
            zero_dist = dist_map[zero_y, zero_x]  # 零置信像素的最近距离

            # 对每个零置信像素，找到最近的非零置信像素
            nearest_conf = np.zeros_like(zero_dist)
            for i in range(len(zero_y)):
                y0, x0 = zero_y[i], zero_x[i]
                d0 = zero_dist[i]

                # 仅在「d0+1」范围内搜索（距离变换保证最近像素在该范围内）
                # 减少搜索范围，降低计算量
                search_radius = int(np.ceil(d0)) + 1
                # 搜索区域边界裁剪（避免越界）
                y_min = max(0, y0 - search_radius)
                y_max = min(H - 1, y0 + search_radius)
                x_min = max(0, x0 - search_radius)
                x_max = min(W - 1, x0 + search_radius)

                # 找到搜索区域内的非零置信像素索引
                in_y_range = (fg_y >= y_min) & (fg_y <= y_max)
                in_x_range = (fg_x >= x_min) & (fg_x <= x_max)
                in_search = in_y_range & in_x_range

                if np.any(in_search):
                    # 计算搜索区域内非零像素到当前零像素的距离
                    search_y = fg_y[in_search]
                    search_x = fg_x[in_search]
                    search_conf = fg_conf_vals[in_search]
                    # 欧氏距离（与距离变换一致）
                    distances = np.sqrt((search_y - y0) ** 2 + (search_x - x0) ** 2)
                    # 找到最近的非零像素
                    min_idx = np.argmin(distances)
                    nearest_conf[i] = search_conf[min_idx]
                else:
                    # 极端情况：搜索范围内无非零像素（用全局最近）
                    distances = np.sqrt((fg_y - y0) ** 2 + (fg_x - x0) ** 2)
                    min_idx = np.argmin(distances)
                    nearest_conf[i] = fg_conf_vals[min_idx]

            # 将最近置信度赋值回张量
            nearest_conf_tensor = torch.from_numpy(nearest_conf).to(device)
            nearest_confidences[b][
                torch.from_numpy(zero_y).to(device), torch.from_numpy(zero_x).to(device)
            ] = nearest_conf_tensor
            # 非零置信像素的最近置信度=自身
            nearest_confidences[b][
                torch.from_numpy(fg_y).to(device), torch.from_numpy(fg_x).to(device)
            ] = torch.from_numpy(fg_conf_vals).to(device)

        return min_distances, nearest_confidences

    def preprocess_confidence(self, confidence_map: torch.Tensor) -> torch.Tensor:
        """
        置信度图预处理：平滑 → 距离感知加权 → 生成最终权重图
        输入：confidence_map (B, H, W) 或 (B, 1, H, W)
        输出：weights (B, H, W) - 距离感知的最终权重图
        """
        # 1. 统一形状为 (B, H, W)
        if confidence_map.dim() == 4:
            confidence_map = confidence_map.squeeze(1)
        assert (
            confidence_map.dim() == 3
        ), f"置信度图维度错误，应为3维(B,H,W)，实际为{confidence_map.dim()}维"
        B, H, W = confidence_map.shape

        # 2. 基础预处理：钳位 + 平滑
        conf_clamped = torch.clamp(confidence_map, 0.0, 1.0)  # 确保[0,1]范围
        if self.weight_smooth:
            # 平滑：避免置信度过于极端
            conf_smoothed = (
                1 - 2 * self.smooth_epsilon
            ) * conf_clamped + self.smooth_epsilon
        else:
            conf_smoothed = conf_clamped

        # 3. 确保非零置信像素的最小置信度
        conf_smoothed = torch.max(
            conf_smoothed,
            torch.tensor(self.min_confidence, device=conf_smoothed.device),
        )

        # 4. 计算距离图和最近置信度图
        min_distances, nearest_confidences = self._compute_min_distance(conf_smoothed)

        # 5. 生成距离感知权重
        weights = torch.ones_like(conf_smoothed)  # 初始化权重为1.0（默认远场背景）
        device = conf_smoothed.device

        # 零置信像素：距离 ≤ 阈值 → 动态权重；距离 > 阈值 → 保持1.0
        non_zero_mask = conf_smoothed > self.min_confidence
        zero_mask = ~non_zero_mask
        near_zero_mask = zero_mask & (min_distances <= self.distance_threshold)
        far_zero_mask = zero_mask & (min_distances > self.distance_threshold)

        # 非零置信像素：权重 = 平滑后的置信度
        weights[non_zero_mask] = conf_smoothed[non_zero_mask]
        # 远场零置信度像素
        weights[far_zero_mask] = 1.0

        # 近场零置信像素：权重 = 最近非零置信度 × (1 - 距离/阈值)
        if near_zero_mask.any():
            # 距离归一化到[0,1]
            normalized_dist = min_distances[near_zero_mask] / self.distance_threshold
            # 线性递减公式：weight = 最近置信度 + (1.0 - 最近置信度) × (1 - 归一化距离)
            # 解释：
            # - 距离=0 → normalized_dist=0 → weight=最近置信度
            # - 距离=阈值 → normalized_dist=1 → weight=1.0
            dynamic_weights = nearest_confidences[near_zero_mask] + (
                1.0 - nearest_confidences[near_zero_mask]
            ) * (1 - normalized_dist)
            weights[near_zero_mask] = dynamic_weights

        # 确保权重在[min_confidence, 1.0]范围内（避免异常值）
        weights = torch.clamp(weights, self.min_confidence, 1.0)

        return weights

    def forward(
        self,
        logits: torch.Tensor,
        confidence_map: torch.Tensor,
        labels: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        前向传播计算损失
        参数：
            logits: 模型输出 (B, 1, H, W) 或 (B, 2, H, W)
            confidence_map: 置信度图 (B, H, W) 或 (B, 1, H, W)，值∈[0,1]
            labels: 真实标签 (B, H, W) 或 (B, 1, H, W)，值为0/1（可选，默认用置信度>0作为标签）
        返回：
            weighted_loss: 距离感知的置信度加权交叉熵损失
        """
        B, H, W = logits.shape[0], logits.shape[2], logits.shape[3]

        # 1. 生成距离感知权重图
        weights = self.preprocess_confidence(confidence_map)
        assert weights.shape == (
            B,
            H,
            W,
        ), f"权重图形状错误，应为(B,H,W)，实际为{weights.shape}"

        # 2. 处理标签（默认用置信度>0作为前景标签）
        if labels is None:
            labels = (confidence_map > 0).float().squeeze()
        # 标签形状统一为 (B, H, W)
        if labels.dim() == 4:
            labels = labels.squeeze(1)
        assert labels.shape == (
            B,
            H,
            W,
        ), f"标签形状错误，应为(B,H,W)，实际为{labels.shape}"
        assert (labels == 0).sum() + (
            labels == 1
        ).sum() == B * H * W, "标签只能包含0（背景）和1（前景）"

        # 3. 计算基础交叉熵损失
        if logits.shape[1] == 1:
            # 1通道logits → 二元交叉熵
            logits_flat = logits.squeeze(1)  # (B, H, W)
            base_loss = F.binary_cross_entropy_with_logits(
                input=logits_flat, target=labels.float(), reduction="none"
            )
        elif logits.shape[1] == 2:
            # 2通道logits → 类别交叉熵
            labels_long = labels.long()
            logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, 2)  # (BHW, 2)
            labels_flat = labels_long.reshape(-1)  # (BHW,)
            base_loss = F.cross_entropy(
                input=logits_flat, target=labels_flat, reduction="none"
            ).reshape(B, H, W)
        else:
            raise ValueError(f"logits通道数错误，应为1或2，实际为{logits.shape[1]}")

        # 4. 距离感知权重加权损失
        weighted_loss = base_loss * weights

        # 5. 损失聚合
        if self.reduction == "mean":
            return weighted_loss.mean()
        elif self.reduction == "sum":
            return weighted_loss.sum()
        elif self.reduction == "none":
            return weighted_loss
        else:
            raise ValueError(f"不支持的reduction方式：{self.reduction}")
