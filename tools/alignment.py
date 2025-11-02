import itertools
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy import ndimage
from scipy import stats
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from typing import List, Tuple, Optional


class InstanceWiseAlignmentOptimizer:
    """
    基于实例级别的掩码对齐优化器
    使用原图点乘mask直接处理每个实例，保持梯度连续
    """

    def __init__(self):
        pass

    def get_instances(
        self, label_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor], int]:
        """提取二值标签图中的每个独立实例"""
        label_np = label_tensor.cpu().numpy().astype(np.uint8)
        labeled, num_objects = ndimage.label(label_np)
        inst_map = torch.from_numpy(labeled).to(label_tensor.device)

        masks = []
        for k in range(1, num_objects + 1):
            mask = inst_map == k
            masks.append(mask)

        return inst_map, masks, num_objects

    def create_base_grid(
        self, height: int, width: int, device: torch.device, align_corners: bool = False
    ) -> torch.Tensor:
        """创建基础网格"""
        i_tensor = torch.arange(height, device=device).float()
        j_tensor = torch.arange(width, device=device).float()
        grid_y, grid_x = torch.meshgrid(i_tensor, j_tensor, indexing="ij")

        if align_corners:
            grid_x_norm = 2 * grid_x / (width - 1) - 1
            grid_y_norm = 2 * grid_y / (height - 1) - 1
        else:
            grid_x_norm = 2 * (grid_x + 0.5) / width - 1
            grid_y_norm = 2 * (grid_y + 0.5) / height - 1

        grid = torch.stack((grid_x_norm, grid_y_norm), dim=-1)
        return grid

    def apply_affine_transform(
        self,
        mask: torch.Tensor,
        dx: torch.Tensor,
        dy: torch.Tensor,
        theta: float = 0.0,
        scale: float = 1.0,
    ) -> torch.Tensor:
        """
        应用仿射变换到图像和掩码

        参数:
            dx, dy: 平移参数（归一化到[-1,1]）
            theta: 旋转角度（弧度）
            scale: 缩放因子

        返回:
            transformed_mask: 变换后的掩码区域
        """
        H, W = mask.shape[-2:]

        # 创建基础网格
        base_grid = self.create_base_grid(H, W, mask.device, align_corners=False)

        shifted_x = base_grid[..., 0] - dx * 2  # 转换为[-2,2]范围
        shifted_y = base_grid[..., 1] - dy * 2
        grid = torch.stack((shifted_x, shifted_y), dim=-1)
        # grid = torch.clamp(grid, -1, 1).unsqueeze(0)
        grid = grid.unsqueeze(0)  # [1, H, W, 2]

        # 应用变换到掩码
        transformed_mask = F.grid_sample(
            mask.unsqueeze(0).unsqueeze(0).float(),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).squeeze()

        return transformed_mask

    def compute_region_monotonicity(
        self, image_region: torch.Tensor, mask_region: torch.Tensor
    ) -> torch.Tensor:
        """
        计算掩码覆盖区域的单调性 (低方差)
        (这是你原始代码中的函数，它很好)
        """
        # (关键) 我们希望在软掩码上计算，所以 S 是软和
        S = mask_region.sum()
        if S < 1e-5:
            return torch.tensor(0.0, device=mask_region.device)

        C = image_region.shape[0] if image_region.dim() == 3 else 1
        scores = []

        for c in range(C):
            channel = image_region[c] if C > 1 else image_region

            # (关键) 使用软掩码进行加权平均
            region_pixels = channel * mask_region

            # E[X^2]
            mean_sq = (region_pixels**2).sum() / S
            # (E[X])^2
            sq_mean = (region_pixels.sum() / S) ** 2

            variance = (mean_sq - sq_mean).abs()

            # 使用指数函数将方差转换为单调性分数
            # (调低 0.1，使方差更敏感)
            score = torch.exp(-variance / 0.05)
            scores.append(score)

        monotonicity_score = torch.mean(torch.stack(scores))
        return torch.clamp(monotonicity_score, 0.0, 1.0)

    def compute_gradient_magnitude(self, image: torch.Tensor) -> torch.Tensor:
        """计算图像每个通道的梯度幅度并取平均"""
        # 获取通道数
        C = image.shape[0] if image.dim() == 3 else 1
        gradient_magnitudes = []

        # Sobel算子
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32,
            device=image.device,
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32,
            device=image.device,
        ).view(1, 1, 3, 3)

        for c in range(C):
            channel = image[c] if C > 1 else image
            channel = channel.unsqueeze(0).unsqueeze(0)

            # 计算梯度
            grad_x = F.conv2d(channel, sobel_x, padding=1)
            grad_y = F.conv2d(channel, sobel_y, padding=1)

            # 计算梯度幅度
            grad_mag = torch.sqrt(grad_x**2 + grad_y**2).squeeze()
            gradient_magnitudes.append(grad_mag)

        # 取所有通道的平均梯度幅度
        avg_gradient = torch.mean(torch.stack(gradient_magnitudes), dim=0)
        return avg_gradient / (avg_gradient.max() + 1e-8)

    def gaussian_blur(
        self, image: torch.Tensor, kernel_size: int = 5, sigma: float = 1.0
    ) -> torch.Tensor:
        """
        高斯模糊 (可处理多通道图像)
        输入: (C, H, W) 或 (B, C, H, W)
        """
        has_batch_dim = image.dim() == 4
        if not has_batch_dim:
            image_batch = image.unsqueeze(0)  # 增加批量维度 (B, C, H, W)
        else:
            image_batch = image

        B, C, H, W = image_batch.shape

        # 创建高斯核
        x = torch.arange(kernel_size, device=image.device) - kernel_size // 2
        y = torch.arange(kernel_size, device=image.device) - kernel_size // 2
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()

        # 调整核形状以进行深度可分离卷积
        kernel_view = kernel.view(1, 1, kernel_size, kernel_size).repeat(C, 1, 1, 1)

        # 应用卷积 (groups=C 意味着每个通道独立使用一个核)
        blurred = F.conv2d(image_batch, kernel_view, padding=kernel_size // 2, groups=C)

        if not has_batch_dim:
            return blurred.squeeze(0)  # 移除批量维度
        return blurred

    def compute_dt_map(self, rgb_image: torch.Tensor) -> torch.Tensor:
        """
        计算(平滑后)图像的距离变换图
        """
        # 1. 转换为灰度图
        if rgb_image.dim() == 3:
            gray_image = (
                0.299 * rgb_image[0] + 0.587 * rgb_image[1] + 0.114 * rgb_image[2]
            )
        else:
            gray_image = rgb_image

        gray_np = gray_image.cpu().numpy()

        # 2. 计算Canny边缘
        gray_np_u8 = (gray_np * 255).astype(np.uint8)

        # (关键) 对模糊图像使用更宽松的Canny阈值
        img_edge = cv2.Canny(gray_np_u8, 5, 15) > 0
        img_edge = img_edge.astype(np.float32)

        if img_edge.sum() == 0:
            # print("警告：未在图像中检测到Canny边缘")
            return torch.zeros_like(gray_image)

        # 3. (关键) 只计算外部距离 (到最近边缘的距离)
        # 我们希望mask边缘“滚”到值为0的地方
        dt_map = ndimage.distance_transform_edt(1 - img_edge)

        dt_tensor = torch.from_numpy(dt_map).to(rgb_image.device)

        # 归一化DT Map，使损失值更稳定
        return dt_tensor / (dt_tensor.max() + 1e-8)

    def compute_mask_edges(self, mask: torch.Tensor) -> torch.Tensor:
        """计算二值掩码的边缘"""
        # 使用卷积计算边缘
        kernel = torch.tensor(
            [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
            dtype=torch.float32,
            device=mask.device,
        ).view(1, 1, 3, 3)

        edges = F.conv2d(mask.float(), kernel, padding=1)

        return edges

    def optimize_instance_offset(
        self,
        original_rgb_image: torch.Tensor,
        masks: List[torch.Tensor],
        dt_map: torch.Tensor,
        initial_dxs: List[float],
        initial_dys: List[float],
        max_iterations: int = 100,
        lr: float = 1e-3,
        reg_weight: float = 0.2,  # L2正则化权重
        mono_weight: float = 0.5,
    ) -> Tuple[List[float], List[float], List[dict]]:
        """
        批量优化所有实例的偏移变量 (使用软边缘DT损失)
        """
        num_objects = len(masks)

        # 初始化所有实例的可学习参数
        dx_params = nn.ParameterList(
            [
                nn.Parameter(
                    torch.tensor(initial_dxs[i], device=original_rgb_image.device)
                )
                for i in range(num_objects)
            ]
        )
        dy_params = nn.ParameterList(
            [
                nn.Parameter(
                    torch.tensor(initial_dys[i], device=original_rgb_image.device)
                )
                for i in range(num_objects)
            ]
        )

        # 使用Adam优化器统一优化所有参数
        optimizer = optim.Adam(list(dx_params) + list(dy_params), lr=lr)

        # 初始化历史记录（每个实例单独记录）
        histories = [
            {"dx": [], "dy": [], "loss": [], "dt_loss": [], "mono_loss": []}
            for _ in range(num_objects)
        ]

        best_losses = [float("inf")] * num_objects
        best_dxs = initial_dxs.copy()
        best_dys = initial_dys.copy()

        for iteration in range(max_iterations):
            optimizer.zero_grad()

            total_loss = torch.tensor(0.0, device=original_rgb_image.device)
            per_instance_losses = []

            for i in range(num_objects):
                mask = masks[i]
                dx = dx_params[i]
                dy = dy_params[i]
                initial_dx = initial_dxs[i]
                initial_dy = initial_dys[i]

                # 1. 应用仿射变换到掩码 (得到软掩码)
                transformed_mask = self.apply_affine_transform(mask, dx, dy)

                mask_sum = transformed_mask.sum()
                if mask_sum < 1e-5:
                    # 给予一个巨大的损失让它“弹回”
                    dx_change = dx - initial_dx
                    dy_change = dy - initial_dy
                    inst_loss = 100 * torch.sqrt(dx_change**2 + dy_change**2 + 1e-8)
                    dt_loss = torch.tensor(0.0, device=original_rgb_image.device)
                    mono_loss = torch.tensor(1.0, device=original_rgb_image.device)
                else:
                    # 2. 计算软掩码的“软边缘”
                    soft_edges_raw = self.compute_mask_edges(
                        transformed_mask.unsqueeze(0).unsqueeze(0)
                    ).squeeze()
                    soft_edges = torch.relu(soft_edges_raw)

                    # 3. 计算损失
                    edge_sum = soft_edges.sum() + 1e-8
                    dt_loss = (dt_map * soft_edges).sum() / edge_sum

                    monotonicity_score = self.compute_region_monotonicity(
                        original_rgb_image, transformed_mask
                    )
                    mono_loss = 1.0 - monotonicity_score

                    # L2 正则化
                    dx_change = dx - initial_dx
                    dy_change = dy - initial_dy
                    reg_loss = reg_weight * torch.sqrt(
                        dx_change**2 + dy_change**2 + 1e-8
                    )

                    inst_loss = dt_loss + (mono_weight * mono_loss) + reg_loss

                # 累加总损失
                total_loss += inst_loss
                per_instance_losses.append(
                    {
                        "inst_loss": inst_loss.item(),
                        "dt_loss": dt_loss.item(),
                        "mono_loss": mono_loss.item(),
                    }
                )

            # 反向传播总损失
            total_loss.backward()

            # 梯度裁剪防止爆炸
            torch.nn.utils.clip_grad_norm_(
                list(dx_params) + list(dy_params), max_norm=1.0
            )

            optimizer.step()

            # 记录历史并更新最佳值
            for i in range(num_objects):
                dx_val = dx_params[i].item()
                dy_val = dy_params[i].item()
                histories[i]["dx"].append(dx_val)
                histories[i]["dy"].append(dy_val)
                histories[i]["loss"].append(per_instance_losses[i]["inst_loss"])
                histories[i]["dt_loss"].append(per_instance_losses[i]["dt_loss"])
                histories[i]["mono_loss"].append(per_instance_losses[i]["mono_loss"])

                # 更新最佳得分
                if per_instance_losses[i]["inst_loss"] < best_losses[i]:
                    best_losses[i] = per_instance_losses[i]["inst_loss"]
                    best_dxs[i] = dx_val
                    best_dys[i] = dy_val

            # 早期停止检查（基于总损失）
            if (
                iteration > 10
                and abs(
                    sum([h["loss"][-1] for h in histories])
                    - sum([h["loss"][-5] for h in histories])
                )
                < 1e-5
            ):
                break

        return best_dxs, best_dys, histories

    def align_instance_with_multi_start(
        self,
        rgb_image: torch.Tensor,
        label_image: torch.Tensor,
        gaussian_mu: List[float] = [0.0, 0.0],  # 高斯分布的均值 [mu_x, mu_y]
        gaussian_sigma: List[float] = [1.0, 1.0],  # 高斯分布的标准差 [sigma_x, sigma_y]
        covariance: List[List[float]] = [
            [1.0, 0.0],
            [0.0, 1.0],
        ],  # 协方差矩阵 [[var_x, cov_xy], [cov_yx, var_y]]
        base_grid: Optional[torch.Tensor] = None,
        search_range_norm: float = 0.01,  # 归一化搜索范围
        num_candidates: int = 8,  # 候选初始化点的数量
        nms_radius_norm: float = 0.001,  # 非极大值抑制的距离阈值
        pyramid_args: Optional[List[Dict]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        批量处理所有实例，返回对齐后的标签和位移场
        返回：
            aligned_label (对齐后的标签)
            disp_field (位移场 [H, W, 2])
            confidence_map: 置信度图 [H, W]，值在0-1之间
        """
        _, masks, num_objects = self.get_instances(label_image)
        H, W = label_image.shape
        device = label_image.device
        if num_objects == 0:
            disp_field = torch.zeros((H, W, 2), device=label_image.device)
            return label_image, disp_field

        # 1. 定义金字塔参数
        if pyramid_args is None:
            pyramid_levels = [
                {
                    "name": "Coarse",
                    "sigma": 20.0,
                    "ksize": 41,
                    "iters": 50,
                    "lr": 2e-3,
                    "reg": 0.3,
                    "mono": 0.1,
                },
                {
                    "name": "Medium",
                    "sigma": 10.0,
                    "ksize": 21,
                    "iters": 50,
                    "lr": 5e-4,
                    "reg": 0.7,
                    "mono": 0.3,
                },
                {
                    "name": "Fine",
                    "sigma": 4.0,
                    "ksize": 9,
                    "iters": 50,
                    "lr": 1e-4,
                    "reg": 0.7,
                    "mono": 1.0,
                },
            ]
        else:
            pyramid_levels = pyramid_args

        # 预计算DT Maps
        dt_maps = {}
        for level in pyramid_levels:
            img_blurred = self.gaussian_blur(rgb_image, level["ksize"], level["sigma"])
            dt_maps[level["name"]] = self.compute_dt_map(img_blurred)

        # 2. 生成候选初始化偏移（每个实例使用相同的候选点集）
        directions = list(itertools.product([1.0], repeat=2))  # [-1.0, 0.0, 1.0]
        candidates = [
            (dx * search_range_norm, dy * search_range_norm) for dx, dy in directions
        ]

        # print(f"开始批量优化 {num_objects} 个实例...")

        # 检查所有掩码的有效性
        valid_masks = []
        valid_indices = []
        for i, mask in enumerate(masks):
            if mask.sum() > 0:
                valid_masks.append(mask)
                valid_indices.append(i)

        if not valid_masks:
            # print("警告: 所有实例掩码均为空，返回原始标签图")
            disp_field = torch.zeros((H, W, 2), device=device)
            return label_image, disp_field

        # 3. 对每个候选点进行批量金字塔优化
        # print(f"开始多点初始化优化 ({len(candidates)} 个候选点)...")
        candidate_results = []

        for initial_dx, initial_dy in candidates:
            # 所有有效实例使用相同的初始偏移
            current_dxs = [initial_dx] * len(valid_masks)
            current_dys = [initial_dy] * len(valid_masks)
            level_histories_list = []  # 每个实例的层级历史

            for level in pyramid_levels:
                # 批量优化所有有效实例
                dx_list, dy_list, histories = self.optimize_instance_offset(
                    rgb_image,
                    valid_masks,
                    dt_maps[level["name"]],
                    initial_dxs=current_dxs,
                    initial_dys=current_dys,
                    max_iterations=level["iters"],
                    lr=level["lr"],
                    reg_weight=level["reg"],
                    mono_weight=level["mono"],
                )
                level_histories_list.append(histories)
                current_dxs, current_dys = dx_list, dy_list

            # 记录每个有效实例的最终结果
            for i in range(len(valid_masks)):
                final_loss = (
                    level_histories_list[-1][i]["loss"][-1]
                    if level_histories_list
                    else float("inf")
                )
                candidate_results.append(
                    {
                        "inst_idx": valid_indices[i],
                        "initial_dx": initial_dx,
                        "initial_dy": initial_dy,
                        "final_dx": current_dxs[i],
                        "final_dy": current_dys[i],
                        "final_loss": final_loss,
                    }
                )

            # print(
            #     f"  初始({initial_dx:.4f}, {initial_dy:.4f}) -> 平均最终损失={np.mean([r['final_loss'] for r in candidate_results if r['initial_dx']==initial_dx]):.6f}"
            # )

        # 4. 对每个实例单独进行NMS
        optimized_offsets = [(0.0, 0.0)] * num_objects
        instance_confidences = [0.0] * num_objects

        for inst_idx in range(num_objects):
            # 收集当前实例的所有候选结果
            inst_results = [r for r in candidate_results if r["inst_idx"] == inst_idx]
            if not inst_results:
                continue  # 跳过空掩码实例

            # 按损失排序
            inst_results.sort(key=lambda x: x["final_loss"])

            # NMS
            selected = []
            nms_radius_sq = nms_radius_norm**2

            for res in inst_results:
                is_suppressed = False
                for sel_res in selected:
                    dist_sq = (res["final_dx"] - sel_res["final_dx"]) ** 2 + (
                        res["final_dy"] - sel_res["final_dy"]
                    ) ** 2
                    if dist_sq < nms_radius_sq:
                        is_suppressed = True
                        break
                if not is_suppressed:
                    selected.append(res)

            # 选择最优解
            if selected:
                best_res = selected[0]
                optimized_offsets[inst_idx] = (
                    best_res["final_dx"],
                    best_res["final_dy"],
                )
                # print(
                #     f"实例 {inst_idx+1}: 最优偏移为 dx={best_res['final_dx']:.4f}, dy={best_res['final_dy']:.4f}, 损失={best_res['final_loss']:.6f}"
                # )
                dx, dy = best_res["final_dx"], best_res["final_dy"]
                confidence = self.calculate_offset_confidence_advanced(
                    dx * W, dy * H, gaussian_mu, gaussian_sigma, covariance
                )
                instance_confidences[inst_idx] = confidence

        if base_grid is None:
            base_grid = self.create_base_grid(H, W, device, align_corners=False)

        aligned_label = torch.zeros((H, W), device=device)
        disp_field = torch.zeros((H, W, 2), device=device)
        confidence_map = torch.zeros((H, W), device=device)
        for inst_idx, (dx, dy) in enumerate(optimized_offsets):
            mask = masks[inst_idx]
            if mask.sum() == 0:
                continue

            transformed_mask = self.apply_affine_transform(mask, dx, dy)
            aligned_label[transformed_mask > 0.5] = 1

            confidence_value = instance_confidences[inst_idx]
            confidence_map[transformed_mask > 0.5] = confidence_value

            union_mask = mask | (transformed_mask > 0.5)

            disp_field[..., 0][union_mask] = dx
            disp_field[..., 1][union_mask] = dy

        return aligned_label, disp_field, confidence_map

    def calculate_offset_confidence(
        self,
        dx: float,
        dy: float,
        gaussian_mu: List[float],
        gaussian_sigma: List[float],
    ) -> float:
        """
        计算偏移量在给定高斯分布下的概率置信度

        参数:
            dx, dy: 偏移量
            gaussian_mu: 高斯分布均值 [mu_x, mu_y]
            gaussian_sigma: 高斯分布标准差 [sigma_x, sigma_y]

        返回:
            confidence: 置信度值 (0-1)
        """
        mu_x, mu_y = gaussian_mu
        sigma_x, sigma_y = gaussian_sigma

        # 转换为numpy数组进行计算
        dx_np = float(dx)
        dy_np = float(dy)
        mu_x_np = float(mu_x)
        mu_y_np = float(mu_y)
        sigma_x_np = float(sigma_x)
        sigma_y_np = float(sigma_y)

        # 计算二维高斯分布的概率密度
        # 假设x和y方向独立
        prob_x = stats.norm.pdf(dx_np, loc=mu_x_np, scale=sigma_x_np)
        prob_y = stats.norm.pdf(dy_np, loc=mu_y_np, scale=sigma_y_np)

        # 联合概率密度（由于独立，所以是乘积）
        joint_prob = prob_x * prob_y

        # 计算在原点（0,0）的概率密度作为参考（最大值）
        max_prob_x = stats.norm.pdf(mu_x_np, loc=mu_x_np, scale=sigma_x_np)
        max_prob_y = stats.norm.pdf(mu_y_np, loc=mu_y_np, scale=sigma_y_np)
        max_joint_prob = max_prob_x * max_prob_y

        # 归一化到0-1范围
        if max_joint_prob > 0:
            normalized_confidence = joint_prob / max_joint_prob
        else:
            normalized_confidence = 0.0

        # 确保在0-1范围内
        confidence = max(0.0, min(1.0, normalized_confidence))

        return confidence

    def calculate_offset_confidence_advanced(
        self,
        dx: float,
        dy: float,
        gaussian_mu: List[float],
        gaussian_sigma: List[float],
        covariance: Optional[List[List[float]]] = None,
    ) -> float:
        """
        高级版：考虑相关性的二维高斯分布置信度计算

        参数:
            dx, dy: 偏移量
            gaussian_mu: 高斯分布均值 [mu_x, mu_y]
            gaussian_sigma: 高斯分布标准差 [sigma_x, sigma_y]
            covariance: 协方差矩阵 [[cov_xx, cov_xy], [cov_yx, cov_yy]]，如果为None则假设独立

        返回:
            confidence: 置信度值 (0-1)
        """
        import numpy as np
        from scipy.stats import multivariate_normal

        # 转换为numpy数组
        point = np.array([dx, dy])
        mu = np.array(gaussian_mu)

        if covariance is None:
            # 如果没有提供协方差矩阵，假设x和y独立
            cov = np.diag([sigma**2 for sigma in gaussian_sigma])
        else:
            cov = np.array(covariance)

        try:
            # 创建多元高斯分布
            rv = multivariate_normal(mean=mu, cov=cov)

            # 计算在给定点的概率密度
            prob = rv.pdf(point)

            # 计算在均值点的概率密度（最大值）
            max_prob = rv.pdf(mu)

            # 归一化到0-1范围
            if max_prob > 0:
                confidence = prob / max_prob
            else:
                confidence = 0.0

        except:
            # 如果计算失败，回退到简单版本
            confidence = self.calculate_offset_confidence(
                dx, dy, gaussian_mu, gaussian_sigma
            )

        return max(0.0, min(1.0, confidence))

    def visualize_alignment_with_confidence(
        self,
        rgb_image: torch.Tensor,
        aligned_label: torch.Tensor,
        confidence_map: torch.Tensor,
        original_label: Optional[torch.Tensor] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """
        可视化对齐结果和置信度图

        参数:
            rgb_image: 原始RGB图像
            aligned_label: 对齐后的标签
            confidence_map: 置信度图
            original_label: 原始标签（可选，用于对比）
            save_path: 保存路径（可选）
        """
        import matplotlib.pyplot as plt

        # 转换为numpy数组用于可视化
        rgb_np = (
            rgb_image.cpu().numpy().transpose(1, 2, 0)
            if rgb_image.dim() == 3
            else rgb_image.cpu().numpy()
        )
        rgb_np = (rgb_np * 255).astype(np.uint8)
        aligned_label_np = aligned_label.cpu().numpy()
        confidence_np = confidence_map.cpu().numpy()

        # 创建可视化图像
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 1. 原始图像
        if rgb_np.shape[-1] == 3:
            axes[0, 0].imshow(rgb_np)
        else:
            axes[0, 0].imshow(rgb_np, cmap="gray")
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis("off")

        # 2. 原始标签（如果有）
        if original_label is not None:
            original_label_np = original_label.cpu().numpy()
            axes[0, 1].imshow(original_label_np, cmap="jet")
            axes[0, 1].set_title("Original Label")
            axes[0, 1].axis("off")
        else:
            axes[0, 1].axis("off")

        # 3. 对齐后的标签
        axes[0, 2].imshow(aligned_label_np, cmap="jet")
        axes[0, 2].set_title("Aligned Label")
        axes[0, 2].axis("off")

        # 4. 置信度图
        im = axes[1, 0].imshow(
            confidence_np,
            cmap="viridis",
            vmin=0,
            vmax=1.0,
        )
        axes[1, 0].set_title("Confidence Map")
        axes[1, 0].axis("off")
        plt.colorbar(im, ax=axes[1, 0])

        # 5. 图像+对齐标签叠加
        overlay = (
            rgb_np.copy() if rgb_np.shape[-1] == 3 else np.stack([rgb_np] * 3, axis=-1)
        )
        overlay[aligned_label_np > 0] = (
            overlay[aligned_label_np > 0] * 0.7 + np.array([255, 0, 0]) * 0.3
        )
        axes[1, 1].imshow(overlay.astype(np.uint8))
        axes[1, 1].set_title("Image + Aligned Label")
        axes[1, 1].axis("off")

        # 6. 图像+置信度热图叠加
        confidence_rgb = plt.cm.viridis(confidence_np)[..., :3]  # 将置信度图转换为RGB
        blend = 0.7 * rgb_np + 0.3 * confidence_rgb * 255
        axes[1, 2].imshow(blend.astype(np.uint8))
        axes[1, 2].set_title("Image + Confidence Heatmap")
        axes[1, 2].axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"可视化结果已保存到: {save_path}")

        # plt.show()
