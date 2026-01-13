import itertools
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy import ndimage
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from typing import List, Tuple, Optional


def img_with_label(image, label, alpha=1):
    label = 255 * label / label.max()
    label = np.repeat(label[:, :, np.newaxis], 3, axis=2)
    img_lab = cv2.addWeighted(image, 1 - alpha, label, alpha, 0)
    return img_lab


def save_by_cv2(path, image):
    cv2.imwrite(path, image)


def get_gradient_map(image_path, method="sobel"):
    """
    提取彩色图像的梯度图

    参数:
        image_path: 图像路径
        method: 梯度计算方法，可选'sobel'或'scharr'（Scharr对小细节更敏感）

    返回:
        gradient_map: 合成的梯度图
        grad_x: 水平方向梯度图
        grad_y: 垂直方向梯度图
    """
    # 读取彩色图像（BGR格式）
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("无法读取图像，请检查路径是否正确")

    # 转换为灰度图（梯度计算通常基于单通道）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 计算水平和垂直方向梯度
    if method == "sobel":
        # Sobel算子：ksize=3为默认值
        grad_x = cv2.Sobel(
            gray, cv2.CV_64F, 1, 0, ksize=3
        )  # 水平方向（x方向，dx=1, dy=0）
        grad_y = cv2.Sobel(
            gray, cv2.CV_64F, 0, 1, ksize=3
        )  # 垂直方向（y方向，dx=0, dy=1）
    elif method == "scharr":
        # Scharr算子（对小细节更敏感，适用于ksize=3的特殊情况）
        grad_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
        grad_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
    elif method == "canny":
        edges = cv2.Canny(gray, 50, 150)
        return edges, None, None
    else:
        raise ValueError("方法仅支持'sobel'或'scharr'")

    # 转换为绝对值（处理负梯度）并归一化到0-255
    grad_x_abs = cv2.convertScaleAbs(grad_x)
    grad_y_abs = cv2.convertScaleAbs(grad_y)

    # 合成梯度图（水平+垂直梯度的加权和）
    gradient_map = cv2.addWeighted(grad_x_abs, 0.5, grad_y_abs, 0.5, 0)

    return gradient_map, grad_x_abs, grad_y_abs


class InstanceWiseAlignmentOptimizer:
    """
    基于实例级别的掩码对齐优化器
    使用原图点乘mask直接处理每个实例，保持梯度连续
    """

    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device

    def get_instances(
        self, label_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor], int]:
        """提取二值标签图中的每个独立实例"""
        label_np = label_tensor.cpu().numpy().astype(np.uint8)
        labeled, num_objects = ndimage.label(label_np)
        inst_map = torch.from_numpy(labeled).to(self.device)

        masks = []
        for k in range(1, num_objects + 1):
            mask = inst_map == k
            masks.append(mask)

        return inst_map, masks, num_objects

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
        base_grid = self.create_base_grid(H, W, self.device, align_corners=False)

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
            return torch.tensor(0.0, device=self.device)

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
            device=self.device,
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32,
            device=self.device,
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
        x = torch.arange(kernel_size, device=self.device) - kernel_size // 2
        y = torch.arange(kernel_size, device=self.device) - kernel_size // 2
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
            print("警告：未在图像中检测到Canny边缘")
            return torch.zeros_like(gray_image)

        # 3. (关键) 只计算外部距离 (到最近边缘的距离)
        # 我们希望mask边缘“滚”到值为0的地方
        dt_map = ndimage.distance_transform_edt(1 - img_edge)

        dt_tensor = torch.from_numpy(dt_map).to(self.device)

        # 归一化DT Map，使损失值更稳定
        return dt_tensor / (dt_tensor.max() + 1e-8)

    def compute_mask_edges(self, mask: torch.Tensor) -> torch.Tensor:
        """计算二值掩码的边缘"""
        # 使用卷积计算边缘
        kernel = torch.tensor(
            [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
            dtype=torch.float32,
            device=self.device,
        ).view(1, 1, 3, 3)

        edges = F.conv2d(mask.float(), kernel, padding=1)

        return edges

    def optimize_instance_offset(
        self,
        original_rgb_image: torch.Tensor,
        mask: torch.Tensor,
        dt_map: torch.Tensor,
        initial_dx: float = 0.0,
        initial_dy: float = 0.0,
        max_iterations: int = 100,
        lr: float = 1e-3,
        reg_weight: float = 0.2,  # L2正则化权重
        mono_weight: float = 0.5,
    ) -> Tuple[float, float, dict]:
        """
        为单个实例优化偏移变量 (使用软边缘DT损失)
        """
        # 初始化可学习参数
        dx = nn.Parameter(torch.tensor(initial_dx, device=self.device))
        dy = nn.Parameter(torch.tensor(initial_dy, device=self.device))

        # (关键) 使用Adam优化器
        optimizer = optim.Adam([dx, dy], lr=lr)

        history = {"dx": [], "dy": [], "loss": [], "dt_loss": [], "mono_loss": []}

        best_loss = float("inf")
        best_dx, best_dy = initial_dx, initial_dy

        for iteration in range(max_iterations):
            optimizer.zero_grad()

            # 1. 应用仿射变换到掩码 (得到软掩码)
            transformed_mask = self.apply_affine_transform(mask, dx, dy)

            mask_sum = transformed_mask.sum()
            if mask_sum < 1e-5:
                # 给予一个巨大的损失让它“弹回”
                total_loss = torch.tensor(1e5, device=self.device, requires_grad=True)
                dt_loss = torch.tensor(0.0)
                mono_loss = torch.tensor(1.0)
            else:
                # 2. 计算软掩码的“软边缘”
                # compute_mask_edges 返回 (B,C,H,W), squeeze 掉
                soft_edges_raw = self.compute_mask_edges(
                    transformed_mask.unsqueeze(0).unsqueeze(0)
                ).squeeze()
                soft_edges = torch.relu(soft_edges_raw)

                # 3. 计算损失
                edge_sum = soft_edges.sum() + 1e-8

                # (关键) DT 损失：我们希望软边缘(soft_edges)所在位置的
                # 距离变换值(dt_map) 尽可能小 (接近0)
                # 我们使用 L1 损失 ( .abs() )，它对异常值更鲁棒
                dt_loss = (dt_map * soft_edges).sum() / edge_sum

                monotonicity_score = self.compute_region_monotonicity(
                    original_rgb_image, transformed_mask
                )
                mono_loss = 1.0 - monotonicity_score

                # L2 正则化 (惩罚大的偏移量)
                # 我们惩罚的是 *从初始值* 开始的偏移，而不是距0的偏移
                dx_change = dx - initial_dx
                dy_change = dy - initial_dy
                reg_loss = reg_weight * torch.sqrt(dx_change**2 + dy_change**2 + 1e-8)

                total_loss = dt_loss + (mono_weight * mono_loss) + reg_loss

            # 反向传播
            total_loss.backward()

            # 梯度裁剪防止爆炸
            torch.nn.utils.clip_grad_norm_([dx, dy], max_norm=1.0)

            optimizer.step()

            # 记录历史
            history["dx"].append(dx.item())
            history["dy"].append(dy.item())
            history["loss"].append(total_loss.item())
            history["dt_loss"].append(dt_loss.item())
            history["mono_loss"].append(mono_loss.item())

            # 更新最佳得分
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_dx, best_dy = dx.item(), dy.item()

            # 早期停止检查
            if iteration > 10 and abs(history["loss"][-1] - history["loss"][-5]) < 1e-6:
                # print(f"  迭代 {iteration+1}: 损失收敛，提前停止。")
                break

        # print(f"  优化完成: dx={best_dx:.4f}, dy={best_dy:.4f}, loss={best_loss:.4f}")
        return best_dx, best_dy, history

    def align_instance_with_multi_start_and_visual(
        self,
        rgb_image: torch.Tensor,
        label_image: torch.Tensor,
        instance_id: int,
        search_range_norm: float = 0.01,  # 归一化搜索范围 (例如 0.05 意味着 +/-5% 图像尺寸)
        num_candidates: int = 8,  # 候选初始化点的数量 (例如 8个方向+中心)
        nms_radius_norm: float = 0.001,  # 非极大值抑制的距离阈值 (归一化)
    ) -> Tuple[float, float, Dict]:
        """
        通过多点初始化和NMS选择最优解来对齐单个实例。

        Args:
            rgb_image: (C, H, W) 原始RGB图像
            label_image: (H, W) 标签图像
            instance_id: 实例ID
            search_range_norm: 初始搜索的归一化范围。
            num_candidates: 初始化的候选点数量。
            nms_radius_norm: NMS抑制半径。

        Returns:
            (best_dx, best_dy, full_history)
        """
        _, masks, num_objects = self.get_instances(label_image)
        if instance_id >= len(masks) or masks[instance_id].sum() == 0:
            print(f"实例 {instance_id} 无效。")
            return 0.0, 0.0, {}
        mask = masks[instance_id]

        # 1. 定义金字塔参数 (与 align_and_visualize_instance 相同)
        # 注意: 如果你调整了参数，这里也需要同步
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

        # 预计算DT Maps
        dt_maps = {}
        for level in pyramid_levels:
            img_blurred = self.gaussian_blur(rgb_image, level["ksize"], level["sigma"])
            dt_maps[level["name"]] = self.compute_dt_map(img_blurred)

        # 2. 生成候选初始化偏移
        # 均匀采样num_candidates个点
        directions = [
            (dx, dy) for dx, dy in itertools.product([-1.0, 0.0, 1.0], repeat=2)
        ]
        candidates = [
            (dx * search_range_norm, dy * search_range_norm) for dx, dy in directions
        ]

        # 3. 对每个候选点进行完整的金字塔优化
        results = []
        print(f"实例 {instance_id}: 开始多点初始化优化 ({len(candidates)} 个候选点)...")

        for initial_dx, initial_dy in candidates:
            current_dx, current_dy = initial_dx, initial_dy
            level_histories = []

            for level in pyramid_levels:
                dx, dy, history = self.optimize_instance_offset(
                    rgb_image,
                    mask,
                    dt_maps[level["name"]],
                    initial_dx=current_dx,  # 上一级的终点是下一级的起点
                    initial_dy=current_dy,
                    max_iterations=level["iters"],
                    lr=level["lr"],
                    reg_weight=level["reg"],
                    mono_weight=level["mono"],
                )
                level_histories.append(history)
                current_dx, current_dy = dx, dy

            # 最终的总损失
            final_loss = (
                level_histories[-1]["loss"][-1] if level_histories else float("inf")
            )

            results.append(
                {
                    "initial_dx": initial_dx,
                    "initial_dy": initial_dy,
                    "final_dx": current_dx,
                    "final_dy": current_dy,
                    "final_loss": final_loss,
                    "history": level_histories,
                }
            )
            print(
                f"  初始({initial_dx:.4f}, {initial_dy:.4f}) -> 最终({current_dx:.4f}, {current_dy:.4f}), L={final_loss:.6f}"
            )

        # 4. 非极大值抑制 (NMS)
        # 按损失（得分）排序：损失越低越好
        results.sort(key=lambda x: x["final_loss"])

        selected_results = []

        # NMS 距离平方阈值
        nms_radius_sq = nms_radius_norm**2

        for res in results:
            is_suppressed = False
            # 检查当前结果是否与已选结果太接近 (即收敛到同一个局部解)
            for sel_res in selected_results:
                dist_sq = (res["final_dx"] - sel_res["final_dx"]) ** 2 + (
                    res["final_dy"] - sel_res["final_dy"]
                ) ** 2

                if dist_sq < nms_radius_sq:
                    is_suppressed = True
                    break

            if not is_suppressed:
                selected_results.append(res)

        # 5. 选择最优解
        if not selected_results:
            # 如果所有解都被抑制了，返回初始中心点（这种情况很少发生）
            print("警告: 所有解都被抑制，返回默认值。")
            return 0.0, 0.0, {}

        # NMS 后第一个解就是损失最低的
        best_result = selected_results[0]
        final_dx = best_result["final_dx"]
        final_dy = best_result["final_dy"]

        print(
            f"实例 {instance_id}: 经过NMS筛选，最终最优偏移为 dx={final_dx:.4f}, dy={final_dy:.4f}"
        )
        print(
            f"  (始于 dx={best_result['initial_dx']:.4f}, dy={best_result['initial_dy']:.4f}), 最终损失={best_result['final_loss']:.6f}"
        )

        # 6. 整合并返回历史记录 (用于可视化)
        # 将所有 level_histories 合并成一个 full_history 结构
        # full_history = {
        #     "dx": [],
        #     "dy": [],
        #     "loss": [],
        #     "dt_loss": [],
        #     "mono_loss": [],
        #     "level": [],
        # }

        # for level_history in best_result["history"]:
        #     for key in ["dx", "dy", "loss", "dt_loss", "mono_loss"]:
        #         full_history[key].extend(level_history[key])
        #     level_name = level_history.get("level_name", "Unknown")

        # 可视化不同初始化点的优化结果
        plt.rcParams["font.sans-serif"] = ["SimHei"]
        plt.rcParams["axes.unicode_minus"] = False

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # 转换图像到numpy格式用于显示
        rgb_np = (rgb_image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        mask_np = mask.cpu().numpy()

        indices = np.where(mask_np)
        center_y = int(np.mean(indices[0]))
        center_x = int(np.mean(indices[1]))

        # 1. 左图：显示所有初始点和优化路径
        ax1.imshow(rgb_np)
        # 显示原始mask
        overlay = rgb_np.copy()
        overlay[mask_np > 0] = overlay[mask_np > 0] * 0.7 + np.array([255, 0, 0]) * 0.3
        ax1.imshow(overlay, alpha=0.5)

        # 为每个选中的结果使用不同颜色
        colors = plt.cm.rainbow(np.linspace(0, 1, len(selected_results)))

        H, W = rgb_image.shape[-2:]
        for res, color in zip(selected_results, colors):
            # 画出初始点
            init_x = int(center_x + res["initial_dx"] * W)
            init_y = int(center_y + res["initial_dy"] * H)
            ax1.plot(
                init_x,
                init_y,
                "o",
                color=color,
                markersize=8,
                label=f'Init ({res["initial_dx"]:.3f}, {res["initial_dy"]:.3f})',
            )

            # 画出最终点
            final_x = int(center_x + res["final_dx"] * W)
            final_y = int(center_y + res["final_dy"] * H)
            ax1.plot(final_x, final_y, "*", color=color, markersize=10)

            # 画出优化路径
            dx_history = []
            dy_history = []
            for level_hist in res["history"]:
                dx_history.extend(level_hist["dx"])
                dy_history.extend(level_hist["dy"])

            path_x = [int(center_x + dx * W) for dx in dx_history]
            path_y = [int(center_y + dy * H) for dy in dy_history]
            ax1.plot(path_x, path_y, "-", color=color, alpha=0.5, linewidth=1)

        ax1.set_title("优化路径可视化")
        ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax1.axis("off")

        # 2. 右图：显示最优结果
        best_result = selected_results[0]
        transformed_mask = self.apply_affine_transform(
            mask,
            torch.tensor(best_result["final_dx"], device=self.device),
            torch.tensor(best_result["final_dy"], device=self.device),
        )
        transformed_mask_np = transformed_mask.cpu().numpy()

        overlay_final = rgb_np.copy()
        # 显示原始mask（红色）
        overlay_final[mask_np > 0] = (
            overlay_final[mask_np > 0] * 0.7 + np.array([255, 0, 0]) * 0.3
        )
        # 显示最优位置mask（绿色）
        overlay_final[transformed_mask_np > 0.3] = (
            overlay_final[transformed_mask_np > 0.3] * 0.7 + np.array([0, 255, 0]) * 0.3
        )

        ax2.imshow(overlay_final)
        ax2.set_title(
            f"最优偏移结果 dx={best_result['final_dx']:.4f}, dy={best_result['final_dy']:.4f}"
        )
        ax2.axis("off")

        plt.tight_layout()
        plt.show()

        # 保存图像
        os.makedirs(os.path.join(SAVE_FIG_DIR, "multi_start"), exist_ok=True)
        fig.savefig(
            os.path.join(
                SAVE_FIG_DIR,
                "multi_start",
                f"instance_{instance_id}_multi_start_optimization.png",
            ),
            bbox_inches="tight",
        )
        plt.close()

    def align_instance_with_multi_start(
        self,
        rgb_image: torch.Tensor,
        label_image: torch.Tensor,
        search_range_norm: float = 0.01,  # 归一化搜索范围 (例如 0.05 意味着 +/-5% 图像尺寸)
        num_candidates: int = 8,  # 候选初始化点的数量 (例如 8个方向+中心)
        nms_radius_norm: float = 0.001,  # 非极大值抑制的距离阈值 (归一化)
        returns_instance: bool = False,
    ) -> Tuple[float, float, Dict]:
        """
        通过多点初始化和NMS选择最优解来对齐单个实例。

        Args:
            rgb_image: (C, H, W) 原始RGB图像
            label_image: (H, W) 标签图像
            instance_id: 实例ID
            search_range_norm: 初始搜索的归一化范围。
            num_candidates: 初始化的候选点数量。
            nms_radius_norm: NMS抑制半径。

        Returns:
            (best_dx, best_dy, full_history)
        """
        _, masks, num_objects = self.get_instances(label_image)

        # 1. 定义金字塔参数 (与 align_and_visualize_instance 相同)
        # 注意: 如果你调整了参数，这里也需要同步
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

        # 预计算DT Maps
        dt_maps = {}
        for level in pyramid_levels:
            img_blurred = self.gaussian_blur(rgb_image, level["ksize"], level["sigma"])
            dt_maps[level["name"]] = self.compute_dt_map(img_blurred)

        # 2. 生成候选初始化偏移
        # 均匀采样num_candidates个点
        # directions = [
        #     (dx, dy)
        #     for dx, dy in itertools.product([1.0], repeat=2)  # [-1.0, 0.0, 1.0]
        # ]
        directions = [(1, -1)]
        candidates = [
            (dx * search_range_norm, dy * search_range_norm) for dx, dy in directions
        ]

        optimized_offsets = []

        for i, mask in enumerate(masks):
            print(f"优化实例 {i+1}/{num_objects} 的偏移量...")

            # 检查掩码是否有效
            if mask.sum() == 0:
                print(f"实例 {i+1}: 掩码为空，跳过")
                optimized_offsets.append((0.0, 0.0, mask, 0.0, 0.0, 0.0))
                continue

            # 3. 对每个候选点进行完整的金字塔优化
            results = []
            print(f"实例 {i}: 开始多点初始化优化 ({len(candidates)} 个候选点)...")

            for initial_dx, initial_dy in candidates:
                current_dx, current_dy = initial_dx, initial_dy
                level_histories = []

                for level in pyramid_levels:
                    dx, dy, history = self.optimize_instance_offset(
                        rgb_image,
                        mask,
                        dt_maps[level["name"]],
                        initial_dx=current_dx,  # 上一级的终点是下一级的起点
                        initial_dy=current_dy,
                        max_iterations=level["iters"],
                        lr=level["lr"],
                        reg_weight=level["reg"],
                        mono_weight=level["mono"],
                    )
                    level_histories.append(history)
                    current_dx, current_dy = dx, dy

                # 最终的总损失
                final_loss = (
                    level_histories[-1]["loss"][-1] if level_histories else float("inf")
                )

                results.append(
                    {
                        "initial_dx": initial_dx,
                        "initial_dy": initial_dy,
                        "final_dx": current_dx,
                        "final_dy": current_dy,
                        "final_loss": final_loss,
                        "history": level_histories,
                    }
                )
                print(
                    f"  初始({initial_dx:.4f}, {initial_dy:.4f}) -> 最终({current_dx:.4f}, {current_dy:.4f}), L={final_loss:.6f}"
                )

            # 4. 非极大值抑制 (NMS)
            # 按损失（得分）排序：损失越低越好
            results.sort(key=lambda x: x["final_loss"])

            selected_results = []

            # NMS 距离平方阈值
            nms_radius_sq = nms_radius_norm**2

            for res in results:
                is_suppressed = False
                # 检查当前结果是否与已选结果太接近 (即收敛到同一个局部解)
                for sel_res in selected_results:
                    dist_sq = (res["final_dx"] - sel_res["final_dx"]) ** 2 + (
                        res["final_dy"] - sel_res["final_dy"]
                    ) ** 2

                    if dist_sq < nms_radius_sq:
                        is_suppressed = True
                        break

                if not is_suppressed:
                    selected_results.append(res)

            # 5. 选择最优解
            if not selected_results:
                optimized_offsets.append((0, 0, mask, 0, 0, 0))

            # NMS 后第一个解就是损失最低的
            best_result = selected_results[0]
            final_dx = best_result["final_dx"]
            final_dy = best_result["final_dy"]

            print(
                f"实例 {i}: 经过NMS筛选，最终最优偏移为 dx={final_dx:.4f}, dy={final_dy:.4f}"
            )
            print(
                f"  (始于 dx={best_result['initial_dx']:.4f}, dy={best_result['initial_dy']:.4f}), 最终损失={best_result['final_loss']:.6f}"
            )
            optimized_offsets.append((final_dx, final_dy, mask, 0, 0, 0))

        if returns_instance:
            return optimized_offsets

        return self.apply_optimized_offsets(label_image, optimized_offsets)

    def instance_wise_alignment(
        self,
        rgb_image: torch.Tensor,
        label_image: torch.Tensor,
        initial_dx: float = 0.0,
        initial_dy: float = 0.0,
    ) -> torch.Tensor:
        """
        基于实例级别的标签对齐主函数
        """
        H, W = rgb_image.shape[-2:]

        # 提取实例
        inst_map, masks, num_objects = self.get_instances(label_image)
        print(f"检测到 {num_objects} 个建筑物实例")

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

        # 预计算所有尺度的 DT Maps
        dt_maps = []
        for level in pyramid_levels:
            print(f"正在计算 sigma={level['sigma']} 的DT Map...")
            img_blurred = self.gaussian_blur(rgb_image, level["ksize"], level["sigma"])
            dt_maps.append(self.compute_dt_map(img_blurred))

        # 存储优化结果
        optimized_offsets = []

        for i, mask in enumerate(masks):
            print(f"优化实例 {i+1}/{num_objects} 的偏移量...")

            # 检查掩码是否有效
            if mask.sum() == 0:
                print(f"实例 {i+1}: 掩码为空，跳过")
                optimized_offsets.append((0.0, 0.0, mask, 0.0, 0.0, 0.0))
                continue

            current_dx, current_dy = initial_dx, initial_dy

            for level, dt_map in zip(pyramid_levels, dt_maps):
                # print(f"  在 sigma={level['sigma']} 尺度上优化...")
                current_dx, current_dy, _ = self.optimize_instance_offset(
                    rgb_image,
                    mask,
                    dt_map,
                    initial_dx=current_dx,
                    initial_dy=current_dy,
                    max_iterations=level["iters"],
                    lr=level["lr"],
                    reg_weight=level["reg"],
                )

            print(f"实例 {i+1}: 最终偏移(dx={current_dx:.4f}, dy={current_dy:.4f})")
            optimized_offsets.append((current_dx, current_dy, mask, 0, 0, 0))

        # 应用优化后的偏移量
        return self.apply_optimized_offsets(label_image, optimized_offsets)

    def apply_optimized_offsets(
        self, label_image: torch.Tensor, optimized_offsets: List[Tuple]
    ) -> torch.Tensor:
        """
        应用优化后的偏移量到标签图像
        使用可微分的网格采样保持梯度连续
        """
        H, W = label_image.shape[-2:]

        # 创建基础网格
        base_grid = self.create_base_grid(H, W, self.device, align_corners=False)

        # 初始化位移场
        disp_field = torch.zeros(H, W, 2, device=self.device)

        # 为每个实例创建位移场
        for dx, dy, mask, *_ in optimized_offsets:
            if abs(dx) < 1e-5 and abs(dy) < 1e-5:  # 忽略零偏移
                continue

            original_mask = mask.clone()
            # 在掩码区域应用位移
            y_indices, x_indices = torch.where(mask)
            new_y = torch.clamp((y_indices.float() + dy * H).round().long(), 0, H - 1)
            new_x = torch.clamp((x_indices.float() + dx * W).round().long(), 0, W - 1)

            # create shifted mask
            shifted_mask = torch.zeros_like(mask)
            shifted_mask[new_y, new_x] = True

            # compute union mask
            union_mask = original_mask | shifted_mask

            # apply displacements to union region
            disp_field[..., 0][union_mask] = dx
            disp_field[..., 1][union_mask] = dy

        # 创建采样网格
        sample_grid = base_grid - disp_field * 2  # 转换为[-2,2]范围
        sample_grid = torch.clamp(sample_grid, -1, 1)

        # 应用位移变换（保持可微分）
        if label_image.dim() == 2:
            label_input = label_image.unsqueeze(0).unsqueeze(0).float()
        else:
            label_input = label_image.unsqueeze(0).float()

        labels_aligned = F.grid_sample(
            label_input,
            sample_grid.unsqueeze(0),
            mode="bilinear",
            align_corners=False,
            padding_mode="zeros",
        )

        return torch.clamp(labels_aligned.squeeze(), 0.0, 1.0)

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

    def align_and_visualize_instance(
        self,
        rgb_image: torch.Tensor,
        label_image: torch.Tensor,
        instance_id: int,
        initial_dx: float = 0.0,
        initial_dy: float = 0.0,
    ) -> None:
        """
        (新) 使用多尺度金字塔策略对齐单个实例并可视化
        """
        plt.rcParams["font.sans-serif"] = ["SimHei"]
        plt.rcParams["axes.unicode_minus"] = False

        # --- 1. 提取实例 ---
        _, masks, num_objects = self.get_instances(label_image)
        if instance_id >= len(masks):
            print(f"实例ID {instance_id} 不存在")
            return
        mask = masks[instance_id]
        if mask.sum() == 0:
            print(f"实例 {instance_id} 的掩码为空")
            return

        # --- 2. 定义金字塔层级 ---
        # (sigma, kernel_size, iterations, learning_rate, reg_weight)
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

        # 预计算 DT Maps
        dt_maps = {}
        blurred_images = {}
        print("正在创建图像金字塔和DT Maps...")
        for level in pyramid_levels:
            name = level["name"]
            img_blurred = self.gaussian_blur(rgb_image, level["ksize"], level["sigma"])
            blurred_images[name] = img_blurred
            dt_maps[name] = self.compute_dt_map(img_blurred)

            # (可选) 保存DT Map以供调试
            # dt_map_vis = (dt_maps[name].cpu().numpy() * 255).astype(np.uint8)
            # cv2.imwrite(
            #     os.path.join(SAVE_FIG_DIR, f"dt_map_{name}_{instance_id}.png"),
            #     dt_map_vis,
            # )

        # --- 3. 执行由粗到精的优化 ---
        current_dx, current_dy = initial_dx, initial_dy
        full_history = {
            "dx": [],
            "dy": [],
            "loss": [],
            "dt_loss": [],
            "mono_loss": [],
            "level": [],
        }

        print(f"开始优化实例 {instance_id} (Init dx={initial_dx}, dy={initial_dy})...")
        for level in pyramid_levels:
            name = level["name"]
            print(f"--- 正在优化 {name} 尺度 (Sigma={level['sigma']}) ---")

            dx, dy, history = self.optimize_instance_offset(
                rgb_image,  # (关键) 传入原始图像
                mask,
                dt_maps[name],
                initial_dx=current_dx,
                initial_dy=current_dy,
                max_iterations=level["iters"],
                lr=level["lr"],
                reg_weight=level["reg"],
                mono_weight=level["mono"],  # (关键) 传入权重
            )

            # 记录历史并更新
            full_history["dx"].extend(history["dx"])
            full_history["dy"].extend(history["dy"])
            full_history["loss"].extend(history["loss"])
            full_history["dt_loss"].extend(history["dt_loss"])
            full_history["mono_loss"].extend(history["mono_loss"])
            full_history["level"].extend([name] * len(history["loss"]))
            current_dx, current_dy = dx, dy

            print(f"  {name} 尺度完成: dx={current_dx:.4f}, dy={current_dy:.4f}")

        final_dx, final_dy = current_dx, current_dy
        print(
            f"实例 {instance_id}: 最终优化偏移 (dx={final_dx:.4f}, dy={final_dy:.4f})"
        )

        # --- 4. 绘制优化过程 ---
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # 1. 总损失变化
        axes[0, 0].plot(full_history["loss"], label="总损失 (DT + Reg)")
        axes[0, 0].plot(full_history["dt_loss"], label="DT损失", linestyle="--")
        axes[0, 0].plot(
            full_history["mono_loss"], label="Mono损失 (方差)", linestyle=":"
        )
        axes[0, 0].set_title("损失优化过程")
        axes[0, 0].set_xlabel("总迭代次数")
        axes[0, 0].set_ylabel("损失")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # 添加尺度分隔线
        level_changes = [
            i
            for i, (l1, l2) in enumerate(
                zip(full_history["level"][:-1], full_history["level"][1:])
            )
            if l1 != l2
        ]
        for xc in level_changes:
            axes[0, 0].axvline(x=xc, color="r", linestyle="--", linewidth=0.8)
            axes[0, 1].axvline(x=xc, color="r", linestyle="--", linewidth=0.8)

        # 2. 偏移量变化
        axes[0, 1].plot(full_history["dx"], label="δx")
        axes[0, 1].plot(full_history["dy"], label="δy")
        axes[0, 1].set_title("偏移量优化过程")
        axes[0, 1].set_xlabel("总迭代次数")
        axes[0, 1].set_ylabel("归一化偏移量")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # 3. 最终DT Map (Fine)
        axes[0, 2].imshow(dt_maps["Fine"].cpu().numpy(), cmap="gray")
        axes[0, 2].set_title("DT Map (Fine 尺度)")
        axes[0, 2].axis("off")

        # 4. 初始位置
        rgb_np = (rgb_image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        mask_np = mask.cpu().numpy()
        vis_initial = rgb_np.copy()
        vis_initial[mask_np > 0] = (
            vis_initial[mask_np > 0] * 0.7 + np.array([255, 0, 0]) * 0.3
        )
        axes[1, 0].imshow(vis_initial)
        axes[1, 0].set_title("初始位置掩码 (红)")
        axes[1, 0].axis("off")

        # 5. 优化后的掩码
        transformed_mask = self.apply_affine_transform(mask, final_dx, final_dy)
        transformed_mask_np = transformed_mask.cpu().numpy()
        vis_final = rgb_np.copy()
        vis_final[transformed_mask_np > 0.3] = (  # 使用0.3阈值进行可视化
            vis_final[transformed_mask_np > 0.3] * 0.7 + np.array([0, 255, 0]) * 0.3
        )
        axes[1, 1].imshow(vis_final)
        axes[1, 1].set_title("优化后掩码 (绿)")
        axes[1, 1].axis("off")

        # 6. 对比
        overlay = rgb_np.copy()
        overlay[mask_np > 0] = overlay[mask_np > 0] * 0.5 + np.array([255, 0, 0]) * 0.5
        overlay[transformed_mask_np > 0.3] = (
            overlay[transformed_mask_np > 0.3] * 0.5 + np.array([0, 255, 0]) * 0.5
        )
        axes[1, 2].imshow(overlay)
        axes[1, 2].set_title("红:初始, 绿:最终")
        axes[1, 2].axis("off")

        plt.tight_layout()
        plt.show()

        # 保存图像
        os.makedirs(os.path.join(SAVE_FIG_DIR, "instance"), exist_ok=True)
        fig.savefig(
            os.path.join(
                SAVE_FIG_DIR, "instance", f"instance_{instance_id}_alignment.png"
            )
        )


class InstanceOffsetStatistics:
    def __init__(self, output_dir="./vis_logs/aligner_instance_wise/"):
        self.all_offsets = []  # (dx, dy)
        self.instance_data = []  # 详细数据
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def add_instance(self, image_name, gt_area, dx, dy, centroid):
        distance = np.sqrt(dx**2 + dy**2)
        self.all_offsets.append((dx, dy))
        self.instance_data.append(
            {
                "image_name": image_name,
                "gt_area": gt_area,
                "dx": dx,
                "dy": dy,
                "distance": distance,
                "gt_centroid": centroid,
            }
        )

    def save_results(self):
        if self.instance_data:
            df = pd.DataFrame(self.instance_data)
            df.to_csv(
                os.path.join(self.output_dir, "instance_offset_emi_details.csv"),
                index=False,
            )
            print(
                f"实例详情已保存到: {os.path.join(self.output_dir, 'instance_offset_emi_details.csv')}"
            )

    def plot_statistics(self):
        if not self.all_offsets:
            print("没有偏移数据可绘制！")
            return

        dx_values = [offset[0] for offset in self.all_offsets]
        dy_values = [offset[1] for offset in self.all_offsets]
        distances = [np.sqrt(dx**2 + dy**2) for dx, dy in self.all_offsets]

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("Instance Offset Analysis (EMI)", fontsize=16, fontweight="bold")

        axes[0, 0].hist(
            dx_values, bins=50, alpha=0.7, color="skyblue", edgecolor="black"
        )
        axes[0, 0].axvline(
            np.mean(dx_values),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {np.mean(dx_values):.2f}",
        )
        axes[0, 0].set_xlabel("X Offset (pixels)")
        axes[0, 0].set_ylabel("Number of Instances")
        axes[0, 0].set_title("X Offset Distribution")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].hist(
            dy_values, bins=50, alpha=0.7, color="lightcoral", edgecolor="black"
        )
        axes[0, 1].axvline(
            np.mean(dy_values),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {np.mean(dy_values):.2f}",
        )
        axes[0, 1].set_xlabel("Y Offset (pixels)")
        axes[0, 1].set_ylabel("Number of Instances")
        axes[0, 1].set_title("Y Offset Distribution")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        axes[0, 2].hist(
            distances, bins=50, alpha=0.7, color="lightgreen", edgecolor="black"
        )
        axes[0, 2].axvline(
            np.mean(distances),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {np.mean(distances):.2f}",
        )
        axes[0, 2].set_xlabel("Total Offset Distance (pixels)")
        axes[0, 2].set_ylabel("Number of Instances")
        axes[0, 2].set_title("Total Offset Distance Distribution")
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        axes[1, 0].scatter(dx_values, dy_values, alpha=0.6, s=10, color="blue")
        axes[1, 0].axhline(0, color="black", linewidth=0.5)
        axes[1, 0].axvline(0, color="black", linewidth=0.5)
        axes[1, 0].set_xlabel("X Offset (pixels)")
        axes[1, 0].set_ylabel("Y Offset (pixels)")
        axes[1, 0].set_title("Offset Direction Distribution")
        axes[1, 0].grid(True, alpha=0.3)

        hb = axes[1, 1].hist2d(dx_values, dy_values, bins=50, cmap="viridis")
        axes[1, 1].axhline(0, color="white", linewidth=0.5)
        axes[1, 1].axvline(0, color="white", linewidth=0.5)
        axes[1, 1].set_xlabel("X Offset (pixels)")
        axes[1, 1].set_ylabel("Y Offset (pixels)")
        axes[1, 1].set_title("2D Offset Distribution Heatmap")
        plt.colorbar(hb[3], ax=axes[1, 1])

        axes[1, 2].axis("off")
        stats_text = f"""
Statistical Summary (EMI):
Total Instances: {len(self.all_offsets)}
X Offset:
  Mean: {np.mean(dx_values):.2f} px
  Std: {np.std(dx_values):.2f} px
  Range: [{np.min(dx_values):.2f}, {np.max(dx_values):.2f}] px

Y Offset:
  Mean: {np.mean(dy_values):.2f} px
  Std: {np.std(dy_values):.2f} px
  Range: [{np.min(dy_values):.2f}, {np.max(dy_values):.2f}] px

Total Distance:
  Mean: {np.mean(distances):.2f} px
  Std: {np.std(distances):.2f} px
  Max Distance: {np.max(distances):.2f} px
        """
        axes[1, 2].text(
            0.1,
            0.9,
            stats_text,
            transform=axes[1, 2].transAxes,
            fontsize=10,
            verticalalignment="top",
            family="monospace",
        )

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "offset_statistics_emi.png"),
            dpi=300,
            bbox_inches="tight",
        )
        # plt.show()

        self.plot_direction_rose(dx_values, dy_values)

    def plot_direction_rose(self, dx_values, dy_values):
        angles = np.arctan2(dy_values, dx_values)
        angles_deg = np.degrees(angles)
        angles_deg[angles_deg < 0] += 360
        distances = [np.sqrt(dx**2 + dy**2) for dx, dy in zip(dx_values, dy_values)]

        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(15, 6), subplot_kw=dict(projection="polar")
        )
        fig.suptitle("Offset Direction Analysis (EMI)", fontsize=16, fontweight="bold")

        n_bins = 36
        hist, bin_edges = np.histogram(angles_deg, bins=n_bins, range=(0, 360))
        theta = np.deg2rad(np.arange(0, 360, 360 / n_bins))

        ax1.bar(
            theta,
            hist,
            width=2 * np.pi / n_bins,
            alpha=0.7,
            color="skyblue",
            edgecolor="black",
        )
        ax1.set_title("Offset Direction Frequency", pad=20)
        ax1.set_theta_zero_location("N")
        ax1.set_theta_direction(-1)

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        mean_distances = []
        for i in range(len(bin_edges) - 1):
            mask = (angles_deg >= bin_edges[i]) & (angles_deg < bin_edges[i + 1])
            if mask.any():
                mean_distances.append(np.mean(np.array(distances)[mask]))
            else:
                mean_distances.append(0)

        ax2.bar(
            theta,
            mean_distances,
            width=2 * np.pi / n_bins,
            alpha=0.7,
            color="lightcoral",
            edgecolor="black",
        )
        ax2.set_title("Mean Offset Distance by Direction", pad=20)
        ax2.set_theta_zero_location("N")
        ax2.set_theta_direction(-1)

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "offset_direction_rose_emi.png"),
            dpi=300,
            bbox_inches="tight",
        )
        # plt.show()

    def get_summary_statistics(self):
        if not self.all_offsets:
            return {}

        dx_values = [offset[0] for offset in self.all_offsets]
        dy_values = [offset[1] for offset in self.all_offsets]
        distances = [np.sqrt(dx**2 + dy**2) for dx, dy in self.all_offsets]

        summary = {
            "total_instances": len(self.all_offsets),
            "dx_mean": float(np.mean(dx_values)),
            "dx_std": float(np.std(dx_values)),
            "dx_min": float(np.min(dx_values)),
            "dx_max": float(np.max(dx_values)),
            "dy_mean": float(np.mean(dy_values)),
            "dy_std": float(np.std(dy_values)),
            "dy_min": float(np.min(dy_values)),
            "dy_max": float(np.max(dy_values)),
            "distance_mean": float(np.mean(distances)),
            "distance_std": float(np.std(distances)),
            "distance_max": float(np.max(distances)),
        }
        return summary


SAVE_FIG_DIR = "./vis_logs/Antakya/offset_analysis/"
os.makedirs(SAVE_FIG_DIR, exist_ok=True)


def main():
    image_paths = "../data/segmentation/Turkey/Antakya/pre/test/images"
    label_paths = "../data/segmentation/Turkey/Antakya/pre/test/labels"
    stats = InstanceOffsetStatistics(output_dir=SAVE_FIG_DIR)
    for file_name in os.listdir(image_paths):
        if not file_name.endswith(".png"):
            continue
        image_name = file_name.split(".")[0]
        image_path = os.path.join(image_paths, file_name)
        label_path = os.path.join(label_paths, file_name)

        image_bgr = cv2.imread(image_path)
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        label = cv2.cvtColor(
            cv2.imread(label_path),
            cv2.COLOR_BGR2GRAY,
        )
        print(
            "image.shape: ",
            image.shape,
            "label.shape: ",
            label.shape,
            "label.unique: ",
            np.unique(label),
        )
        H, W = image.shape[0], image.shape[1]
        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            label = cv2.resize(label, (H, W), interpolation=cv2.INTER_NEAREST)
            print("label.shape: ", label.shape)

        # img_lab = img_with_label(
        #     image_bgr.astype(np.float32), label.astype(np.float32), alpha=0.5
        # )
        # save_by_cv2("image+label_alpha05.png", img_lab)

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        rgb_image = torch.tensor(image).permute(2, 0, 1).to(device) / 255.0
        label_image = torch.tensor(label, dtype=torch.float32).to(device) / 255.0

        # 创建优化器
        aligner = InstanceWiseAlignmentOptimizer(device=device)

        # aligner.align_and_visualize_instance(
        #     rgb_image, label_image, instance_id=20, initial_dx=0.01, initial_dy=0.01
        # )
        # aligner.align_instance_with_multi_start_and_visual(
        #     rgb_image,
        #     label_image,
        #     instance_id=20,
        #     search_range_norm=0.01,
        #     num_candidates=8,
        #     nms_radius_norm=0.001,
        # )

        # print("开始对齐所有实例...")
        # aligned_labels_tensor = aligner.align_instance_with_multi_start(
        #     rgb_image,
        #     label_image,
        #     search_range_norm=0.01,
        #     num_candidates=8,
        #     nms_radius_norm=0.001,
        # )
        # print("所有实例对齐完成。")
        # aligned_labels_np = (aligned_labels_tensor.cpu().numpy() * 255).astype(np.uint8)
        # img_lab_shifted = img_with_label(
        #     image_bgr.astype(np.float32), aligned_labels_np.astype(np.float32), alpha=0.5
        # )
        # save_by_cv2(
        #     os.path.join(SAVE_FIG_DIR, f"{image_name}_image+label_shifted_all.png"), img_lab_shifted
        # )

        # 获取所有实例偏移
        optim_instances = aligner.align_instance_with_multi_start(
            rgb_image,
            label_image,
            search_range_norm=0.01,
            num_candidates=8,
            nms_radius_norm=0.001,
            returns_instance=True,
        )
        for i, (dx, dy, mask, *_) in enumerate(optim_instances):
            indices = torch.where(mask)
            centroid = (
                float(indices[1].float().mean()),
                float(indices[0].float().mean()),
            )
            stats.add_instance(image_name, int(mask.sum().item()), dx, dy, centroid)

    stats.save_results()
    stats.plot_statistics()
    summary = stats.get_summary_statistics()
    print("\n=== 偏移量统计汇总 (EMI方法) ===")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
