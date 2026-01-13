import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy.ndimage as ndimage
import cv2
import itertools
from typing import List, Tuple, Optional, Dict

# Matplotlib 用于绘图
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from tools.alignment import InstanceWiseAlignmentOptimizer


def generate_loss_landscape(
    optimizer: InstanceWiseAlignmentOptimizer,
    rgb_image: torch.Tensor,
    mask: torch.Tensor,
    pyramid_level: Dict,
    search_range_px: int = 20,
    step_px: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算给定图像、掩码和金字塔层级的损失景观

    参数:
        optimizer: 优化器实例
        rgb_image: 裁剪后的 RGB 图像张量 (C, H, W)
        mask: 裁剪后的偏移标签张量 (H, W)
        pyramid_level: 包含 'sigma', 'ksize', 'reg', 'mono' 的字典
        search_range_px: 搜索的像素范围 (例如 20, 将搜索 -20 到 +20)
        step_px: 搜索步长 (像素)

    返回:
        DX_PX: X 轴偏移网格 (像素)
        DY_PX: Y 轴偏移网格 (像素)
        LOSS_MAP: 对应的损失值网格
    """
    device = rgb_image.device
    H, W = rgb_image.shape[-2:]

    # 1. 预计算该层级所需的 DT Map
    print(f"  Calculating DT Map (sigma={pyramid_level['sigma']})...")
    img_blurred = optimizer.gaussian_blur(
        rgb_image, pyramid_level["ksize"], pyramid_level["sigma"]
    )
    dt_map = optimizer.compute_dt_map(img_blurred)

    # 2. 创建搜索网格 (像素单位)
    dx_range = np.arange(-search_range_px, search_range_px + step_px, step_px)
    dy_range = np.arange(-search_range_px, search_range_px + step_px, step_px)
    DX_PX, DY_PX = np.meshgrid(dx_range, dy_range)

    # 3. 将像素偏移转换为归一化偏移 (dx_norm = P_pixels / W)
    DX_NORM = DX_PX / W
    DY_NORM = DY_PX / H

    LOSS_MAP = np.zeros_like(DX_PX, dtype=np.float32)

    # 4. 遍历网格计算损失
    print("  Calculating loss grid...")
    for i in range(DX_NORM.shape[0]):
        for j in range(DX_NORM.shape[1]):
            dx_norm_val = DX_NORM[i, j]
            dy_norm_val = DY_NORM[i, j]

            # 将 numpy 值转换为 tensor
            dx_tensor = torch.tensor(dx_norm_val, device=device, dtype=torch.float32)
            dy_tensor = torch.tensor(dy_norm_val, device=device, dtype=torch.float32)

            # --- 这部分逻辑完全复刻自您的 optimize_instance_offset 函数 ---

            # 1. 应用仿射变换 (平移)
            transformed_mask = optimizer.apply_affine_transform(
                mask, dx_tensor, dy_tensor
            )

            mask_sum = transformed_mask.sum()
            if mask_sum < 1e-5:
                # 如果掩码移出边界, 给一个高损失
                inst_loss = torch.tensor(10.0, device=device)

            else:
                # 2. 计算软边缘
                soft_edges_raw = optimizer.compute_mask_edges(
                    transformed_mask.unsqueeze(0).unsqueeze(0)
                ).squeeze()
                soft_edges = torch.relu(soft_edges_raw)

                # 3. 计算 DT 损失 (边缘对齐损失)
                edge_sum = soft_edges.sum() + 1e-8
                dt_loss = (dt_map * soft_edges).sum() / edge_sum

                # 4. 计算单调性损失 (内部一致性损失)
                monotonicity_score = optimizer.compute_region_monotonicity(
                    rgb_image, transformed_mask
                )
                mono_loss = 1.0 - monotonicity_score

                # 5. L2 正则化损失 (惩罚远离原点 (0,0) 的偏移)
                # 在这里，我们假设 (0,0) 是初始位置
                reg_loss = pyramid_level["reg"] * torch.sqrt(
                    dx_tensor**2 + dy_tensor**2 + 1e-8
                )

                # 6. 总损失
                inst_loss = dt_loss + (pyramid_level["mono"] * mono_loss) + reg_loss

            LOSS_MAP[i, j] = inst_loss.item()

    return DX_PX, DY_PX, LOSS_MAP


# --- 3. 新增：用于绘制 3D 损失景观的函数 ---


def plot_loss_landscape(
    DX: np.ndarray,
    DY: np.ndarray,
    Z_LOSS: np.ndarray,
    title: str,
    cmap: str = "viridis",
    save_path: str = "./",
):
    """
    绘制学术风格的 3D 损失曲面图
    """
    print(f"Plotting '{title}'...")

    # 设置学术风格
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": "Arial",  # 或者 'Times New Roman'
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    # 绘制 3D 曲面
    # rstride/cstride 控制采样密度, antialiased 抗锯齿
    surf = ax.plot_surface(
        DX, DY, Z_LOSS, cmap=cmap, rstride=1, cstride=1, antialiased=True, alpha=0.9
    )

    # 标签和标题
    ax.set_xlabel("\nOffset dx (pixels)", linespacing=2)
    ax.set_ylabel("\nOffset dy (pixels)", linespacing=2)
    ax.set_zlabel("\nLoss Value", linespacing=2)
    # ax.set_title(title, pad=20, weight="bold")

    # 添加颜色条
    cbar = fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.1)
    cbar.set_label("Loss Value")

    # 调整视角 (仰角, 方位角)
    ax.view_init(elev=35, azim=-120)

    # 找到最低点
    min_loss_idx = np.unravel_index(np.argmin(Z_LOSS), Z_LOSS.shape)
    min_loss = Z_LOSS[min_loss_idx]
    min_dx = DX[min_loss_idx]
    min_dy = DY[min_loss_idx]

    # 在最低点绘制一个标记
    ax.plot(
        [min_dx],
        [min_dy],
        [min_loss * 0.95],  # 稍微放低一点防止被曲面遮挡 [min_loss * 0.95]
        "rx",
        markersize=10,
        markeredgewidth=3,
        label=f"Loss Min ({min_loss:.3f}) at ({min_dx}, {min_dy})",
    )

    # 调整布局
    plt.tight_layout()

    # 保存为高分辨率图像
    filename = f"{title}.png"
    save_path = os.path.join(save_path, filename)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {save_path}")
    # plt.show()


def load_and_crop(file_path, x, y, crop_size, is_mask=False):
    """加载、裁剪图像/掩码，并根据需要转换为灰度图"""
    if not os.path.exists(file_path):
        # 即使文件不存在，也要返回一个黑色占位符，保持布局
        print(f"警告：文件不存在 -> {file_path}")
        return np.zeros((crop_size, crop_size, 3), dtype=np.uint8)

    img = cv2.imread(
        file_path, cv2.IMREAD_COLOR if not is_mask else cv2.IMREAD_GRAYSCALE
    )
    if img is None:
        print(f"警告：无法读取图像 -> {file_path}")
        return np.zeros((crop_size, crop_size, 3), dtype=np.uint8)

    # 确保裁剪尺寸不超出图像范围
    H, W = img.shape[:2]
    x_end = min(x + crop_size, W)
    y_end = min(y + crop_size, H)

    # 裁剪图像
    cropped_img = img[y:y_end, x:x_end]

    # 如果裁剪后尺寸不足 (x_end - x < crop_size)，则填充黑色
    H_cropped, W_cropped = cropped_img.shape[:2]
    if H_cropped != crop_size or W_cropped != crop_size:
        # 确定需要填充的尺寸
        H_pad = crop_size - H_cropped
        W_pad = crop_size - W_cropped

        # 创建一个全黑的画布，然后将裁剪部分粘贴上去
        if is_mask:
            # Mask 返回单通道，后续需要上色
            canvas = np.zeros((crop_size, crop_size), dtype=np.uint8)
            canvas[:H_cropped, :W_cropped] = cropped_img
            return canvas
        else:
            # 图像返回3通道 BGR
            canvas = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
            canvas[:H_cropped, :W_cropped, :] = cropped_img
            return canvas

    # 如果是彩色图像，直接返回；如果是灰度 mask，返回灰度
    if is_mask:
        return cropped_img

    # 如果是彩色图像，但读取时是灰度图 (不应该发生)，则转为 BGR
    if cropped_img.ndim == 2:
        return cv2.cvtColor(cropped_img, cv2.COLOR_GRAY2BGR)

    return cropped_img


# --- 4. 主执行模块 ---

if __name__ == "__main__":

    # 0. 检查设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    SAVE_PATH = "fig_results/self-alignment"
    os.makedirs(SAVE_PATH, exist_ok=True)

    # --- TODO: 在此处替换为您自己的数据 ---
    ORIGIN_IMAGE_PATH = (
        "../data/segmentation/Turkey/Islahiye/pre/val/images/14848_4096.png"
    )
    LABEL_PATH = "../data/segmentation/Turkey/Islahiye/pre/val/labels/14848_4096.png"

    CROP_X = 748
    CROP_Y = 256

    # 裁剪窗口大小
    CROP_SIZE = 256

    # 1. 创建模拟数据 (一个裁剪后的图像和标签)
    image = load_and_crop(ORIGIN_IMAGE_PATH, CROP_X, CROP_Y, CROP_SIZE)
    mask = load_and_crop(LABEL_PATH, CROP_X, CROP_Y, CROP_SIZE, is_mask=True)

    image = (
        torch.from_numpy(image.astype(np.float32)).to(device).permute(2, 0, 1) / 255.0
    )
    mask = torch.from_numpy(mask.astype(np.float32)).to(device) / 255.0
    # # 假设我们有一个 100x100 的裁剪区域
    # H, W = 100, 100

    # # 创建一个模拟图像: 中间有一个白色的"建筑物"
    # image = torch.zeros((3, H, W), device=device)
    # # "真实" 建筑物的边界
    # image[:, 30:70, 30:70] = 1.0
    # # 添加一些渐变来帮助 DT Map
    # image[0, 30:70, 30:70] = torch.linspace(0.5, 1.0, 40).view(-1, 1)

    # # 创建一个模拟标签: 它偏离了 "真实" 建筑物
    # # 真实位置中心在 (50, 50), 标签中心在 (60, 60)
    # # 意味着它向右 (dx) 偏移了 10 像素, 向下 (dy) 偏移了 10 像素
    # # 我们的算法应该找到 (-10, -10) 附近的最优解
    # mask = torch.zeros((H, W), device=device)
    # mask[40:80, 40:80] = 1.0  # 偏移的标签

    # -------------------------------------

    # 2. 实例化您的优化器
    optimizer = InstanceWiseAlignmentOptimizer()

    # 3. 从您的代码中复制金字塔参数
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

    # 4. 定义搜索范围
    SEARCH_RANGE_PX = 40  # 搜索 +/- 25 像素
    STEP_PX = 1  # 每隔 1 像素计算一次损失

    # 5. 循环生成并绘制每个层级的损失图
    cmaps = ["cividis", "plasma", "viridis"]  # 为不同层级使用不同色表

    for i, level in enumerate(pyramid_levels):
        print(f"\n--- Generating landscape for level: {level['name']} ---")

        # 计算损失景观
        DX, DY, LOSS_MAP = generate_loss_landscape(
            optimizer,
            image,
            mask,
            level,
            search_range_px=SEARCH_RANGE_PX,
            step_px=STEP_PX,
        )

        # 绘制 3D 图像
        plot_title = f"{level['name']}"
        plot_loss_landscape(
            DX, DY, LOSS_MAP, title=plot_title, cmap="viridis", save_path=SAVE_PATH
        )

    print("\nDone.")
