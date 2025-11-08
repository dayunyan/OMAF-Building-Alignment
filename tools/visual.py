import os
from typing import Optional
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
from PIL import Image

import torchvision.transforms as transforms


def save_tensor_as_png(tensor, save_path, file_name):
    """
    Save a tensor as a PNG image.

    Args:
        tensor (torch.Tensor): The input tensor to save. Shape should be (C, H, W).
        save_path (str): Directory where the image will be saved.
        file_name (str): Name of the output PNG file.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Ensure the tensor is on CPU and detach it from the computation graph
    tensor = tensor.cpu().detach()

    # Normalize tensor to [0, 255] and convert to uint8
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min()) * 255
    tensor = tensor.byte()

    # Convert tensor to PIL image
    transform = transforms.ToPILImage()
    image = transform(tensor)

    # Save the image as PNG
    image.save(os.path.join(save_path, f"{file_name}.png"))


def save_mask_as_png(mask_array: np.ndarray, save_path: str):
    """保存 0/255 的二值 numpy 掩码为 PNG 文件"""
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    # 使用 PIL 确保以 8 位灰度模式保存 (cv2.imwrite 可能因环境问题保存成 BGR)
    # 数组形状应为 (H, W)，值应为 0 或 255
    img = Image.fromarray(mask_array, mode="L")
    img.save(save_path)


def visualize_masks(mask1, mask2, save_path, file_name):
    """
    可视化两个二值掩膜，用不同颜色表示重叠区域
    参数:
        mask1: torch.Tensor, 形状为[h, w], 值为0或1
        mask2: torch.Tensor, 形状为[h, w], 值为0或1
        output_path: str, 输出图像保存路径
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 确保输入是CPU上的张量
    if mask1.is_cuda:
        mask1 = mask1.cpu()
    if mask2.is_cuda:
        mask2 = mask2.cpu()

    # 转换为numpy数组
    mask1 = (mask1 - mask1.min()) / (mask1.max() - mask1.min())
    mask2 = (mask2 - mask2.min()) / (mask2.max() - mask2.min())
    mask1_np = mask1.numpy().astype(np.uint8)
    mask2_np = mask2.numpy().astype(np.uint8)

    # 创建RGB图像 (H, W, 3)
    h, w = mask1_np.shape
    result = np.zeros((h, w, 3), dtype=np.uint8)

    # 计算不同区域
    only_mask1 = (mask1_np == 1) & (mask2_np == 0)  # 只存在于第一个掩膜
    only_mask2 = (mask1_np == 0) & (mask2_np == 1)  # 只存在于第二个掩膜
    overlap = (mask1_np == 1) & (mask2_np == 1)  # 两者重叠区域

    # 分配颜色 (R, G, B)
    result[only_mask1] = [255, 0, 0]  # 红色：仅第一个掩膜
    result[only_mask2] = [0, 255, 0]  # 绿色：仅第二个掩膜
    result[overlap] = [255, 255, 0]  # 黄色：重叠区域

    # 保存图像
    img = Image.fromarray(result)
    img.save(os.path.join(save_path, f"{file_name}.png"))

    return img  # 可选：返回PIL图像对象


def visualize_aligned_mask(
    image,
    prediction,
    align_conf,
    mean: Optional[np.ndarray] = np.array([0.485, 0.456, 0.406]),
    std: Optional[np.ndarray] = np.array([0.229, 0.224, 0.225]),
    save_path: Optional[str] = None,
    cmap: str = "jet",  # 置信度渐变配色（可换 viridis/plasma）
):
    """
    在原图上叠加红色半透明预测前景和置信度渐变图
    Args:
        image: 标准化后的图像，shape=(C,H,W)或(H,W,C)，Tensor/numpy
        prediction: 预测结果（0=背景，1=前景），shape=(H,W)，Tensor/numpy
        align_conf: 置信度图（0-1），shape=(H,W)，Tensor/numpy
        mean: 标准化时使用的均值，shape=(3,)，默认 ImageNet 均值 [0.485, 0.456, 0.406]
        std: 标准化时使用的标准差，shape=(3,)，默认 ImageNet 标准差 [0.229, 0.224, 0.225]
        save_path: 图像保存路径（None 则直接显示）
        cmap: 置信度渐变配色方案
    """

    def to_numpy(tensor) -> np.ndarray:
        """将 Tensor 转为 numpy 数组，处理梯度和设备"""
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return tensor

    # 转换所有输入为 numpy
    image_np = to_numpy(image)
    pred_np = to_numpy(prediction)
    conf_np = to_numpy(align_conf)

    if image_np.ndim == 3:
        if image_np.shape[0] == 3:  # (C,H,W) → (H,W,C)
            image_np = np.transpose(image_np, (1, 2, 0))
    else:
        raise ValueError(f"图像维度错误，应为 3 维，实际为 {image_np.ndim} 维")

    image_np = image_np * std + mean
    image_np = np.clip(image_np, 0.0, 1.0)
    image_uint8 = (image_np * 255).astype(np.uint8)

    # 确保预测图为 (H,W) 二值图
    pred_np = pred_np.squeeze()
    pred_mask = (pred_np > 0.5).astype(np.uint8)

    red_mask = np.zeros_like(image_uint8)
    red_mask[pred_mask == 1, 2] = 255  # 红色通道（BGR 中索引 2 是红色）

    # 红色蒙版半透明叠加（alpha=0.5）
    image_with_pred = cv2.addWeighted(image_uint8, 1.0, red_mask, 0.5, 0)

    conf_np = np.clip(conf_np.squeeze(), 0.0, 1.0)

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # 绘制原图 + 红色预测蒙版
    ax.imshow(image_np)
    ax.imshow(red_mask)  # 叠加红色半透明预测

    # 绘制置信度渐变（半透明叠加）
    conf_im = ax.imshow(conf_np, cmap=cmap, alpha=0.6)  # alpha 控制透明度

    # 设置画布样式
    ax.axis("off")
    ax.set_title(
        "Image + Red Prediction (α=0.5) + Confidence Gradient", fontsize=14, pad=20
    )

    cbar = plt.colorbar(conf_im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Alignment Confidence (0-1)", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()
    # bbox_inches='tight'：自动裁剪空白区域，保证颜色条不被截断
    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight",
        pad_inches=0.2,  # 边缘留白，避免内容贴边
    )

    plt.close(fig)


def visualize_grayscale_as_pseudocolor(
    tensor, save_path, file_name, cmap="viridis", vmin=None, vmax=None
):
    """
    将灰度张量转换为伪彩色图像并保存

    参数:
        tensor: torch.Tensor, 形状为[h, w]的灰度张量
        output_path: str, 输出图像保存路径
        cmap: str或Colormap对象, 使用的颜色映射 (默认为'viridis')
        vmin: float, 颜色映射的最小值 (自动计算如果为None)
        vmax: float, 颜色映射的最大值 (自动计算如果为None)

    返回:
        PIL.Image对象
    """
    # 确保输入是CPU上的张量
    if tensor.is_cuda:
        tensor = tensor.cpu()

    # 转换为numpy数组
    data = tensor.numpy()

    # 获取数据范围
    if vmin is None:
        vmin = np.min(data)
    if vmax is None:
        vmax = np.max(data)

    # 创建图形
    plt.figure(figsize=(10, 8))

    # 绘制伪彩色图像
    im = plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)

    # 添加颜色条
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label("Value")

    # 设置标题
    plt.title("Pseudocolor Visualization")

    # 保存图像
    plt.savefig(
        os.path.join(save_path, f"{file_name}.png"), bbox_inches="tight", dpi=150
    )
    plt.close()


def _unnormalize_image(tensor_image):
    """将标准化的Tensor图像反归一化为 (H,W,C) numpy 数组 (0-255)"""
    # 假设的均值和标准差 (与 teq_dataset.py 中一致)
    mean = np.array([0.508, 0.458, 0.430])
    std = np.array([0.194, 0.172, 0.158])

    img_np = tensor_image.cpu().permute(1, 2, 0).numpy()
    img_np = (img_np * std + mean) * 255.0
    img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    return img_np


def _apply_offsets_to_mask(shifted_mask, bboxes, offsets_px):
    """
    (内部函数) 将每个实例的偏移应用到其在语义掩码中的像素上。

    Args:
        shifted_mask (np.ndarray): (H, W) 二值掩码，包含所有 *未修正* 的实例。
        bboxes (np.ndarray): (N, 4) 实例的 BBox [x1, y1, x2, y2]。
        offsets_px (np.ndarray): (N, 2) 每个实例的像素偏移 [dx, dy]。

    Returns:
        np.ndarray: (H, W) 修正后的二值掩码。
    """
    H, W = shifted_mask.shape
    corrected_mask = np.zeros_like(shifted_mask, dtype=np.uint8)

    # 1. 创建一个实例ID掩码，以便我们知道哪个像素属于哪个BBox
    # (注意：BBox 重叠时，ID 较大的会覆盖较小的)
    instance_id_mask = np.zeros_like(shifted_mask, dtype=np.int32)
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        # 仅在BBox内的掩码像素上分配实例ID
        bbox_mask_region = shifted_mask[y1:y2, x1:x2] > 0
        instance_id_mask[y1:y2, x1:x2][bbox_mask_region] = i + 1

    # 2. 逐个实例应用偏移
    for i in range(len(bboxes)):
        instance_id = i + 1
        dx, dy = offsets_px[i]

        # 获取该实例的纯二值掩码
        instance_pixels = (instance_id_mask == instance_id).astype(np.uint8)
        if instance_pixels.sum() == 0:
            continue

        # 3. 创建平移矩阵并应用
        # M = [[1, 0, dx], [0, 1, dy]]
        M = np.float32([[1, 0, dx], [0, 1, dy]])

        # 使用 warpAffine 对该实例的像素进行平移
        shifted_instance_pixels = cv2.warpAffine(instance_pixels, M, (W, H))

        # 将平移后的像素粘贴到最终的修正掩码上
        corrected_mask[shifted_instance_pixels > 0] = 1

    return corrected_mask


def visualize_correction_overlay(
    image_tensor,
    shifted_mask_tensor,
    bboxes_tensor,
    pred_offsets_ratio_tensor,
    save_path,
    file_name,
    alpha=0.5,
):
    """
    可视化 1: 在原图上叠加 偏移的掩码(红色) 和 预测修正后的掩码(绿色)。

    Args:
        image_tensor (torch.Tensor): (C, H, W) 标准化的图像。
        shifted_mask_tensor (torch.Tensor): (H, W) 带偏移的二值掩码。
        bboxes_tensor (torch.Tensor): (N, 4) 实例的 BBox。
        pred_offsets_ratio_tensor (torch.Tensor): (N, 2) 预测的偏移比率。
        save_path (str): 保存目录。
        file_name (str): 保存文件名 (不含.png)。
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 1. 转换数据
    img_rgb = _unnormalize_image(image_tensor)
    shifted_mask = shifted_mask_tensor.cpu().numpy().astype(np.uint8)
    bboxes = bboxes_tensor.cpu().numpy()
    pred_offsets_ratio = pred_offsets_ratio_tensor.cpu().numpy()

    H, W = shifted_mask.shape
    input_size = np.array([W, H], dtype=np.float32)
    pred_offsets_px = pred_offsets_ratio * input_size

    # 2. 计算修正后的掩码
    pred_corrected_mask = _apply_offsets_to_mask(shifted_mask, bboxes, pred_offsets_px)

    # 3. 创建可视化
    # (H, W, 3)
    overlay = img_rgb.copy()

    # 蓝色: 偏移的掩码
    overlay[shifted_mask == 1] = (
        overlay[shifted_mask == 1] * (1 - alpha) + np.array([0, 0, 255]) * alpha
    )
    # 红色: 预测修正后的掩码
    overlay[pred_corrected_mask == 1] = (
        overlay[pred_corrected_mask == 1] * (1 - alpha) + np.array([255, 0, 0]) * alpha
    )
    # 重叠区域 (R+G=Yellow)

    # 保存图像
    img = Image.fromarray(overlay.astype(np.uint8))
    img.save(os.path.join(save_path, f"{file_name}.png"))


def visualize_emi_vs_pred_overlay(
    image_tensor,
    shifted_mask_tensor,
    bboxes_tensor,
    pred_offsets_ratio_tensor,
    gt_offsets_ratio_tensor,
    save_path,
    file_name,
    alpha=0.5,
):
    """
    可视化 2: 在原图上叠加 预测修正的掩码(蓝色) 和 GT修正的掩码(绿色)。

    Args:
        (参数同上, 增加了 gt_offsets_ratio_tensor)
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 1. 转换数据
    img_rgb = _unnormalize_image(image_tensor)
    shifted_mask = shifted_mask_tensor.cpu().numpy().astype(np.uint8)
    bboxes = bboxes_tensor.cpu().numpy()
    pred_offsets_ratio = pred_offsets_ratio_tensor.cpu().numpy()
    gt_offsets_ratio = gt_offsets_ratio_tensor.cpu().numpy()

    H, W = shifted_mask.shape
    input_size = np.array([W, H], dtype=np.float32)

    # 2. 计算两个修正掩码
    pred_offsets_px = pred_offsets_ratio * input_size
    pred_corrected_mask = _apply_offsets_to_mask(shifted_mask, bboxes, pred_offsets_px)

    gt_offsets_px = gt_offsets_ratio * input_size
    gt_corrected_mask = _apply_offsets_to_mask(shifted_mask, bboxes, gt_offsets_px)

    # 3. 创建可视化
    overlay = img_rgb.copy()

    # 绿色: 预测修正
    overlay[pred_corrected_mask == 1] = (
        overlay[pred_corrected_mask == 1] * (1 - alpha) + np.array([255, 0, 0]) * alpha
    )
    # 蓝色: GT修正
    overlay[gt_corrected_mask == 1] = (
        overlay[gt_corrected_mask == 1] * (1 - alpha) + np.array([0, 0, 255]) * alpha
    )
    # 重叠区域 (B+G=Cyan)

    # 保存图像
    img = Image.fromarray(overlay.astype(np.uint8))
    img.save(os.path.join(save_path, f"{file_name}.png"))


def visualize_gt_vs_pred_overlay(
    image_tensor,
    shifted_mask_tensor,
    gt_mask_tensor,
    bboxes_tensor,
    pred_offsets_ratio_tensor,
    save_path,
    file_name,
    alpha=0.5,
):
    """
    可视化 1: 在原图上叠加 准确的掩码(绿色) 和 预测修正后的掩码(红色)。

    Args:
        image_tensor (torch.Tensor): (C, H, W) 标准化的图像。
        shifted_mask_tensor (torch.Tensor): (H, W) 带偏移的二值掩码。
        gt_mask_tensor (torch.Tensor): (H, W) 准确的二值掩码。
        bboxes_tensor (torch.Tensor): (N, 4) 实例的 BBox。
        pred_offsets_ratio_tensor (torch.Tensor): (N, 2) 预测的偏移比率。
        save_path (str): 保存目录。
        file_name (str): 保存文件名 (不含.png)。
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 1. 转换数据
    img_rgb = _unnormalize_image(image_tensor)
    shifted_mask = shifted_mask_tensor.cpu().numpy().astype(np.uint8)
    gt_mask_tensor = gt_mask_tensor.cpu().numpy().astype(np.uint8)
    bboxes = bboxes_tensor.cpu().numpy()
    pred_offsets_ratio = pred_offsets_ratio_tensor.cpu().numpy()

    H, W = shifted_mask.shape
    input_size = np.array([W, H], dtype=np.float32)
    pred_offsets_px = pred_offsets_ratio * input_size

    # 2. 计算修正后的掩码
    pred_corrected_mask = _apply_offsets_to_mask(shifted_mask, bboxes, pred_offsets_px)

    # 3. 创建可视化
    # (H, W, 3)
    overlay = img_rgb.copy()

    # 红色: 预测修正后的掩码
    overlay[pred_corrected_mask == 1] = (
        overlay[pred_corrected_mask == 1] * (1 - alpha) + np.array([255, 0, 0]) * alpha
    )
    # 绿色: 准确的掩码
    overlay[gt_mask_tensor == 1] = (
        overlay[gt_mask_tensor == 1] * (1 - alpha) + np.array([0, 255, 0]) * alpha
    )
    # 重叠区域 (R+G=Yellow)

    # 保存图像
    img = Image.fromarray(overlay.astype(np.uint8))
    img.save(os.path.join(save_path, f"{file_name}.png"))
