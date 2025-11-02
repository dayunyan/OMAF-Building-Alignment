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
