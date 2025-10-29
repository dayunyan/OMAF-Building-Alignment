import os
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
