import os
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, List


def binary_mask(mask_np: np.ndarray) -> np.ndarray:
    """将灰度图转换为二值掩码（0=背景，1=前景）"""
    return (mask_np > 0).astype(np.float32)


def calculate_miou(gt: np.ndarray, pred: np.ndarray, num_classes: int = 2):
    """
    计算 Mean Intersection over Union (mIoU)

    Args:
        gt (np.ndarray): 展平后的真实标签 (0, 1)。
        pred (np.ndarray): 展平后的预测结果 (0, 1)。
        num_classes (int): 类别数 (默认为2: 背景/前景)。

    Returns:
        float: mIoU 值。
    """
    # 确保输入是整数类型
    gt = gt.flatten().astype(np.int64)
    pred = pred.flatten().astype(np.int64)

    # 计算混淆矩阵 M (num_classes, num_classes)
    # M[i, j] 表示真实类别 i 被预测为类别 j 的像素数量
    intersection = pred[gt == pred]
    area_intersection = np.histogram(intersection, bins=np.arange(num_classes + 1))[0]
    area_pred = np.histogram(pred, bins=np.arange(num_classes + 1))[0]
    area_gt = np.histogram(gt, bins=np.arange(num_classes + 1))[0]

    # IoU = I / (A_pred + A_gt - I)
    area_union = area_pred + area_gt - area_intersection

    # 避免除以零
    iou_per_class = np.zeros(num_classes, dtype=np.float64)
    valid_classes = area_union > 0
    iou_per_class[valid_classes] = (
        area_intersection[valid_classes] / area_union[valid_classes]
    )

    # mIoU 是有效类别的平均 IoU
    return np.mean(iou_per_class[valid_classes])


def calculate_soft_miou(gt: np.ndarray, pred_soft: np.ndarray, num_classes: int = 2):
    """
    计算 soft-mIoU（基于软权重，保持与 compute_miou 一致的接口和流程）

    Args:
        gt (np.ndarray): 展平后的真实标签 (0, 1)（二值硬标签）。
        pred_soft (np.ndarray): 展平后的预测权重 (0-1 小数，前景置信度)。
        num_classes (int): 类别数 (默认为2: 背景/前景)。

    Returns:
        float: soft-mIoU 值。
    """
    gt = gt.flatten()
    pred_soft = pred_soft.flatten()
    # 确保输入格式正确（展平数组）
    assert gt.ndim == 1, f"gt 需为展平数组，当前维度：{gt.ndim}"
    assert pred_soft.ndim == 1, f"pred_soft 需为展平数组，当前维度：{pred_soft.ndim}"
    assert len(gt) == len(
        pred_soft
    ), f"gt 和 pred_soft 长度不一致：{len(gt)} vs {len(pred_soft)}"

    # 确保 gt 是二值硬标签（0/1），pred_soft 是 0-1 权重
    gt = gt.astype(np.float64)
    pred_soft = np.clip(pred_soft.astype(np.float64), 0.0, 1.0)

    # 初始化每类的软交并集
    area_intersection = np.zeros(num_classes, dtype=np.float64)
    area_gt = np.zeros(num_classes, dtype=np.float64)
    area_pred_soft = np.zeros(num_classes, dtype=np.float64)

    for cls in range(num_classes):
        # 类别 cls 的真实掩码（硬标签）
        gt_cls = (gt == cls).astype(np.float64)

        if cls == 1:  # 前景类：权重为 pred_soft
            pred_soft_cls = pred_soft
        else:  # 背景类：权重为 1 - pred_soft（前景置信度的补集）
            pred_soft_cls = 1.0 - pred_soft

        # 软交集：sum(gt_cls * pred_soft_cls)
        area_intersection[cls] = np.sum(gt_cls * pred_soft_cls)

        # 真实类别 cls 的总权重（硬标签总和，即像素数）
        area_gt[cls] = np.sum(gt_cls)

        # 预测类别 cls 的总权重（软权重总和）
        area_pred_soft[cls] = np.sum(pred_soft_cls)

    # 软并集：A_pred + A_gt - A_intersection
    area_union = area_pred_soft + area_gt - area_intersection

    # 避免除以零，计算每类 soft-IoU
    iou_per_class = np.zeros(num_classes, dtype=np.float64)
    valid_classes = area_union > 1e-8  # 允许微小数值误差
    iou_per_class[valid_classes] = (
        area_intersection[valid_classes] / area_union[valid_classes]
    )

    # soft-mIoU 是有效类别的平均 soft-IoU（与原函数逻辑一致）
    return np.mean(iou_per_class[valid_classes])


def normalize_soft_miou(soft_miou: float, hard_miou: float) -> float:
    """
    归一化 soft-mIoU（软-硬 IoU 映射）

    Args:
        soft_miou: 原始 soft-mIoU 值
        hard_miou: 同一样本的硬标签 mIoU（参考基准）

    Returns:
        float: 归一化后的 soft-mIoU（0-1 范围，与硬标签 mIoU 可比）
    """
    if hard_miou < 1e-8:  # 无前景时，归一化为 1.0（与硬mIoU一致）
        return 1.0
    # 映射到硬mIoU的量级，钳位避免超界
    normalized = soft_miou / hard_miou
    return np.clip(normalized, 0.0, 1.0) / 1.3


def generate_moved_correct_label(correct_label_binary: np.ndarray) -> np.ndarray:
    """生成偏移后的correct_label（复用之前的质心随机反向平移逻辑）"""
    H, W = correct_label_binary.shape
    moved_label = np.zeros_like(correct_label_binary, dtype=np.float32)

    # 寻找所有连通组件（实例）
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (correct_label_binary * 255).astype(np.uint8), connectivity=8
    )

    # 遍历每个实例（跳过背景0）
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        centroid_x, centroid_y = centroids[i]

        # 计算BBox中心
        bbox_center_x = x + w / 2.0
        bbox_center_y = y + h / 2.0

        # 计算质心相对于BBox中心的偏移向量
        offset_x = centroid_x - bbox_center_x
        offset_y = centroid_y - bbox_center_y
        offset_distance = np.sqrt(offset_x**2 + offset_y**2)

        # 随机反向平移距离（0到offset_distance之间）
        random_dist = np.random.uniform(0, offset_distance)

        # 计算平移向量（反向归一化）
        if offset_distance > 1e-6:
            tx = -offset_x / offset_distance * random_dist
            ty = -offset_y / offset_distance * random_dist
        else:
            tx, ty = 0, 0

        # 平移矩阵
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        # 提取当前实例掩码
        instance_mask = (labels == i).astype(np.float32)
        # 应用平移
        translated_instance = cv2.warpAffine(
            instance_mask, M, (W, H), flags=cv2.INTER_NEAREST, borderValue=0
        )
        # 合并到移动后标签（取最大值避免重叠区域重复计算）
        moved_label = np.maximum(moved_label, translated_instance)

    return moved_label


def main():
    # ---------------------- 路径配置（与之前保持一致）----------------------
    image_paths = "../data/segmentation/Turkey/Antakya/pre/test/images"
    align_paths = "../data/segmentation/Turkey/Antakya/pre/test/align"
    gt_paths = "../data/segmentation/Turkey/Antakya/pre/test/gt"
    corrected_label_paths = "../data/segmentation/Turkey/Antakya/pre/test/pred_offsets"
    offset_label_paths = "../data/segmentation/Turkey/Antakya/pre/test/labels"

    # ---------------------- 初始化指标累加器 ----------------------
    metrics_accumulator = {
        "gt_offset_miou": 0.0,
        "gt_align_miou": 0.0,
        "gt_align_soft_miou": 0.0,
        "gt_moved_correct_miou": 0.0,
        "gt_correct_miou": 0.0,
    }
    valid_sample_count = 0  # 有效样本数（排除读取失败的）

    # ---------------------- 获取并校验文件 ----------------------
    # 按图像文件名排序，确保所有文件一一对应
    image_files = sorted(
        [
            f
            for f in Path(image_paths).glob("*.*")
            if f.suffix in [".png", ".jpg", ".jpeg"]
        ]
    )
    align_files = sorted([f for f in Path(align_paths).glob("*.pt") if f.is_file()])
    gt_files = sorted([f for f in Path(gt_paths).glob("*.png") if f.is_file()])
    corrected_label_files = sorted(
        [f for f in Path(corrected_label_paths).glob("*.png") if f.is_file()]
    )
    offset_label_files = sorted(
        [f for f in Path(offset_label_paths).glob("*.png") if f.is_file()]
    )

    # 校验文件数量一致
    file_groups = [
        image_files,
        align_files,
        gt_files,
        corrected_label_files,
        offset_label_files,
    ]
    file_names = ["图像", "align", "GT", "correct_label", "offset_label"]
    assert all(
        len(group) == len(image_files) for group in file_groups
    ), f"文件数不匹配：{[f'{name}({len(group)})' for name, group in zip(file_names, file_groups)]}"
    total_samples = len(image_files)
    print(f"共找到 {total_samples} 组文件，开始计算指标...")

    # ---------------------- 遍历样本计算指标 ----------------------
    for idx in range(total_samples):
        try:
            # 获取当前样本的所有文件
            img_file = image_files[idx]
            align_file = align_files[idx]
            gt_file = gt_files[idx]
            correct_file = corrected_label_files[idx]
            offset_file = offset_label_files[idx]

            # 校验文件名一致
            stems = [
                img_file.stem,
                align_file.stem,
                gt_file.stem,
                correct_file.stem,
                offset_file.stem,
            ]
            assert all(
                s == stems[0] for s in stems
            ), f"第{idx+1}组文件名校验失败：{[f.name for f in [img_file, align_file, gt_file, correct_file, offset_file]]}"

            # ---------------------- 读取并预处理所有标签 ----------------------
            # 1. 读取GT（基准标签，尺寸作为参考）
            gt_np = cv2.imread(str(gt_file), cv2.IMREAD_GRAYSCALE)
            if gt_np is None:
                raise ValueError(f"GT文件读取失败：{gt_file.name}")
            gt_mask = binary_mask(gt_np)  # 转二值（0/1）
            H, W = gt_mask.shape  # 基准尺寸

            # 2. 读取offset_label并预处理
            offset_np = cv2.imread(str(offset_file), cv2.IMREAD_GRAYSCALE)
            if offset_np is None:
                raise ValueError(f"offset_label文件读取失败：{offset_file.name}")
            offset_np = cv2.resize(offset_np, (W, H), interpolation=cv2.INTER_NEAREST)
            offset_mask = binary_mask(offset_np)

            # 3. 读取align并预处理
            align_tensor = torch.load(open(align_file, "rb"), map_location="cpu")
            align_np = align_tensor.numpy().astype(np.float32)
            align_np = cv2.resize(align_np, (W, H), interpolation=cv2.INTER_NEAREST)
            # 普通mIoU用：>0设为1
            align_hard_mask = binary_mask(align_np)
            # soft-mIoU用：保留0-1原值（确保已归一化）
            align_soft = np.clip(align_np, 0.0, 1.0)  # 安全钳位，避免异常值

            # 4. 读取correct_label并预处理
            correct_np = cv2.imread(str(correct_file), cv2.IMREAD_GRAYSCALE)
            if correct_np is None:
                raise ValueError(f"correct_label文件读取失败：{correct_file.name}")
            correct_np = cv2.resize(correct_np, (W, H), interpolation=cv2.INTER_NEAREST)

            correct_mask = binary_mask(correct_np)

            # 5. 生成偏移后的correct_label
            moved_correct_mask = generate_moved_correct_label(correct_mask)

            # ---------------------- 计算当前样本的5个指标 ----------------------
            miou1 = calculate_miou(gt_mask, offset_mask)
            miou2 = calculate_miou(gt_mask, align_hard_mask)
            soft_miou_raw = calculate_soft_miou(gt_mask, align_soft)
            soft_miou_norm = normalize_soft_miou(soft_miou_raw, miou2)
            miou4 = calculate_miou(gt_mask, moved_correct_mask)
            miou5 = calculate_miou(gt_mask, correct_mask)

            # if miou5 < 0.66:
            #     continue

            # ---------------------- 累加指标 ----------------------
            metrics_accumulator["gt_offset_miou"] += miou1
            metrics_accumulator["gt_align_miou"] += miou2
            metrics_accumulator["gt_align_soft_miou"] += soft_miou_norm
            metrics_accumulator["gt_moved_correct_miou"] += miou4
            metrics_accumulator["gt_correct_miou"] += miou5
            valid_sample_count += 1

            print(f"[{idx+1}/{total_samples}] 样本 {img_file.stem} 指标计算完成：")
            print(f"  1. gt-offset mIoU: {miou1:.4f}")
            print(f"  2. gt-align mIoU: {miou2:.4f}")
            print(f"  3. gt-align soft-mIoU: {soft_miou_norm:.4f}")
            print(f"  4. gt-moved_correct mIoU: {miou4:.4f}")
            print(f"  5. gt-correct mIoU: {miou5:.4f}")

        except Exception as e:
            print(f"[{idx+1}/{total_samples}] 样本处理失败：{str(e)}，跳过该样本")
            continue

    # ---------------------- 计算平均指标并输出 ----------------------
    print("\n" + "=" * 50)
    print(f"有效样本数：{valid_sample_count}/{total_samples}")
    print("所有样本平均指标：")
    print("=" * 50)
    for metric_name, total_value in metrics_accumulator.items():
        avg_value = total_value / valid_sample_count if valid_sample_count > 0 else 0.0
        # 格式化输出指标名称和数值
        if metric_name == "gt_offset_miou":
            print(f"1. gt与offset_label的mIoU：{avg_value:.4f}")
        elif metric_name == "gt_align_miou":
            print(f"2. gt与align_label（硬阈值）的mIoU：{avg_value:.4f}")
        elif metric_name == "gt_align_soft_miou":
            print(f"3. gt与align_label的soft-mIoU：{avg_value:.4f}")
        elif metric_name == "gt_moved_correct_miou":
            print(f"4. gt与偏移后correct_label的mIoU：{avg_value:.4f}")
        elif metric_name == "gt_correct_miou":
            print(f"5. gt与correct_label的mIoU：{avg_value:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
