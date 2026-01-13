import cv2
import numpy as np
import os
import shutil
from scipy import ndimage  # 仍然使用scipy的shift函数进行平移，因为它很方便

# --- 核心功能函数 ---


def calculate_miou_with_background(mask_pred, mask_gt):
    """计算两个二值掩码之间的 mIoU (前景 + 背景)"""
    pred_bool = mask_pred > 127
    gt_bool = mask_gt > 127
    intersection_fg = np.logical_and(pred_bool, gt_bool).sum()
    union_fg = np.logical_or(pred_bool, gt_bool).sum()
    iou_fg = intersection_fg / union_fg if union_fg != 0 else 0
    pred_bg_bool = ~pred_bool
    gt_bg_bool = ~gt_bool
    intersection_bg = np.logical_and(pred_bg_bool, gt_bg_bool).sum()
    union_bg = np.logical_or(pred_bg_bool, gt_bg_bool).sum()
    iou_bg = intersection_bg / union_bg if union_bg != 0 else 0
    return (iou_fg + iou_bg) / 2.0


def align_footprints_fast(image_path, label_path, gt_path, max_offset=40):
    """
    【根据您的建议优化】
    使用裁剪模板和cv2.matchTemplate高效地执行对齐。
    """
    # 1. 加载图像和标签
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    misaligned_label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    ground_truth = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

    if image is None or misaligned_label is None or ground_truth is None:
        return None, (0, 0)

    h, w = misaligned_label.shape
    if h <= 2 * max_offset or w <= 2 * max_offset:
        print(
            f"Warning: Image size of {os.path.basename(image_path)} is too small for the given max_offset. Skipping."
        )
        return None, (0, 0)

    # 2. 计算图像的梯度幅值图
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = cv2.normalize(
        cv2.magnitude(grad_x, grad_y), None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    # 3. 【核心优化】创建裁剪后的模板
    # 我们从偏移标签的中心区域裁剪出一个模板
    template = misaligned_label[
        max_offset : h - max_offset, max_offset : w - max_offset
    ]

    # 4. 使用 cv2.matchTemplate 在梯度图中寻找模板的最佳匹配位置
    # 搜索范围是整个梯度图
    result = cv2.matchTemplate(gradient_magnitude, template, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(result)

    # 5. 计算全局偏移量
    # max_loc 是模板在梯度图中的左上角匹配位置 (x, y)
    # 模板本身是从原标签 (max_offset, max_offset) 位置开始裁剪的
    # 因此，平移量 = 匹配位置 - 裁剪起始位置
    top_left_x, top_left_y = max_loc
    shift_x = top_left_x - max_offset
    shift_y = top_left_y - max_offset

    # 6. 应用偏移量
    corrected_label = ndimage.shift(
        misaligned_label, shift=[shift_y, shift_x], mode="constant", cval=0
    )

    # 7. 计算 mIoU
    miou = calculate_miou_with_background(corrected_label, ground_truth)

    return miou, (shift_x, shift_y)


# --- 主程序 ---


def main():
    # 数据目录
    BASE_DIR = "/root/workspace/zjj/xjd/data/segmentation/Turkey/Antakya/pre/test"
    IMAGE_DIR, LABEL_DIR, GT_DIR = [
        os.path.join(BASE_DIR, d) for d in ["images", "labels", "gt"]
    ]
    print("\nStarting alignment process...")

    image_files = [
        f for f in os.listdir(IMAGE_DIR) if f.endswith((".png", ".jpg", ".tif"))
    ]
    if not image_files:
        print("No images found in the directory.")
        return

    before_all_miou = []
    all_miou_scores = []
    for filename in image_files:
        paths = [os.path.join(d, filename) for d in [IMAGE_DIR, LABEL_DIR, GT_DIR]]
        if not all(os.path.exists(p) for p in paths):
            print(f"Skipping {filename}: corresponding file not found.")
            continue

        original_label = cv2.imread(paths[1], 0)
        gt_label = cv2.imread(paths[2], 0)
        # 计算原始未对齐的 mIoU
        original_miou = calculate_miou_with_background(original_label, gt_label)

        # 执行对齐并计算校正后的 mIoU
        corrected_miou, offset = align_footprints_fast(*paths, max_offset=40)

        if corrected_miou is not None:
            before_all_miou.append(original_miou)
            all_miou_scores.append(corrected_miou)
            print(f"File: {filename}")
            print(f"  - Found best offset at (dx, dy) = ({offset[0]}, {offset[1]})")
            print(f"  - mIoU before alignment: {original_miou:.4f}")
            print(f"  - mIoU after alignment:  {corrected_miou:.4f}\n")

    if all_miou_scores:
        before_average_miou = np.mean(before_all_miou)
        average_miou = np.mean(all_miou_scores)
        print("-----------------------------------------")
        print(f"Average mIoU before across all images: {before_average_miou:.4f}")
        print(f"Average mIoU across all images: {average_miou:.4f}")
        print("-----------------------------------------")


if __name__ == "__main__":
    main()
