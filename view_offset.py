

import os
import cv2
import numpy as np


def img_with_label_advanced(image, label, color_map=None, alpha=0.5):
    """
    支持多个标签值对应不同颜色
    
    参数:
        image: 原图像 (H, W, 3) 或 (H, W)
        label: 标签图像 (H, W)，可以包含多个标签值
        color_map: 颜色映射字典，如{1: [255, 0, 0], 2: [0, 255, 0]}
        alpha: 叠加透明度，0-1之间
    
    返回:
        img_lab: 叠加后的图像
    """
    if color_map is None:
        color_map = {1: [255, 0, 0]}
    
    h, w = label.shape
    color_label = np.zeros((h, w, 3), dtype=np.float32)
    
    # 为每个标签值设置对应颜色
    for label_value, color in color_map.items():
        mask = label == label_value
        color_label[mask] = color
    
    img_lab = cv2.addWeighted(image, 1 - alpha, color_label, alpha, 0)
    
    return img_lab


def save_by_cv2(path, image):
    cv2.imwrite(path, image)
    
def main():
    image_paths = "../data/segmentation/Turkey/Islahiye/pre/test/images"
    label_paths = "../data/segmentation/Turkey/Islahiye/pre/test/labels"
    gt_paths = "../data/segmentation/Turkey/Islahiye/pre/test/gt"
    for file_name in os.listdir(image_paths):
        if not file_name.endswith(".png"):
            continue
        image_name = file_name.split(".")[0]
        image_path = os.path.join(image_paths, file_name)
        label_path = os.path.join(label_paths, file_name)
        gt_path = os.path.join(gt_paths, file_name)

        image_bgr = cv2.imread(image_path)
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        label = cv2.cvtColor(
            cv2.imread(label_path),
            cv2.COLOR_BGR2GRAY,
        ) / 255.0
        gt = cv2.cvtColor(
            cv2.imread(gt_path),
            cv2.COLOR_BGR2GRAY,
        ) / 255.0
        
        label_gt = np.zeros_like(label)
        label_gt[label > 0.5] = 1
        label_gt[gt > 0.5] = 2
        
        img_lab_gt = img_with_label_advanced(
            image_bgr.astype(np.float32), label_gt.astype(np.float32), color_map={1: [0, 0, 255], 2: [0, 255, 0]}, alpha=0.5
        )
        save_by_cv2(
            os.path.join("vis_logs/offset_analysis", f"{image_name}_image+label+gt.png"), img_lab_gt
        )

if __name__ == "__main__":
    main()