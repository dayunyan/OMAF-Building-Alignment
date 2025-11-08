import os
import torch
import numpy as np
import cv2
from tqdm import tqdm

from tools.alignment import InstanceWiseAlignmentOptimizer
from offset_instance_check import DetailedPTChecker


def build_seg_dataset_with_instance_conf(root, check=False):
    if not os.path.exists(root):
        return

    if check:
        checker = DetailedPTChecker(max_bbox_aspect_ratio=500)

    for dir_name in tqdm(os.listdir(root), position=1, desc=f"Processing directory"):
        images_dir = os.path.join(root, dir_name, "images")
        labels_dir = os.path.join(root, dir_name, "labels")
        save_dir = os.path.join(root, dir_name, "instances")
        os.makedirs(save_dir, exist_ok=True)
        if check:
            results = checker.check_directory(save_dir, fix_issues=True)
            checker.generate_report(results)

            if checker.stats["invalid_files"] > 0:
                print(f"警告: {dir_name} 目录中有无效文件！")
                print(f"发现 {checker.stats['invalid_bboxes']} 个不合法的bbox")
        else:
            main(images_dir, labels_dir, save_dir)


def main(image_paths, label_paths, save_paths):
    gaussian_mu = [8.46682349428956, 22.00696511111716]  # 均值偏移
    gaussian_sigma = [8.84103344498332, 11.79890553008147]  # 标准差
    covariance = [
        78.16387237531364,
        20.141596273248012,
        20.141596273248012,
        139.2141717077871,
    ]
    for file_name in tqdm(os.listdir(image_paths), position=0, desc=f"Processing file"):
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

        device = "cuda:1" if torch.cuda.is_available() else "cpu"

        rgb_image = torch.tensor(image).permute(2, 0, 1).to(device) / 255.0
        label_image = torch.tensor(label, dtype=torch.float32).to(device)

        # 创建优化器
        aligner = InstanceWiseAlignmentOptimizer()

        outputs = aligner.align_instance_with_multi_start_output_dict(
            rgb_image,
            label_image,
            gaussian_mu,
            gaussian_sigma,
            covariance,
            search_range_norm=0.01,
            num_candidates=8,
            nms_radius_norm=0.001,
        )
        torch.save(outputs, os.path.join(save_paths, f"{image_name}.pt"))


if __name__ == "__main__":
    # main()
    build_seg_dataset_with_instance_conf(
        root="../data/segmentation/Turkey/Islahiye/pre", check=True
    )
