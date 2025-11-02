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
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from typing import List, Tuple, Optional

from tools.alignment import InstanceWiseAlignmentOptimizer


SAVE_FIG_DIR = "./vis_logs/aligner_instance_confidence/"
os.makedirs(SAVE_FIG_DIR, exist_ok=True)


def build_seg_dataset_with_align_conf(root):
    if not os.path.exists(root):
        return

    for dir_name in tqdm(os.listdir(root), position=1, desc=f"Processing directory"):
        images_dir = os.path.join(root, dir_name, "images")
        labels_dir = os.path.join(root, dir_name, "labels")
        save_dir = os.path.join(root, dir_name, "align")
        os.makedirs(save_dir, exist_ok=True)
        worker(images_dir, labels_dir, save_dir)


def worker(image_paths, label_paths, save_paths):
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
        label_image = torch.tensor(label, dtype=torch.float32).to(device) / 255.0

        # 创建优化器
        aligner = InstanceWiseAlignmentOptimizer()

        aligned_labels_tensor, disp_field, confidence_map = (
            aligner.align_instance_with_multi_start(
                rgb_image,
                label_image,
                gaussian_mu,
                gaussian_sigma,
                covariance,
                search_range_norm=0.01,
                num_candidates=8,
                nms_radius_norm=0.001,
            )
        )
        torch.save(confidence_map, os.path.join(save_paths, f"{image_name}.pt"))


def main():
    image_paths = "../data/segmentation/Turkey/Islahiye/pre/test/images"
    label_paths = "../data/segmentation/Turkey/Islahiye/pre/test/labels"

    gaussian_mu = [8.46682349428956, 22.00696511111716]  # 均值偏移
    gaussian_sigma = [8.84103344498332, 11.79890553008147]  # 标准差
    covariance = [
        78.16387237531364,
        20.141596273248012,
        20.141596273248012,
        139.2141717077871,
    ]
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
        H, W = image.shape[0], image.shape[1]
        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            label = cv2.resize(label, (H, W), interpolation=cv2.INTER_NEAREST)
            print("label.shape: ", label.shape)

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        rgb_image = torch.tensor(image).permute(2, 0, 1).to(device) / 255.0
        label_image = torch.tensor(label, dtype=torch.float32).to(device) / 255.0

        # 创建优化器
        aligner = InstanceWiseAlignmentOptimizer()

        print("开始对齐所有实例...")
        aligned_labels_tensor, disp_field, confidence_map = (
            aligner.align_instance_with_multi_start(
                rgb_image,
                label_image,
                gaussian_mu,
                gaussian_sigma,
                covariance,
                search_range_norm=0.01,
                num_candidates=8,
                nms_radius_norm=0.001,
            )
        )
        print("所有实例对齐完成。")
        aligner.visualize_alignment_with_confidence(
            rgb_image,
            aligned_labels_tensor,
            confidence_map,
            original_label=label_image,
            save_path=os.path.join(SAVE_FIG_DIR, f"{image_name}_aligned.png"),
        )


if __name__ == "__main__":
    # main()
    build_seg_dataset_with_align_conf(root="../data/segmentation/Turkey/Islahiye/pre")
