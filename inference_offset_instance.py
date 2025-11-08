import argparse
from pathlib import Path
import glob
from PIL import Image
import cv2
import numpy as np
import torch
import albumentations as albu
from tools.cfg import py2cfg
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import os

from train_offset_instance import Supervision_Train

# 导入你的数据集 (它包含了 TeqDataset 和 collate_fn)
from geoseg.datasets.teq_instance_dataset import (
    TeqInstanceDataset,
    instance_collate_fn,
    get_val_transform,
)

# 导入你的可视化 (用于推理)
from tools.visual import (
    visualize_correction_overlay,
    visualize_emi_vs_pred_overlay,
    visualize_gt_vs_pred_overlay,
)


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # ... (保持 seed_everything)


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg(
        "-i",
        "--image_path",
        type=str,
        required=True,
        help="Path to test images folder",
    )
    arg(
        "-id",
        "--instance_path",
        type=str,
        required=False,
        help="Path to instance .pt files folder",
    )
    arg(
        "-m",
        "--mask_path",
        type=str,
        required=False,
        help="Path to shifted label masks folder",
    )
    arg(
        "-c",
        "--config_path",
        type=Path,
        required=True,
        help="Path to config (e.g., v6_deeplab.py)",
    )
    arg(
        "-o",
        "--output_path",
        type=Path,
        help="Path to save results.",
        required=True,
    )
    arg("-b", "--batch-size", help="batch size", type=int, default=4)
    return parser.parse_args()


def main():
    args = get_args()
    seed_everything(42)
    config = py2cfg(args.config_path)

    # 1. 创建输出目录
    output_npy_path = os.path.join(args.output_path, "npy")
    output_vis_path = os.path.join(args.output_path, "vis_png")
    os.makedirs(output_npy_path, exist_ok=True)
    os.makedirs(output_vis_path, exist_ok=True)

    # 2. 加载模型
    model = Supervision_Train.load_from_checkpoint(
        os.path.join(config.weights_path, config.test_weights_name + ".ckpt"),
        config=config,
        strict=False,
    )
    model.cuda()
    model.eval()

    # 3. 创建数据集和数据加载器 (使用 'test' 模式)
    # 注意: TeqDataset 的 'data_root' 是所有子目录的父目录
    # 假设 args.image_path 等是 data_root 下的子目录名
    data_root = os.path.abspath(os.path.join(args.image_path, ".."))
    img_dir = "images"
    instance_dir = "instances"
    mask_dir = "labels"
    gt_dir = "gt"

    dataset = TeqInstanceDataset(
        data_root=data_root,
        mode="test",
        img_dir=img_dir,
        instance_dir=instance_dir,
        mask_dir=mask_dir,
        gt_dir=gt_dir,
        transform=get_val_transform(),  # 使用验证集变换
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=False,
        pin_memory=True,
        collate_fn=instance_collate_fn,
    )

    # 4. 推理循环
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            if batch is None:
                continue

            img = batch["img"].cuda()
            gt_offsets_list = [b.cuda() for b in batch["gt_offsets"]]
            gt_confidences_list = [b.cuda() for b in batch["gt_confidences"]]
            bboxes_list = [b.cuda() for b in batch["bboxes"]]
            centroids_list = [c.cuda() for c in batch["centroids"]]
            mask = batch["mask"]
            gt_mask = batch["gt_mask"]
            img_ids = batch["img_id"]

            # 模型前向传播
            predicted_offsets = model(img, bboxes_list, centroids_list)  # [N_total, 2]

            # 将 [N_total, 2] 拆分回列表 [B, (N_i, 2)]
            num_instances_per_image = [len(b) for b in bboxes_list]
            pred_offsets_list = torch.split(
                predicted_offsets.detach(), num_instances_per_image, dim=0
            )

            # 5. 保存结果
            for i in range(len(img_ids)):
                img_id = img_ids[i]
                if len(bboxes_list[i]) == 0:
                    continue

                # 获取第 i 个样本的数据
                img_tensor_i = batch["img"][i]
                mask_i = mask[i]
                gt_mask_i = gt_mask[i]
                bboxes_i = bboxes_list[i]
                pred_offsets_i = pred_offsets_list[i]
                gt_offsets_i = gt_offsets_list[i]

                # 5.1. 保存 .npy 文件
                save_dict = {
                    "bboxes": bboxes_i.cpu().numpy(),
                    "pred_offsets_ratio": pred_offsets_i.cpu().numpy(),
                }
                np.save(os.path.join(output_npy_path, f"{img_id}.npy"), save_dict)

                # 5.2. 保存可视化 .png 文件
                visualize_correction_overlay(
                    image_tensor=img_tensor_i,
                    shifted_mask_tensor=mask_i,
                    bboxes_tensor=bboxes_i,
                    pred_offsets_ratio_tensor=pred_offsets_i,
                    save_path=output_vis_path,
                    file_name=f"{img_id}_pred_correction",
                )
                visualize_emi_vs_pred_overlay(
                    image_tensor=img_tensor_i,
                    shifted_mask_tensor=mask_i,
                    bboxes_tensor=bboxes_i,
                    pred_offsets_ratio_tensor=pred_offsets_i,
                    gt_offsets_ratio_tensor=gt_offsets_i,
                    save_path=output_vis_path,
                    file_name=f"{img_id}_emi_vs_pred",
                )
                visualize_gt_vs_pred_overlay(
                    image_tensor=img_tensor_i,
                    shifted_mask_tensor=mask_i,
                    gt_mask_tensor=gt_mask_i,
                    bboxes_tensor=bboxes_i,
                    pred_offsets_ratio_tensor=pred_offsets_i,
                    save_path=output_vis_path,
                    file_name=f"{img_id}_gt_vs_pred",
                )

    print(f"Inference complete. Results saved to {args.output_path}")


if __name__ == "__main__":
    main()
