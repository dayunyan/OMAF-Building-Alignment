import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
from PIL import Image
from pathlib import Path
import random

from train_offset_instance import Supervision_Train
from geoseg.datasets.teq_instance_dataset import (
    TeqInstanceDataset,
    instance_collate_fn,
    get_val_transform,
    ORIGIN_IMG_SIZE,
)
from tools.cfg import py2cfg
from tools.offset import apply_offsets_to_mask
from tools.visual import save_mask_as_png


def seed_everything(seed):
    """设置随机种子，确保结果可复现"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Generate corrected binary labels for all dataset splits."
    )
    arg = parser.add_argument
    arg(
        "-c",
        "--config_path",
        type=Path,
        help="Path to the model config file (e.g., v6_deeplab.py).",
        required=True,
    )
    arg(
        "-d",
        "--data_root",
        type=str,
        help="Root directory containing train/val/test folders (e.g., .../pre).",
        required=True,
    )
    return parser.parse_args()


def generate_labels_for_split(
    model: Supervision_Train, config, split_data_root: str, mode: str
):

    print(f"\n--- 正在处理 {mode.upper()} 分割 ---")

    # 1. 路径设置
    # split_data_root 应为 train/val/test 文件夹的路径
    img_dir = "images"
    instance_dir = "instances"
    mask_dir = "labels"  # 包含原始偏移标签的文件夹
    output_dir = "pred_offsets"  # 修正后的输出文件夹

    full_output_path = os.path.join(split_data_root, output_dir)
    os.makedirs(full_output_path, exist_ok=True)

    # 2. 数据集和加载器设置
    try:
        # data_root 设置为 split_data_root，让 TeqInstanceDataset 查找其子目录
        dataset = TeqInstanceDataset(
            data_root=split_data_root,
            mode=mode,
            img_dir=img_dir,
            instance_dir=instance_dir,
            mask_dir=mask_dir,
            transform=get_val_transform(),  # 使用验证/测试集变换
        )
    except Exception as e:
        print(
            f"!!! 错误: 无法初始化 '{mode}' 数据集于 '{split_data_root}'。跳过。错误信息: {e}"
        )
        return

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.val_batch_size,
        num_workers=0,
        shuffle=False,
        pin_memory=True,
        collate_fn=instance_collate_fn,
    )

    # 3. 推理循环
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"生成 {mode} 标签"):
            if batch is None:
                continue

            # 将数据移动到 GPU
            img = batch["img"].cuda()
            bboxes_list = [b.cuda() for b in batch["bboxes"]]
            centroids_list = [c.cuda() for c in batch["centroids"]]
            mask = batch["mask"]  # (B, H, W)
            img_ids = batch["img_id"]

            # 过滤掉所有图像都没有实例的批次
            if not bboxes_list or all(len(b) == 0 for b in bboxes_list):
                continue

            # 模型前向传播
            predicted_offsets = model(img, bboxes_list, centroids_list)  # [N_total, 2]

            # 将 [N_total, 2] 拆分回列表
            num_instances_per_image = [len(b) for b in bboxes_list]
            pred_offsets_list = torch.split(
                predicted_offsets.detach().cpu(), num_instances_per_image, dim=0
            )

            # 4. 应用偏移和保存
            for i in range(len(img_ids)):
                img_id = img_ids[i]

                # 获取 CPU 上的 numpy 数据
                bboxes_i = bboxes_list[i].cpu().numpy()
                pred_offsets_ratio_i = pred_offsets_list[i].numpy()
                mask_i = mask[i].numpy().astype(np.uint8)

                if len(bboxes_i) == 0:
                    # 如果没有实例，保存一张全黑的掩码
                    blank_mask = np.zeros(ORIGIN_IMG_SIZE, dtype=np.uint8)
                    save_mask_as_png(
                        blank_mask, os.path.join(full_output_path, f"{img_id}.png")
                    )
                    continue
                H, W = mask_i.shape
                input_size = np.array([W, H], dtype=np.float32)

                # 从 512 缩放到 1024
                mask_name = os.path.join(
                    split_data_root, mask_dir, img_id + dataset.mask_suffix
                )
                original_mask_1024 = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
                original_mask_1024 = (original_mask_1024 > 0).astype(np.uint8)
                bboxes_1024 = bboxes_i * (ORIGIN_IMG_SIZE[0] / H)
                input_size_1024 = np.array(ORIGIN_IMG_SIZE, dtype=np.float32)

                # 将预测的比率偏移转换为像素偏移
                pred_offsets_px_1024 = pred_offsets_ratio_i * input_size_1024

                # 应用模型预测的像素偏移
                corrected_mask_i = (
                    apply_offsets_to_mask(
                        shifted_mask=original_mask_1024,
                        bboxes=bboxes_1024,
                        offsets_px=pred_offsets_px_1024,
                    )
                    * 255.0
                )

                # 保存修正后的二值标签 (0/255)
                save_mask_as_png(
                    corrected_mask_i.astype(np.uint8),
                    os.path.join(full_output_path, f"{img_id}.png"),
                )

    print(f"-> 成功将修正后的标签保存至: {full_output_path}")


def main():
    args = get_args()
    seed_everything(42)
    config = py2cfg(args.config_path)

    # 1. 加载模型
    model_path = os.path.join(config.weights_path, config.test_weights_name + ".ckpt")
    if not os.path.exists(model_path):
        print(f"!!! 错误: 无法找到模型权重文件: {model_path}")
        return

    print(f"正在加载模型: {model_path}")
    model = Supervision_Train.load_from_checkpoint(
        model_path,
        config=config,
        strict=False,
    )
    model.cuda()
    model.eval()

    # 2. 遍历并处理所有分割
    data_root = args.data_root

    # 假设您的数据根目录结构是 {data_root}/{split_name}/...
    splits = ["train", "val", "test"]

    for split in splits:
        split_data_root = os.path.join(data_root, split)
        if not os.path.exists(split_data_root):
            print(f"警告: 目录 '{split_data_root}' 不存在。跳过 '{split}' 分割。")
            continue

        generate_labels_for_split(model, config, split_data_root, split)

    print("\n所有分割的修正标签生成任务完成。")


if __name__ == "__main__":
    main()
