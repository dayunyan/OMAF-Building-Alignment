# 在 teq_dataset.py 中

import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from PIL import Image
import random

CLASSES = ("Background", "Building")
PALETTE = [[0, 0, 0], [255, 255, 255]]

ORIGIN_IMG_SIZE = (1024, 1024)
INPUT_IMG_SIZE = (512, 512)
# TEST_IMG_SIZE = (1024, 1024) # 似乎没用到

# -----------------------------------------------------------------
# ！！！关键：新的数据增强管道 ！！！
# -----------------------------------------------------------------


def get_training_transform():
    """
    新的训练数据增强。
    - 移除了所有旋转和翻转 (Issue 2)
    - 使用 Keypoint 变换来处理质心和偏移 (Issue 1, 3, 5)
    - 使用 label_fields 来同步所有数据的裁剪 (Issue 1)
    """
    return albu.Compose(
        [
            # # 1. 缩放 (替换了旧的 RandomScale)
            # # 保持图像的宽高比，缩放到一个随机尺寸
            # albu.RandomScale(
            #     scale_limit=(0.5, 1.5), interpolation=cv2.INTER_CUBIC, p=0.5
            # ),
            # # 2. 填充到裁剪尺寸 (如果图像小于 512)
            # albu.PadIfNeeded(
            #     min_height=INPUT_IMG_SIZE[0],
            #     min_width=INPUT_IMG_SIZE[1],
            #     border_mode=cv2.BORDER_CONSTANT,
            #     value=0,
            #     mask_value=0,
            # ),
            # # 3. 随机裁剪 (替换了旧的 SmartCrop)
            # albu.RandomCrop(height=INPUT_IMG_SIZE[0], width=INPUT_IMG_SIZE[1]),
            albu.Resize(height=INPUT_IMG_SIZE[0], width=INPUT_IMG_SIZE[1]),
            # 4. 归一化
            albu.Normalize(mean=[0.508, 0.458, 0.430], std=[0.194, 0.172, 0.158]),
            # 5. 转换为张量
            ToTensorV2(),
        ],
        bbox_params=albu.BboxParams(
            format="pascal_voc",
            label_fields=["instance_labels"],  # ！！！关键：同步 BBox 和 标签
        ),
        keypoint_params=albu.KeypointParams(
            format="xy",
            label_fields=["keypoint_labels"],  # ！！！关键：同步 Keypoints 和 标签
        ),
        additional_targets={"gt_mask": "mask"},
    )


def get_val_transform():
    """
    验证集变换（通常只是 Resize 和 Normalize）
    """
    return albu.Compose(
        [
            albu.Resize(height=INPUT_IMG_SIZE[0], width=INPUT_IMG_SIZE[1]),
            albu.Normalize(mean=[0.508, 0.458, 0.430], std=[0.194, 0.172, 0.158]),
            ToTensorV2(),
        ],
        bbox_params=albu.BboxParams(
            format="pascal_voc", label_fields=["instance_labels"]
        ),
        keypoint_params=albu.KeypointParams(
            format="xy", label_fields=["keypoint_labels"]
        ),
        additional_targets={"gt_mask": "mask"},
    )


class TeqInstanceDataset(Dataset):
    def __init__(
        self,
        data_root="data/segmentation/Turkey/Islahiye/pre/test",
        mode="val",
        img_dir="images",
        img_suffix=".png",
        instance_dir="instances",  # 存放实例 .pt 文件的目录
        instance_suffix=".pt",
        mask_dir="labels",
        mask_suffix=".png",
        gt_dir="gt",
        gt_suffix=".png",
        transform=None,
    ):
        self.data_root = data_root
        self.img_dir = img_dir
        self.img_suffix = img_suffix
        self.instance_dir = instance_dir
        self.instance_suffix = instance_suffix
        self.mask_dir = mask_dir
        self.mask_suffix = mask_suffix
        self.gt_dir = gt_dir
        self.gt_suffix = gt_suffix
        self.transform = transform
        self.mode = mode
        self.img_ids = self.get_img_ids(self.data_root, self.img_dir)
        self.origin_size_tensor = torch.tensor(
            [ORIGIN_IMG_SIZE[1], ORIGIN_IMG_SIZE[0]], dtype=torch.float32
        )
        self.input_size_tensor = torch.tensor(
            [INPUT_IMG_SIZE[1], INPUT_IMG_SIZE[0]], dtype=torch.float32
        )

    def __getitem__(self, index):
        try:
            img_id = self.img_ids[index]

            # 1. 加载图像 (使用 CV2, Albumentations 需要)
            img_name = osp.join(self.data_root, self.img_dir, img_id + self.img_suffix)
            img = cv2.imread(img_name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 2. 加载实例数据
            instance_name = osp.join(
                self.data_root, self.instance_dir, img_id + self.instance_suffix
            )
            instance_data = torch.load(instance_name, map_location="cpu")

            bboxes = instance_data["bboxes"]
            centroids = instance_data["centroids"]
            gt_offsets_ratio = instance_data["gt_offsets"]
            gt_confidences = instance_data["gt_confidences"]

            num_instances = bboxes.shape[0]
            if num_instances == 0:
                raise ValueError("No instances in this image.")

            mask_name = osp.join(
                self.data_root, self.mask_dir, img_id + self.mask_suffix
            )
            mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Mask not found: {mask_name}")
            # 将掩码值从 0/255 转换为 0/1
            mask = (mask > 0).astype(np.uint8)
            if self.mode == "test":
                gt_name = osp.join(
                    self.data_root, self.gt_dir, img_id + self.mask_suffix
                )
                gt_mask = cv2.imread(gt_name, cv2.IMREAD_GRAYSCALE)
                gt_mask = (gt_mask > 0).astype(np.uint8)

            # 3. 准备变换

            # 3.1. 将比率偏移转换为像素偏移
            gt_offsets_px = gt_offsets_ratio * self.origin_size_tensor
            centroids_np = centroids.cpu().numpy()
            gt_offsets_px_np = gt_offsets_px.cpu().numpy()

            # 获取原始图像尺寸 W 和 H
            W = self.origin_size_tensor[0].item()  # 图像宽度
            H = self.origin_size_tensor[1].item()  # 图像高度

            EPSILON = 1.0
            W_LIMIT = W - EPSILON  # 1023.999999
            H_LIMIT = H - EPSILON

            # 3.2. 准备关键点：[ (cx,cy), (cx+dx, cy+dy), ... ]
            keypoints = []
            for i in range(num_instances):
                cx, cy = centroids_np[i]
                dx, dy = gt_offsets_px_np[i]
                x_end, y_end = cx + dx, cy + dy

                # 1. 计算将 start point 和 end point 整体拉回 [0, W] x [0, H] 所需的平移量
                # 水平方向的修正：
                min_x = min(cx, x_end)
                max_x = max(cx, x_end)

                # 修正量：如果最左侧坐标 < 0，则向右推 (shift > 0)
                shift_x_right = max(0.0, 0.0 - min_x)
                # 修正量：如果最右侧坐标 > W，则向左推 (shift < 0)
                shift_x_left = min(0.0, W_LIMIT - max_x)

                # 最终的水平平移修正量
                shift_x = shift_x_right + shift_x_left

                # 垂直方向的修正：
                min_y = min(cy, y_end)
                max_y = max(cy, y_end)

                shift_y_down = max(0.0, 0.0 - min_y)
                shift_y_up = min(0.0, H_LIMIT - max_y)

                # 最终的垂直平移修正量
                shift_y = shift_y_down + shift_y_up

                # 2. 应用修正量：保持偏移向量 (dx, dy) 不变
                cx_norm = cx + shift_x
                cy_norm = cy + shift_y
                x_end_norm = x_end + shift_x
                y_end_norm = y_end + shift_y

                # 4. 将修正后的关键点添加到列表中（Albumentations-safe points）
                keypoints.append((cx, cy))
                keypoints.append((cx_norm, cy_norm))  # 质心 (start point)
                keypoints.append((x_end_norm, y_end_norm))  # 终点 (end point)

            # 3.3. 准备标签 (用于 Albumentations 同步)
            # 我们使用 [0, 1, 2, ...] 作为标签
            instance_labels = list(range(num_instances))
            # keypoint_labels 必须与 keypoints 数量一致
            # 我们给 (start, end) 对相同的标签
            keypoint_labels = [label for label in instance_labels for _ in range(3)]

            if self.transform:
                if self.mode == "test":
                    packed = dict(
                        image=img,
                        mask=mask,
                        gt_mask=gt_mask,
                        bboxes=bboxes.tolist(),
                        keypoints=keypoints,
                        instance_labels=instance_labels,  # 用于 bboxes
                        keypoint_labels=keypoint_labels,  # 用于 keypoints
                    )
                else:
                    packed = dict(
                        image=img,
                        mask=mask,
                        bboxes=bboxes.tolist(),
                        keypoints=keypoints,
                        instance_labels=instance_labels,  # 用于 bboxes
                        keypoint_labels=keypoint_labels,  # 用于 keypoints
                    )
                augmented = self.transform(**packed)

                img = augmented["image"]
                mask = augmented["mask"]
                if self.mode == "test":
                    gt_mask = augmented["gt_mask"]
                aug_bboxes = torch.tensor(augmented["bboxes"], dtype=torch.float32)
                aug_keypoints = augmented["keypoints"]

                # ！！！关键：aug_labels 告诉我们哪些原始实例在裁剪后被保留
                aug_labels = augmented["instance_labels"]

                # 验证 keypoints 是否也被正确过滤了
                aug_kp_labels = augmented["keypoint_labels"]

                # 如果所有实例都被裁剪，重新采样
                if len(aug_labels) == 0:
                    raise ValueError("All instances cropped out.")

                # 4. 后处理：从变换后的数据中重建
                final_centroids = []
                final_gt_offsets = []
                final_gt_confidences = []

                # A. 确保 keypoints 和 bbox 标签一致
                # 我们只关心 'aug_labels' 中的实例
                aug_labels_set = set(aug_labels)

                # B. 创建一个字典来快速查找 keypoints
                kp_dict = {}
                for kp, label in zip(aug_keypoints, aug_kp_labels):
                    if label not in aug_labels_set:
                        continue  # 这个实例的 bbox 被裁了
                    if label not in kp_dict:
                        kp_dict[label] = []
                    kp_dict[label].append(kp)

                # C. 过滤 BBoxes
                final_bboxes = []

                for i, original_index in enumerate(aug_labels):
                    # 检查此实例的 keypoints 是否完整保留
                    if (
                        original_index not in kp_dict
                        or len(kp_dict[original_index]) != 3
                    ):
                        # 实例的 bbox 保留了，但质心或终点被裁了，丢弃
                        continue

                    final_bboxes.append(aug_bboxes[i])

                    kp_origin, kp_start, kp_end = kp_dict[original_index]

                    aug_cx, aug_cy = kp_origin
                    aug_dx = kp_end[0] - kp_start[0]
                    aug_dy = kp_end[1] - kp_start[1]

                    final_centroids.append((aug_cx, aug_cy))
                    final_gt_offsets.append((aug_dx, aug_dy))

                    # 找到此实例对应的原始置信度
                    final_gt_confidences.append(gt_confidences[original_index])

                if len(final_gt_offsets) == 0:
                    raise ValueError("All instances lost during keypoint filtering.")

                final_bboxes = torch.stack(final_bboxes)
                final_centroids = torch.tensor(final_centroids, dtype=torch.float32)
                final_gt_confidences = torch.stack(final_gt_confidences)

                # 4.1. 重新归一化偏移 (Issue 3)
                # 使用 INPUT_IMG_SIZE 作为新的基准
                final_gt_offsets_px = torch.tensor(
                    final_gt_offsets, dtype=torch.float32
                )
                final_gt_offsets_ratio = final_gt_offsets_px / self.input_size_tensor

            else:
                # 如果没有 transform（例如 val），我们仍然需要归一化
                # 注意：val transform 也应该这样做，这里简化了
                img = ToTensorV2()(image=img)["image"]  # 仅示例
                mask = torch.from_numpy(mask).long()
                if self.mode == "test":
                    gt_mask = torch.from_numpy(gt_mask).long()
                final_bboxes = bboxes
                final_centroids = centroids
                final_gt_offsets_ratio = gt_offsets_ratio
                final_gt_confidences = gt_confidences

            results = dict(
                img_id=img_id,
                img=img.float(),
                mask=mask,
                bboxes=final_bboxes,
                centroids=final_centroids,
                gt_offsets=final_gt_offsets_ratio,  # 始终返回比率
                gt_confidences=final_gt_confidences,
            )
            if self.mode == "test":
                results["gt_mask"] = gt_mask

            return results

        except ValueError as e:
            # print(f"Warning: Error processing {self.img_ids[index]} ({e}). Resampling.")
            # 发生任何错误（如无实例、裁剪后无实例），则随机重新采样
            return self.__getitem__(random.randint(0, len(self.img_ids) - 1))

    def __len__(self):
        return len(self.img_ids)

    def get_img_ids(self, data_root, img_dir):
        img_filename_list = os.listdir(osp.join(data_root, img_dir))
        img_filename_list = [
            f
            for f in img_filename_list
            if "post" not in f and f.endswith(self.img_suffix)
        ]
        img_ids = [str(id.split(".")[0]) for id in img_filename_list]
        return img_ids


# -----------------------------------------------------------------
# ！！！Collate Fn (保持不变)！！！
# -----------------------------------------------------------------
def instance_collate_fn(batch):
    """
    处理可变数量实例的批次。
    'batch' 是一个列表，列表中的每个元素都是 TeqDataset.__getitem__ 返回的 'results' 字典。
    """
    # 过滤掉 None 的项 (如果 __getitem__ 返回 None 的话)
    batch = [item for item in batch if item is not None]
    if not batch:
        return None  # 或者抛出异常

    img_ids = [item["img_id"] for item in batch]
    images = torch.stack([item["img"] for item in batch], dim=0)

    bboxes_list = [item["bboxes"] for item in batch]
    centroids_list = [item["centroids"] for item in batch]
    gt_offsets_list = [item["gt_offsets"] for item in batch]
    gt_confidences_list = [item["gt_confidences"] for item in batch]

    masks = torch.stack([item["mask"] for item in batch], dim=0)
    packed = {
        "img_id": img_ids,
        "img": images,
        "mask": masks,
        "bboxes": bboxes_list,
        "centroids": centroids_list,
        "gt_offsets": gt_offsets_list,
        "gt_confidences": gt_confidences_list,
    }
    if "gt_mask" in batch[0] and batch[0]["gt_mask"] is not None:
        gt_masks = torch.stack([item["gt_mask"] for item in batch], dim=0)
        packed["gt_mask"] = gt_masks

    return packed
