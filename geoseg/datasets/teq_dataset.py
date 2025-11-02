import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import matplotlib.pyplot as plt
import albumentations as albu

import matplotlib.patches as mpatches
from PIL import Image
import random
from .transform import *

CLASSES = ("Background", "Building")
PALETTE = [[0, 0, 0], [255, 255, 255]]

ORIGIN_IMG_SIZE = (1024, 1024)
INPUT_IMG_SIZE = (512, 512)
TEST_IMG_SIZE = (1024, 1024)


def get_training_transform():
    additional_targets = {"align_conf": "mask"}  # 将pred_mask视为与mask相同类型
    train_transform = [
        albu.Resize(height=INPUT_IMG_SIZE[0], width=INPUT_IMG_SIZE[1]),
        albu.RandomRotate90(p=0.5),
        albu.Normalize(mean=[0.508, 0.458, 0.430], std=[0.194, 0.172, 0.158]),
    ]
    return albu.Compose(train_transform, additional_targets=additional_targets)


def train_aug(img, mask, align_conf=None):
    crop_aug = Compose(
        [
            RandomScale(scale_list=[0.5, 0.75, 1.0, 1.25, 1.5], mode="value"),
            SmartCropV1(
                crop_size=512, max_ratio=0.75, ignore_index=len(CLASSES), nopad=False
            ),
        ]
    )
    img, mask, align_conf = crop_aug(img, mask, align_conf)
    img, mask = np.array(img), np.array(mask)
    if align_conf is not None:
        align_conf = np.array(align_conf)
        aug = get_training_transform()(
            image=img.copy(), mask=mask.copy(), align_conf=align_conf.copy()
        )
        img, mask, align_conf = aug["image"], aug["mask"], aug["align_conf"]
    else:
        aug = get_training_transform()(image=img.copy(), mask=mask.copy())
        img, mask = aug["image"], aug["mask"]
    return img, mask, align_conf


def get_val_transform():
    additional_targets = {
        "align_conf": "mask",
        "gt": "mask",
    }  # 将pred_mask视为与mask相同类型
    val_transform = [
        albu.Resize(height=INPUT_IMG_SIZE[0], width=INPUT_IMG_SIZE[1]),
        albu.Normalize(mean=[0.508, 0.458, 0.430], std=[0.194, 0.172, 0.158]),
    ]
    return albu.Compose(val_transform, additional_targets=additional_targets)


def val_aug(img, mask, align_conf=None, gt=None):
    img, mask = np.array(img), np.array(mask)
    if align_conf is not None and gt is not None:
        align_conf = np.array(align_conf)
        gt = np.array(gt)
        aug = get_val_transform()(
            image=img.copy(),
            mask=mask.copy(),
            align_conf=align_conf.copy(),
            gt=gt.copy(),
        )
        img, mask, align_conf, gt = (
            aug["image"],
            aug["mask"],
            aug["align_conf"],
            aug["gt"],
        )
    else:
        if align_conf is not None:
            align_conf = np.array(align_conf)
            aug = get_val_transform()(
                image=img.copy(), mask=mask.copy(), align_conf=align_conf.copy()
            )
            img, mask, align_conf = aug["image"], aug["mask"], aug["align_conf"]
        elif gt is not None:
            gt = np.array(gt)
            aug = get_val_transform()(image=img.copy(), mask=mask.copy(), gt=gt.copy())
            img, mask, gt = aug["image"], aug["mask"], aug["gt"]
        else:
            aug = get_val_transform()(image=img.copy(), mask=mask.copy())
            img, mask = aug["image"], aug["mask"]
    return img, mask, align_conf, gt


class TeqDataset(Dataset):
    def __init__(
        self,
        data_root="data/segmentation/Turkey/Islahiye/pre/test",
        mode="val",
        img_dir="images",
        mask_dir="labels",
        test_gt_dir="gt",
        img_suffix=".png",
        mask_suffix=".png",
        transform=val_aug,
        mosaic_ratio=0.0,
        img_size=ORIGIN_IMG_SIZE,
        pred_dir=None,
        align_dir=None,
        align_suffix=".pt",
    ):
        self.data_root = data_root
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.gt_dir = test_gt_dir
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.transform = transform
        self.mode = mode
        self.mosaic_ratio = mosaic_ratio
        self.img_size = img_size
        self.pred_dir = pred_dir
        self.align_dir = align_dir
        self.align_suffix = align_suffix

        self.img_ids = self.get_img_ids(
            self.data_root,
            self.img_dir,
            self.mask_dir,
            self.gt_dir if mode == "test" else None,
        )

    def __getitem__(self, index):
        p_ratio = random.random()
        if p_ratio > self.mosaic_ratio or self.mode == "val" or self.mode == "test":
            if self.mode == "test":
                img, mask, align_conf, gt = self.load_img_and_mask(
                    index,
                    self.img_ids,
                    self.data_root,
                    self.img_dir,
                    self.img_suffix,
                    self.mask_dir,
                    self.mask_suffix,
                    self.pred_dir,
                    self.align_dir,
                    self.align_suffix,
                    gt_dir=self.gt_dir,
                )
            else:
                img, mask, align_conf, gt = self.load_img_and_mask(
                    index,
                    self.img_ids,
                    self.data_root,
                    self.img_dir,
                    self.img_suffix,
                    self.mask_dir,
                    self.mask_suffix,
                    self.pred_dir,
                    self.align_dir,
                    self.align_suffix,
                )
        else:
            img, mask, align_conf, gt = self.load_mosaic_img_and_mask(
                index,
                self.img_ids,
                self.data_root,
                self.img_dir,
                self.img_suffix,
                self.mask_dir,
                self.mask_suffix,
                self.pred_dir,
                self.align_dir,
                self.align_suffix,
            )
        if self.transform:
            if self.mode == "test":
                img, mask, align_conf, gt = self.transform(img, mask, align_conf, gt)
            else:
                img, mask, align_conf = self.transform(img, mask, align_conf)

        img = torch.from_numpy(img).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask / 255.0).long()
        # if pred_mask is not None:
        #     pred_mask = torch.from_numpy(pred_mask / 255.0).long()
        if align_conf is not None:
            align_conf = torch.from_numpy(align_conf)

        if gt is not None:
            gt = torch.from_numpy(gt / 255.0).long()

        img_id = self.img_ids[index]
        results = dict(
            img_id=img_id, img=img, gt_semantic_seg=mask, align_conf=align_conf
        )
        if self.mode == "test":
            results.update(dict(align_gt=gt))
        return results

    def __len__(self):
        return len(self.img_ids)

    def get_img_ids(self, data_root, img_dir, mask_dir, gt_dir=None):
        img_filename_list = os.listdir(osp.join(data_root, img_dir))
        mask_filename_list = os.listdir(osp.join(data_root, mask_dir))
        if gt_dir:
            gt_filename_list = os.listdir(osp.join(data_root, gt_dir))
        for _ in range(len(img_filename_list) - 1, -1, -1):
            if "post" in img_filename_list[_]:
                img_filename_list.pop(_)
                mask_filename_list.pop(_)

        assert (
            len(img_filename_list) == len(mask_filename_list) == len(gt_filename_list)
            if gt_dir
            else len(img_filename_list) == len(mask_filename_list)
        )
        img_ids = [str(id.split(".")[0]) for id in img_filename_list]
        return img_ids

    def load_img_and_mask(
        self,
        index,
        img_ids,
        data_root,
        img_dir,
        img_suffix,
        mask_dir,
        mask_suffix,
        pred_dir=None,
        align_dir=None,
        align_suffix=".pt",
        gt_dir=None,
    ):
        img_id = img_ids[index]
        img_name = osp.join(data_root, img_dir, img_id + img_suffix)
        mask_name = osp.join(data_root, mask_dir, img_id + mask_suffix)
        img = Image.open(img_name).convert("RGB")
        mask = Image.open(mask_name).convert("L")
        pred_mask = None
        align_conf = None
        gt = None
        if pred_dir:
            pred_name = osp.join(data_root, pred_dir, img_id + mask_suffix)
            if osp.exists(pred_name):
                pred_mask = Image.open(pred_name).convert("L")

        if align_dir:
            conf_name = osp.join(data_root, align_dir, img_id + align_suffix)
            if osp.exists(conf_name):
                align_conf = torch.load(conf_name).cpu().numpy()
                align_conf = Image.fromarray(align_conf, mode="F")

        if gt_dir:
            gt_name = osp.join(data_root, gt_dir, img_id + mask_suffix)
            if osp.exists(gt_name):
                gt = Image.open(gt_name).convert("L")

        return img, mask, align_conf, gt

    def load_mosaic_img_and_mask(
        self,
        index,
        img_ids,
        data_root,
        img_dir,
        img_suffix,
        mask_dir,
        mask_suffix,
        pred_dir=None,
        align_dir=None,
        align_suffix=".pt",
    ):
        indexes = [index] + [random.randint(0, len(img_ids) - 1) for _ in range(3)]
        img_a, mask_a, align_conf_a, _ = self.load_img_and_mask(
            indexes[0],
            img_ids,
            data_root,
            img_dir,
            img_suffix,
            mask_dir,
            mask_suffix,
            pred_dir,
            align_dir,
            align_suffix,
        )
        img_b, mask_b, align_conf_b, _ = self.load_img_and_mask(
            indexes[1],
            img_ids,
            data_root,
            img_dir,
            img_suffix,
            mask_dir,
            mask_suffix,
            pred_dir,
            align_dir,
            align_suffix,
        )
        img_c, mask_c, align_conf_c, _ = self.load_img_and_mask(
            indexes[2],
            img_ids,
            data_root,
            img_dir,
            img_suffix,
            mask_dir,
            mask_suffix,
            pred_dir,
            align_dir,
            align_suffix,
        )
        img_d, mask_d, align_conf_d, _ = self.load_img_and_mask(
            indexes[3],
            img_ids,
            data_root,
            img_dir,
            img_suffix,
            mask_dir,
            mask_suffix,
            pred_dir,
            align_dir,
            align_suffix,
        )

        img_a, mask_a = np.array(img_a), np.array(mask_a)
        img_b, mask_b = np.array(img_b), np.array(mask_b)
        img_c, mask_c = np.array(img_c), np.array(mask_c)
        img_d, mask_d = np.array(img_d), np.array(mask_d)
        if align_conf_a is not None:
            align_conf_a = np.array(align_conf_a)
            align_conf_b = np.array(align_conf_b)
            align_conf_c = np.array(align_conf_c)
            align_conf_d = np.array(align_conf_d)

        h = self.img_size[0]
        w = self.img_size[1]

        start_x = w // 4
        strat_y = h // 4
        # The coordinates of the splice center
        offset_x = random.randint(start_x, (w - start_x))
        offset_y = random.randint(strat_y, (h - strat_y))

        crop_size_a = (offset_x, offset_y)
        crop_size_b = (w - offset_x, offset_y)
        crop_size_c = (offset_x, h - offset_y)
        crop_size_d = (w - offset_x, h - offset_y)

        random_crop_a = albu.RandomCrop(width=crop_size_a[0], height=crop_size_a[1])
        random_crop_b = albu.RandomCrop(width=crop_size_b[0], height=crop_size_b[1])
        random_crop_c = albu.RandomCrop(width=crop_size_c[0], height=crop_size_c[1])
        random_crop_d = albu.RandomCrop(width=crop_size_d[0], height=crop_size_d[1])

        if align_conf_a is not None:
            croped_a = random_crop_a(
                image=img_a.copy(), mask=mask_a.copy(), align_conf=align_conf_a.copy()
            )
            croped_b = random_crop_b(
                image=img_b.copy(), mask=mask_b.copy(), align_conf=align_conf_b.copy()
            )
            croped_c = random_crop_c(
                image=img_c.copy(), mask=mask_c.copy(), align_conf=align_conf_c.copy()
            )
            croped_d = random_crop_d(
                image=img_d.copy(), mask=mask_d.copy(), align_conf=align_conf_d.copy()
            )

            img_crop_a, mask_crop_a, align_conf_crop_a = (
                croped_a["image"],
                croped_a["mask"],
                croped_a["align_conf"],
            )
            img_crop_b, mask_crop_b, align_conf_crop_b = (
                croped_b["image"],
                croped_b["mask"],
                croped_b["align_conf"],
            )
            img_crop_c, mask_crop_c, align_conf_crop_c = (
                croped_c["image"],
                croped_c["mask"],
                croped_c["align_conf"],
            )
            img_crop_d, mask_crop_d, align_conf_crop_d = (
                croped_d["image"],
                croped_d["mask"],
                croped_d["align_conf"],
            )
        else:
            croped_a = random_crop_a(image=img_a.copy(), mask=mask_a.copy())
            croped_b = random_crop_b(image=img_b.copy(), mask=mask_b.copy())
            croped_c = random_crop_c(image=img_c.copy(), mask=mask_c.copy())
            croped_d = random_crop_d(image=img_d.copy(), mask=mask_d.copy())

            img_crop_a, mask_crop_a = croped_a["image"], croped_a["mask"]
            img_crop_b, mask_crop_b = croped_b["image"], croped_b["mask"]
            img_crop_c, mask_crop_c = croped_c["image"], croped_c["mask"]
            img_crop_d, mask_crop_d = croped_d["image"], croped_d["mask"]

        top = np.concatenate((img_crop_a, img_crop_b), axis=1)
        bottom = np.concatenate((img_crop_c, img_crop_d), axis=1)
        img = np.concatenate((top, bottom), axis=0)

        top_mask = np.concatenate((mask_crop_a, mask_crop_b), axis=1)
        bottom_mask = np.concatenate((mask_crop_c, mask_crop_d), axis=1)
        mask = np.concatenate((top_mask, bottom_mask), axis=0)

        mask = np.ascontiguousarray(mask)
        img = np.ascontiguousarray(img)
        img = Image.fromarray(img)
        mask = Image.fromarray(mask)

        align_conf = None
        if align_conf_a is not None:
            top_align_conf = np.concatenate(
                (align_conf_crop_a, align_conf_crop_b), axis=1
            )
            bottom_align_conf = np.concatenate(
                (align_conf_crop_c, align_conf_crop_d), axis=1
            )
            align_conf = np.concatenate((top_align_conf, bottom_align_conf), axis=0)
            align_conf = np.ascontiguousarray(align_conf)
            align_conf = Image.fromarray(align_conf)
        # print(img.shape)

        return img, mask, align_conf, None


def show_img_mask_seg(seg_path, img_path, mask_path, start_seg_index):
    seg_list = os.listdir(seg_path)
    seg_list = [f for f in seg_list if f.endswith(".png")]
    fig, ax = plt.subplots(2, 3, figsize=(18, 12))
    seg_list = seg_list[start_seg_index : start_seg_index + 2]
    patches = [
        mpatches.Patch(color=np.array(PALETTE[i]) / 255.0, label=CLASSES[i])
        for i in range(len(CLASSES))
    ]
    for i in range(len(seg_list)):
        seg_id = seg_list[i]
        img_seg = cv2.imread(f"{seg_path}/{seg_id}", cv2.IMREAD_UNCHANGED)
        img_seg = img_seg.astype(np.uint8)
        img_seg = Image.fromarray(img_seg).convert("P")
        img_seg.putpalette(np.array(PALETTE, dtype=np.uint8))
        img_seg = np.array(img_seg.convert("RGB"))
        mask = cv2.imread(f"{mask_path}/{seg_id}", cv2.IMREAD_UNCHANGED)
        mask = mask.astype(np.uint8)
        mask = Image.fromarray(mask).convert("P")
        mask.putpalette(np.array(PALETTE, dtype=np.uint8))
        mask = np.array(mask.convert("RGB"))
        img_id = str(seg_id.split(".")[0]) + ".tif"
        img = cv2.imread(f"{img_path}/{img_id}", cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax[i, 0].set_axis_off()
        ax[i, 0].imshow(img)
        ax[i, 0].set_title("RS IMAGE " + img_id)
        ax[i, 1].set_axis_off()
        ax[i, 1].imshow(mask)
        ax[i, 1].set_title("Mask True " + seg_id)
        ax[i, 2].set_axis_off()
        ax[i, 2].imshow(img_seg)
        ax[i, 2].set_title("Mask Predict " + seg_id)
        ax[i, 2].legend(
            handles=patches,
            bbox_to_anchor=(1.05, 1),
            loc=2,
            borderaxespad=0.0,
            fontsize="large",
        )


def show_seg(seg_path, img_path, start_seg_index):
    seg_list = os.listdir(seg_path)
    seg_list = [f for f in seg_list if f.endswith(".png")]
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    seg_list = seg_list[start_seg_index : start_seg_index + 2]
    patches = [
        mpatches.Patch(color=np.array(PALETTE[i]) / 255.0, label=CLASSES[i])
        for i in range(len(CLASSES))
    ]
    for i in range(len(seg_list)):
        seg_id = seg_list[i]
        img_seg = cv2.imread(f"{seg_path}/{seg_id}", cv2.IMREAD_UNCHANGED)
        img_seg = img_seg.astype(np.uint8)
        img_seg = Image.fromarray(img_seg).convert("P")
        img_seg.putpalette(np.array(PALETTE, dtype=np.uint8))
        img_seg = np.array(img_seg.convert("RGB"))
        img_id = str(seg_id.split(".")[0]) + ".tif"
        img = cv2.imread(f"{img_path}/{img_id}", cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax[i, 0].set_axis_off()
        ax[i, 0].imshow(img)
        ax[i, 0].set_title("RS IMAGE " + img_id)
        ax[i, 1].set_axis_off()
        ax[i, 1].imshow(img_seg)
        ax[i, 1].set_title("Seg IMAGE " + seg_id)
        ax[i, 1].legend(
            handles=patches,
            bbox_to_anchor=(1.05, 1),
            loc=2,
            borderaxespad=0.0,
            fontsize="large",
        )


def show_mask(img, mask, img_id):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 12))
    patches = [
        mpatches.Patch(color=np.array(PALETTE[i]) / 255.0, label=CLASSES[i])
        for i in range(len(CLASSES))
    ]
    mask = mask.astype(np.uint8)
    mask = Image.fromarray(mask).convert("P")
    mask.putpalette(np.array(PALETTE, dtype=np.uint8))
    mask = np.array(mask.convert("RGB"))
    ax1.imshow(img)
    ax1.set_title("RS IMAGE " + str(img_id) + ".tif")
    ax2.imshow(mask)
    ax2.set_title("Mask " + str(img_id) + ".png")
    ax2.legend(
        handles=patches,
        bbox_to_anchor=(1.05, 1),
        loc=2,
        borderaxespad=0.0,
        fontsize="large",
    )
