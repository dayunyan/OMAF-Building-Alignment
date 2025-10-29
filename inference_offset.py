import argparse
from pathlib import Path
import glob
from PIL import Image
import ttach as tta
import cv2
import numpy as np
import torch
import albumentations as albu

# from catalyst.dl import SupervisedRunner
from skimage.morphology import remove_small_holes, remove_small_objects
from tools.cfg import py2cfg
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from train_offset_v2 import *
import random
import os


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def building_to_rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [0, 0, 0]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [255, 255, 255]
    return mask_rgb


def pv2rgb(mask):  # Potsdam and vaihingen
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [0, 255, 0]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [255, 255, 255]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [255, 0, 0]
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [255, 255, 0]
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [0, 204, 255]
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [0, 0, 255]
    return mask_rgb


def landcoverai_to_rgb(mask):
    w, h = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(w, h, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [255, 255, 255]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [233, 193, 133]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [255, 0, 0]
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [0, 255, 0]
    mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
    return mask_rgb


def uavid2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [128, 0, 0]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [128, 64, 128]
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [0, 128, 0]
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [128, 128, 0]
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [64, 0, 128]
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [192, 0, 192]
    mask_rgb[np.all(mask_convert == 6, axis=0)] = [64, 64, 0]
    mask_rgb[np.all(mask_convert == 7, axis=0)] = [0, 0, 0]
    mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
    return mask_rgb


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg(
        "-i",
        "--image_path",
        type=Path,
        required=True,
        help="Path to  huge image folder",
        default="../data/xBD/test/images",
    )
    arg(
        "-c",
        "--config_path",
        type=Path,
        required=True,
        help="Path to config",
        default="config/xBD/unetformer.py",
    )
    arg(
        "-o",
        "--output_path",
        type=Path,
        help="Path to save resulting masks.",
        required=True,
        default="fig_results/xbd/unetformer",
    )
    arg(
        "-t",
        "--tta",
        help="Test time augmentation.",
        default=None,
        choices=[None, "d4", "lr"],
    )
    arg("-ph", "--patch-height", help="height of patch size", type=int, default=512)
    arg("-pw", "--patch-width", help="width of patch size", type=int, default=512)
    arg("-b", "--batch-size", help="batch size", type=int, default=4)
    arg(
        "-d",
        "--dataset",
        help="dataset",
        default="pv",
        choices=["pv", "landcoverai", "uavid", "building"],
    )
    return parser.parse_args()


def get_img_padded(image, patch_size):
    oh, ow = image.shape[0], image.shape[1]
    rh, rw = oh % patch_size[0], ow % patch_size[1]

    width_pad = 0 if rw == 0 else patch_size[1] - rw
    height_pad = 0 if rh == 0 else patch_size[0] - rh
    # print(oh, ow, rh, rw, height_pad, width_pad)
    h, w = oh + height_pad, ow + width_pad

    pad = albu.PadIfNeeded(
        min_height=h,
        min_width=w,
        position="bottom_right",
        border_mode=0,
        value=[0, 0, 0],
    )(image=image)
    img_pad = pad["image"]
    return img_pad, height_pad, width_pad


class InferenceDataset(Dataset):
    def __init__(self, tile_list=None, transform=albu.Normalize()):
        self.tile_list = tile_list
        self.transform = transform

    def __getitem__(self, index):
        patch = self.tile_list[index]
        if isinstance(patch, tuple):
            img = patch[0]
            gt = patch[1]
        else:
            img = patch
            gt = None
        img_id = index
        aug = self.transform(image=img)
        img = aug["image"]
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        results = dict(img_id=img_id, img=img, gt=gt)
        return results

    def __len__(self):
        return len(self.tile_list)


def get_inf_dataset(tile_list):
    return InferenceDataset(
        tile_list=tile_list,
        transform=albu.Normalize(mean=[0.508, 0.458, 0.430], std=[0.194, 0.172, 0.158]),
    )


def make_dataset_for_one_huge_image(img_path, patch_size):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.IMREAD_COLOR)
    tile_list = []
    image_pad, height_pad, width_pad = get_img_padded(img.copy(), patch_size)

    output_height, output_width = image_pad.shape[0], image_pad.shape[1]

    for x in range(0, output_height, patch_size[0]):
        for y in range(0, output_width, patch_size[1]):
            image_tile = image_pad[x : x + patch_size[0], y : y + patch_size[1]]
            tile_list.append(image_tile)

    dataset = get_inf_dataset(tile_list=tile_list)
    return (
        dataset,
        width_pad,
        height_pad,
        output_width,
        output_height,
        image_pad,
        img.shape,
    )


def make_dataset_for_one_huge_image_with_gt(
    img_path, gt_path, patch_size, mode="resize"
):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gt = cv2.imread(gt_path, cv2.IMREAD_COLOR)
    tile_list = []
    if mode == "resize":
        intp = cv2.INTER_AREA if patch_size[-1] < img.shape[-1] else cv2.INTER_CUBIC
        img = cv2.resize(img, patch_size, interpolation=intp)
        gt = cv2.resize(gt, patch_size)
        height_pad, width_pad, output_height, output_width, image_pad, gt_pad = (
            0,
            0,
            img.shape[0],
            img.shape[1],
            img,
            gt,
        )

        tile_list.append((img, gt))
    elif mode == "crop":
        image_pad, height_pad, width_pad = get_img_padded(img.copy(), patch_size)
        gt_pad, _, _ = get_img_padded(gt.copy(), patch_size)
        gt_pad = cv2.cvtColor(gt_pad, cv2.COLOR_BGR2GRAY)

        output_height, output_width = image_pad.shape[0], image_pad.shape[1]

        for x in range(0, output_height, patch_size[0]):
            for y in range(0, output_width, patch_size[1]):
                image_tile = image_pad[x : x + patch_size[0], y : y + patch_size[1]]
                gt_tile = gt_pad[x : x + patch_size[0], y : y + patch_size[1]]
                tile_list.append((image_tile, gt_tile))
    else:
        raise NotImplementedError(f"mode must be one of [resize, crop]")

    dataset = get_inf_dataset(tile_list=tile_list)
    return (
        dataset,
        width_pad,
        height_pad,
        output_width,
        output_height,
        image_pad,
        gt_pad,
        img.shape,
    )


def offset_logits(logits: torch.Tensor, offset: torch.Tensor):
    """
    此函数使用 offset 张量对 logits 张量中的像素位置进行移动。

    参数:
    logits (torch.Tensor): 形状为 [B, C, H, W] 的张量，表示模型输出的 logits
    offset (torch.Tensor): 形状为 [B, 2, H, W] 的张量，其中两个通道分别表示在 x 和 y 方向上的偏移量

    返回:
    torch.Tensor: 移动后的 logits 张量
    """
    B, C, H, W = logits.shape
    pred = logits.argmax(dim=1, keepdim=True).expand(-1, 2, -1, -1)
    # 检查输入张量的形状是否符合预期
    assert offset.shape == (B, 2, H, W), "Offset 张量的形状应为 [B, 2, H, W]"
    new_offset = torch.where(pred == 1, offset, 0)

    # 生成坐标网格
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    y = y.unsqueeze(0).expand(B, -1, -1).to(logits.device)
    x = x.unsqueeze(0).expand(B, -1, -1).to(logits.device)
    # 计算新的坐标
    new_y = (y - torch.round(H * new_offset[:, 1, :, :])).to(torch.int64)
    new_x = (x - torch.round(W * new_offset[:, 0, :, :])).to(torch.int64)
    # 确保新坐标在合法范围内
    new_y = torch.clamp(new_y, 0, H - 1).unsqueeze(1).expand(-1, C, -1, -1)
    new_x = torch.clamp(new_x, 0, W - 1).unsqueeze(1).expand(-1, C, -1, -1)
    # print(f"new_y: {new_y},\n new_x: {new_x}")
    # 初始化 shifted_logits
    shifted_logits_y = logits.clone()
    # 使用广播机制更新 shifted_logits
    shifted_logits_y.scatter_(
        2,
        new_y,
        logits,  # .gather(2, new_y.unsqueeze(1).expand(-1, C, -1, -1)),
    )
    # print(f"shifted_logits: {shifted_logits_y}")
    new_x_shifted = torch.zeros_like(new_x)
    new_x_shifted.scatter_(2, new_y, new_x)
    # print(f"new_x_shifted: {new_x_shifted}")
    shifted_logits_xy = logits.clone()
    shifted_logits_xy.scatter_(
        3,
        new_x_shifted,
        shifted_logits_y,  # .gather(3, new_x.unsqueeze(1).expand(-1, C, -1, -1)),
    )
    # print(f"shifted_logits: {shifted_logits_xy}")
    return shifted_logits_xy


class ModelOutputAdapter(nn.Module):
    def __init__(self, original_model):
        super(ModelOutputAdapter, self).__init__()
        self.original_model = original_model

    def forward(self, x):
        output_dict = self.original_model(x)
        adapted_output = output_dict["logits"]
        return adapted_output


def main():
    args = get_args()
    seed_everything(42)
    patch_size = (args.patch_height, args.patch_width)
    config = py2cfg(args.config_path)
    # model = Supervision_Train(config=config)
    model = Supervision_Train.load_from_checkpoint(
        os.path.join(config.weights_path, config.test_weights_name + ".ckpt"),
        config=config,
        strict=False,
    )
    adp_model = ModelOutputAdapter(model)

    model.cuda()
    model.eval()
    adp_model.cuda()
    adp_model.eval()

    if args.tta == "lr":
        transforms = tta.Compose([tta.HorizontalFlip(), tta.VerticalFlip()])
        adp_model = tta.SegmentationTTAWrapper(adp_model, transforms)
    elif args.tta == "d4":
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                # tta.VerticalFlip(),
                # tta.Rotate90(angles=[0, 90, 180, 270]),
                tta.Scale(scales=[0.75, 1, 1.25, 1.5, 1.75]),
                # tta.Multiply(factors=[0.8, 1, 1.2])
            ]
        )
        adp_model = tta.SegmentationTTAWrapper(adp_model, transforms)

    img_paths = []
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(os.path.join(args.output_path, "logits"), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, "logits_vis"), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, "offset_vis"), exist_ok=True)
    for ext in ("*.tif", "*.png", "*.jpg"):
        img_paths.extend(glob.glob(os.path.join(args.image_path, ext)))
    img_paths.sort()
    # print(img_paths)
    gt_dir = os.path.join(os.path.dirname(args.image_path), "gt")
    for img_path in tqdm(img_paths, leave=True, position=0):
        img_name = img_path.split("/")[-1]
        gt_name = img_name.replace(".png", ".png")
        gt_path = os.path.join(gt_dir, gt_name)
        # print('origin mask', original_mask.shape)
        (
            dataset,
            width_pad,
            height_pad,
            output_width,
            output_height,
            img_pad,
            gt_pad,
            img_shape,
        ) = make_dataset_for_one_huge_image_with_gt(img_path, gt_path, patch_size)
        # print('img_padded', img_pad.shape)
        output_mask = np.zeros(shape=(output_height, output_width), dtype=np.uint8)
        output_logits = np.zeros(
            shape=(config.num_classes, output_height, output_width)
        )
        output_offset = np.zeros(shape=(2, output_height, output_width))
        output_tiles = []
        k = 0
        with torch.no_grad():
            dataloader = DataLoader(
                dataset=dataset,
                batch_size=args.batch_size,
                drop_last=False,
                shuffle=False,
            )
            for input in tqdm(dataloader, leave=False, position=1):
                # raw_prediction NxCxHxW
                raw_predictions = adp_model(input["img"].cuda())
                pre_offset = model(input["img"].cuda())["offset"]
                # raw_predictions = offset_logits(raw_predictions, pre_offset)
                # print('raw_pred shape:', raw_predictions.shape)
                raw_predictions = nn.Softmax(dim=1)(raw_predictions)
                # input_images['features'] NxCxHxW C=3
                predictions = raw_predictions.argmax(dim=1)
                image_ids = input["img_id"]
                # print('prediction', predictions.shape)
                # print(np.unique(predictions))

                for i in range(predictions.shape[0]):
                    logits = raw_predictions[i].cpu().numpy()
                    mask = predictions[i].cpu().numpy()
                    offset = pre_offset[i].cpu().numpy()
                    output_tiles.append(
                        (mask, image_ids[i].cpu().numpy(), logits, offset)
                    )

        for m in range(0, output_height, patch_size[0]):
            for n in range(0, output_width, patch_size[1]):
                output_mask[m : m + patch_size[0], n : n + patch_size[1]] = (
                    output_tiles[k][0]
                )
                # print(output_tiles[k][1])
                output_logits[:, m : m + patch_size[0], n : n + patch_size[1]] = (
                    output_tiles[k][2]
                )
                output_offset[:, m : m + patch_size[0], n : n + patch_size[1]] = (
                    output_tiles[k][3]
                )
                k = k + 1

        output_mask = output_mask[-img_shape[0] :, -img_shape[1] :]
        output_logits = output_logits[:, -img_shape[0] :, -img_shape[1] :]
        output_offset = output_offset[:, -img_shape[0] :, -img_shape[1] :]

        # print('mask', output_mask.shape)
        if args.dataset == "landcoverai":
            output_mask = landcoverai_to_rgb(output_mask)
        elif args.dataset == "pv":
            output_mask = pv2rgb(output_mask)
        elif args.dataset == "uavid":
            output_mask = uavid2rgb(output_mask)
        elif args.dataset == "building":
            output_mask = building_to_rgb(output_mask)
        else:
            output_mask = output_mask
        # print(img_shape, output_mask.shape)
        # assert img_shape == output_mask.shape
        cv2.imwrite(os.path.join(args.output_path, "logits_vis", img_name), output_mask)
        np.save(
            os.path.join(args.output_path, "logits", img_name.replace(".png", ".npy")),
            output_logits,
        )
        cv2.imwrite(
            os.path.join(
                args.output_path, "offset_vis", img_name.replace(".png", "_x.png")
            ),
            255
            * (output_offset[0] - output_offset[0].min())
            / (output_offset[0].max() - output_offset[0].min()),
        )
        cv2.imwrite(
            os.path.join(
                args.output_path, "offset_vis", img_name.replace(".png", "_y.png")
            ),
            255
            * (output_offset[1] - output_offset[1].min())
            / (output_offset[1].max() - output_offset[1].min()),
        )


if __name__ == "__main__":
    main()
