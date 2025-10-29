from typing import Dict, List
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import os
import random
import torch
import torch.nn.functional as F
from torch import nn
import cv2
import numpy as np
import argparse
from scipy import ndimage
from pathlib import Path
from tools.cfg import py2cfg
from tools.metric import Evaluator
from pytorch_lightning.loggers import CSVLogger
from tools.uda import prob_2_entropy
from tools.visual import (
    save_tensor_as_png,
    visualize_masks,
    visualize_grayscale_as_pseudocolor,
)
from tools.offset import offset_tensor_v3


# ---- helper functions for inner optimization (per-sample, per-instance) ----
def get_instances(label_tensor):
    """
    输入: label_tensor (torch.Tensor), 形状 [H, W], 值 0（背景）或 1（目标类别）或多类别标签
    输出: inst_map (torch.Tensor), 形状 [H, W], 值 0（背景）, 1...K（实例ID）
            masks (list of torch.BoolTensor), 每个实例的二值掩码
    """
    label_np = label_tensor.cpu().numpy().astype(int)
    labeled, num_objects = ndimage.label(label_np)
    inst_map = torch.from_numpy(labeled).to(label_tensor.device)

    masks = []
    for k in range(1, num_objects + 1):
        mask = inst_map == k
        masks.append(mask)

    return inst_map, masks, num_objects


def create_base_grid(height, width, device, align_corners=True):
    """
    创建归一化到 [-1, 1] 的基础网格，用于 grid_sample。
    输出: grid (torch.Tensor), 形状 [H, W, 2], 其中 grid[..., 0] 是 x 坐标, grid[..., 1] 是 y 坐标。
    """
    # 注意：align_corners True/False 会影响归一化公式；训练/推理时与 grid_sample 的 align_corners 一致即可
    i_tensor = torch.arange(height, device=device).float()
    j_tensor = torch.arange(width, device=device).float()
    grid_y, grid_x = torch.meshgrid(i_tensor, j_tensor, indexing="ij")
    if align_corners:
        grid_x_norm = 2 * grid_x / (width - 1) - 1
        grid_y_norm = 2 * grid_y / (height - 1) - 1
    else:
        grid_x_norm = 2 * (grid_x + 0.5) / width - 1
        grid_y_norm = 2 * (grid_y + 0.5) / height - 1
    grid = torch.stack((grid_x_norm, grid_y_norm), dim=-1)  # [H, W, 2]
    return grid


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Path to the config.", required=True)
    return parser.parse_args()


class Supervision_Train(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.automatic_optimization = True
        self.net = config.net

        self.loss = config.loss

        self.metrics_train = Evaluator(num_class=config.num_classes)
        self.metrics_val = Evaluator(num_class=config.num_classes)

        self.training_step_outputs = []
        self.validation_step_outputs = []

    def forward(self, x):
        # only net is used in the prediction/inference
        seg_pre = self.net(x)
        return seg_pre

    def get_avg_loss(self, step_outputs):
        if isinstance(step_outputs, List):
            if isinstance(step_outputs[0], Dict):
                loss_names = step_outputs[0].keys()
                loss_len = len(step_outputs)
                loss = {}
                for ln in loss_names:
                    loss[ln] = (
                        sum([o[ln].cpu().detach().numpy() for o in step_outputs])
                        / loss_len
                    )
            elif isinstance(step_outputs[0], torch.Tensor):
                loss = {
                    "loss": sum([o.cpu().detach().numpy() for o in step_outputs])
                    / len(step_outputs)
                }
        else:
            loss = step_outputs
        return loss

    def training_step(self, batch, batch_idx):
        img, mask, pred_mask = (
            batch["img"],
            batch["gt_semantic_seg"],
            batch["pred_semantic_seg"],
        )
        output = self.net(img)

        # prepare some configs for inner optimization (can be configured in config)
        inner_iters = getattr(self.config, "inner_iters", 3)
        inner_lr = getattr(self.config, "inner_lr", 0.1)
        amplification = getattr(self.config, "amplification", 1.0)
        lambda_consistency = getattr(
            self.config, "lambda_consistency", 0.5
        )  # Consistency weight
        # target class for instance extraction; if not provided, treat any non-zero as foreground
        has_target = hasattr(self.config, "target_class")
        target_class = getattr(self.config, "target_class", 1)

        # build shifted-mask batch by optimizing per-instance deltas per sample
        N = mask.shape[0]
        H, W = mask.shape[-2], mask.shape[-1]
        device = mask.device
        mask_shifted_batch = torch.zeros_like(mask, dtype=mask.dtype, device=device)

        base_grid = create_base_grid(H, W, device, align_corners=False)  # [H, W, 2]

        for i in range(N):
            # extract sample logits for foreground channel
            logits_i = output["logits"][i : i + 1]  # [1, C, H, W]
            C = logits_i.shape[1]
            if C == 1:
                logits_fg = logits_i
            else:
                # assume building/foreground class index is target_class
                logits_fg = logits_i[:, target_class : target_class + 1, ...]

            # prepare binary foreground mask from label
            label_i = mask[i]
            if has_target:
                fg = (label_i == target_class).to(torch.float32)
            else:
                fg = (label_i > 0).to(torch.float32)

            # if no objects, just copy original label
            inst_map, masks, num_objects = get_instances(fg)
            if num_objects == 0:
                mask_shifted_batch[i] = mask[i]
                continue

            # Get current image for feature consistency
            img_i = img[i]  # [C, H, W]

            # initialize per-instance deltas (scalars) with gradients
            # initialize per-instance deltas (scalars) as Parameters and prepare an Adam optimizer
            deltas = []
            delta_params = []
            for k in range(num_objects):
                dx = torch.nn.Parameter(torch.randn(1, device=device) * 2.0 / W)  # ±2
                dy = torch.nn.Parameter(torch.randn(1, device=device) * 2.0 / H)
                deltas.append([dx, dy, masks[k].to(device)])
                delta_params.extend([dx, dy])

            # use Adam to optimize per-instance deltas
            inner_opt = torch.optim.Adam(delta_params, lr=inner_lr)

            # inner optimization loop (optimize deltas only)
            for _inner in range(inner_iters):
                inner_opt.zero_grad()

                # construct displacement field in pixel units (vectorized, differentiable)
                disp_field = torch.zeros(H, W, 2, device=device)
                consistency_loss = 0.0
                valid_objects = 0
                for dx, dy, mask_inst in deltas:
                    # create original mask for union computation
                    original_mask = mask_inst.clone()

                    # create shifted indices
                    y_indices, x_indices = torch.where(mask_inst)
                    new_y = torch.clamp(
                        (y_indices.float() - dy * H).round().long(), 0, H - 1
                    )
                    new_x = torch.clamp(
                        (x_indices.float() - dx * W).round().long(), 0, W - 1
                    )

                    # create shifted mask
                    shifted_mask = torch.zeros_like(mask_inst)
                    shifted_mask[new_y, new_x] = True

                    # compute union mask
                    union_mask = original_mask | shifted_mask

                    # apply displacements to union region
                    disp_field[..., 0][union_mask] = dx
                    disp_field[..., 1][union_mask] = dy

                    # Variance-based consistency
                    shifted_patch = img_i[:, shifted_mask].permute(
                        1, 0
                    )  # [N_pixels, C]
                    if shifted_patch.shape[0] < 10:
                        continue
                    consistency_loss += shifted_patch.var(dim=0).mean()
                    valid_objects += 1

                if valid_objects > 0:
                    consistency_loss /= valid_objects

                # normalize to [-1,1] for grid_sample (align_corners=False)
                disp_field_norm = disp_field.clone()
                disp_field_norm[..., 0] = disp_field[..., 0] * 2.0
                disp_field_norm[..., 1] = disp_field[..., 1] * 2.0

                sample_grid = base_grid + disp_field * 2  # [H, W, 2]
                sample_grid = torch.clamp(sample_grid, -1, 1)

                # apply grid_sample to the (soft) label
                labels_shifted = F.grid_sample(
                    fg.unsqueeze(0).unsqueeze(0),
                    sample_grid.unsqueeze(0),
                    mode="bilinear",
                    align_corners=False,
                    padding_mode="zeros",
                )  # [1,1,H,W]
                labels_shifted = torch.clamp(labels_shifted, 0.0, 1.0)

                # compute inner loss comparing network logits (detached) and shifted soft labels
                # both tensors should have same shape [1,1,H,W]
                seg_loss = F.binary_cross_entropy_with_logits(
                    logits_fg.detach(), labels_shifted
                )
                # Combine segmentation loss and consistency loss
                loss_inner = seg_loss + lambda_consistency * consistency_loss
                # print(f"Inner loss: {loss_inner.item()}")

                # backprop to deltas and step optimizer
                loss_inner.backward()

                # amplify gradients
                for param in delta_params:
                    if param.grad is not None:
                        param.grad *= amplification
                inner_opt.step()

                # DEBUG
                # if _inner % 2 == 0:
                #     print(f"Inner iter {_inner}: loss={loss_inner.item():.4f}")
                #     for idx, (dx, dy, _) in enumerate(deltas):
                #         print(f"  Object {idx}: dx={dx.item():.4f}, dy={dy.item():.4f}")

            # after inner optimization compute final shifted label (use last deltas)
            disp_field = torch.zeros(H, W, 2, device=device)
            for dx, dy, mask_inst in deltas:
                original_mask = mask_inst.clone()

                y_indices, x_indices = torch.where(mask_inst)
                new_y = torch.clamp(
                    (y_indices.float() - dy * H).round().long(), 0, H - 1
                )
                new_x = torch.clamp(
                    (x_indices.float() - dx * W).round().long(), 0, W - 1
                )

                shifted_mask = torch.zeros_like(mask_inst)
                shifted_mask[new_y, new_x] = True
                union_mask = original_mask | shifted_mask

                disp_field[..., 0][union_mask] = dx
                disp_field[..., 1][union_mask] = dy

            disp_field_norm = disp_field.clone()
            disp_field_norm[..., 0] = disp_field[..., 0] * 2.0
            disp_field_norm[..., 1] = disp_field[..., 1] * 2.0
            sample_grid = base_grid + disp_field_norm
            sample_grid = torch.clamp(sample_grid, -1, 1)
            labels_shifted_final = F.grid_sample(
                fg.unsqueeze(0).unsqueeze(0),
                sample_grid.unsqueeze(0),
                mode="bilinear",
                align_corners=False,
                padding_mode="zeros",
            )  # [1,1,H,W]

            # convert to original label dtype (round to hard labels)
            mask_shifted_batch[i] = torch.clamp(
                labels_shifted_final.squeeze(0).squeeze(0), 0.0, 1.0
            )

        # now compute final loss using shifted labels
        mask_shifted_long = mask_shifted_batch.round().to(mask.dtype)

        # visualize the shifted masks for debugging
        visualize_masks(
            output["logits"][0].cpu().detach().argmax(dim=0),
            mask[0],
            os.path.join(
                self.config.visualize_name, f"epoch_{self.current_epoch}", "train_debug"
            ),
            f"{batch['img_id'][0]}_OAM",
        )
        visualize_masks(
            mask_shifted_long[0],
            mask[0],
            os.path.join(
                self.config.visualize_name, f"epoch_{self.current_epoch}", "train_debug"
            ),
            f"{batch['img_id'][0]}_SAM",
        )
        # UnetFormerLoss expects (logits, labels) where logits can be (main, aux)
        if "logits_aux" in output:
            logits_input = (output["logits"], output["logits_aux"])
        else:
            logits_input = output["logits"]
        loss = self.loss(logits_input, mask_shifted_long)

        pre_mask = nn.Softmax(dim=1)(output["logits"]).argmax(dim=1)

        for i in range(mask_shifted_long.shape[0]):
            self.metrics_train.add_batch(
                mask_shifted_long[i].cpu().detach().numpy(), pre_mask[i].cpu().numpy()
            )

        self.training_step_outputs.append(loss.detach())
        return loss

    def on_train_epoch_start(self):
        if self.current_epoch < self.config.warmup_epoch:
            # During warmup we typically freeze the backbone and train the decoder/head.
            # Keep decoder parameters trainable so the model has something to update
            # during the warmup period.
            for param in self.net.backbone.parameters():
                param.requires_grad = False
            for param in self.net.decoder.parameters():
                param.requires_grad = True
        else:
            # After warmup, make sure backbone and decoder are trainable.
            for param in self.net.backbone.parameters():
                param.requires_grad = True
            for param in self.net.decoder.parameters():
                param.requires_grad = True

    def on_train_epoch_end(self):
        mIoU = np.nanmean(self.metrics_train.Intersection_over_Union())
        F1 = np.nanmean(self.metrics_train.F1())
        OA = np.nanmean(self.metrics_train.OA())
        iou_per_class = self.metrics_train.Intersection_over_Union()
        eval_value = {
            "mIoU": mIoU,
            "F1": F1,
            "OA": OA,
        }
        print("train:", eval_value)

        iou_value = {}
        for class_name, iou in zip(self.config.classes, iou_per_class):
            iou_value[class_name] = iou
        print("iou_value", iou_value)
        self.metrics_train.reset()
        log_dict = {
            "train_mIoU": mIoU,
            "train_F1": F1,
            "train_OA": OA,
            **self.get_avg_loss(self.training_step_outputs),
        }
        self.log_dict(log_dict, prog_bar=True)
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        img, mask, pred_mask = (
            batch["img"],
            batch["gt_semantic_seg"],
            batch["pred_semantic_seg"],
        )

        output = self(img)
        mask_32_list = []
        for img_id in batch["img_id"]:
            mask_path = os.path.join(
                "/root/workspace/zjj/xjd/data/segmentation/Turkey/Islahiye/pre/test/labels",
                f"{img_id}.png",
            )
            mask_img = cv2.resize(
                cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE), (512, 512)
            )
            mask_tensor = torch.from_numpy(mask_img).to(torch.float32) / 255.0
            mask_32_list.append(mask_tensor)
        mask_32 = torch.stack(mask_32_list, dim=0).to(mask.device)
        if "offset" in output:
            mask_offset = offset_tensor_v3(
                mask_32.unsqueeze(1), output["offset"], sample_mode="nearest"
            ).squeeze_(1)
        else:
            mask_offset = mask_32

        pre_mask = nn.Softmax(dim=1)(output["logits"]).argmax(dim=1)
        for i in range(mask.shape[0]):
            self.metrics_val.add_batch(mask[i].cpu().numpy(), pre_mask[i].cpu().numpy())

        # save the prediction result
        if (self.current_epoch + 1) % self.config.check_val_every_n_epoch == 0:
            for i in range(img.shape[0]):
                save_tensor_as_png(
                    pre_mask[i],
                    os.path.join(
                        self.config.visualize_name, f"epoch_{self.current_epoch}"
                    ),
                    batch["img_id"][i],
                )
                save_tensor_as_png(
                    mask[i],
                    os.path.join(
                        self.config.visualize_name, f"epoch_{self.current_epoch}"
                    ),
                    f"{batch['img_id'][i]}_mask",
                )
                visualize_masks(
                    pre_mask[i],
                    mask[i],
                    os.path.join(
                        self.config.visualize_name, f"epoch_{self.current_epoch}"
                    ),
                    f"{batch['img_id'][i]}_PAM",
                )
                if "offset" in output:
                    visualize_grayscale_as_pseudocolor(
                        (output["offset"][i, 0] - output["offset"][i, 0].min())
                        / (output["offset"][i, 0].max() - output["offset"][i, 0].min()),
                        os.path.join(
                            self.config.visualize_name, f"epoch_{self.current_epoch}"
                        ),
                        f"{batch['img_id'][i]}_x",
                        cmap="jet",
                        vmin=0,
                        vmax=1,
                    )
                    visualize_grayscale_as_pseudocolor(
                        (output["offset"][i, 1] - output["offset"][i, 1].min())
                        / (output["offset"][i, 1].max() - output["offset"][i, 1].min()),
                        os.path.join(
                            self.config.visualize_name, f"epoch_{self.current_epoch}"
                        ),
                        f"{batch['img_id'][i]}_y",
                        cmap="jet",
                        vmin=0,
                        vmax=1,
                    )
                    visualize_masks(
                        mask_offset[i],
                        mask[i],
                        os.path.join(
                            self.config.visualize_name, f"epoch_{self.current_epoch}"
                        ),
                        f"{batch['img_id'][i]}_MAO",
                    )
        # call loss depending on whether model/output provides offset
        if "offset" in output:
            loss = self.loss(
                (output["logits"], output.get("logits_aux")),
                output["offset"],
                mask,
                pred_mask,
                output.get("feature", None),
            )
            out = {"loss_val": loss["loss"]}
        else:
            if "logits_aux" in output:
                logits_input = (output["logits"], output["logits_aux"])
            else:
                logits_input = output["logits"]
            loss_val = self.loss(logits_input, mask)
            # UnetFormerLoss returns a scalar tensor
            out = {"loss_val": loss_val}
        self.validation_step_outputs.append(out)

        return out

    def on_validation_epoch_end(self):
        mIoU = np.nanmean(self.metrics_val.Intersection_over_Union())
        F1 = np.nanmean(self.metrics_val.F1())
        OA = np.nanmean(self.metrics_val.OA())
        iou_per_class = self.metrics_val.Intersection_over_Union()
        eval_value = {
            "mIoU": mIoU,
            "F1": F1,
            "OA": OA,
        }
        print("val:", eval_value)
        iou_value = {}
        for class_name, iou in zip(self.config.classes, iou_per_class):
            iou_value[class_name] = iou
        print("iou_value", iou_value)
        self.metrics_val.reset()
        log_dict = {
            "val_mIoU": mIoU,
            "val_F1": F1,
            "val_OA": OA,
            **self.get_avg_loss(self.validation_step_outputs),
        }
        self.log_dict(log_dict, prog_bar=True)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = self.config.optimizer
        lr_scheduler = self.config.lr_scheduler

        return [optimizer], [lr_scheduler]

    def train_dataloader(self):

        return self.config.train_loader

    def val_dataloader(self):

        return self.config.val_loader


# training
def main():
    args = get_args()
    config = py2cfg(args.config_path)
    seed_everything(42)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=config.save_top_k,
        monitor=config.monitor,
        save_last=config.save_last,
        mode=config.monitor_mode,
        dirpath=config.weights_path,
        filename=config.weights_name,
    )
    logger = CSVLogger("lightning_logs", name=config.log_name)

    # model = Supervision_Train.load_from_checkpoint(
    #     os.path.join(config.weights_path, config.test_weights_name + ".ckpt"),
    #     config=config,
    # )
    model = Supervision_Train(config)
    if config.pretrained_ckpt_path:
        model = Supervision_Train.load_from_checkpoint(
            config.pretrained_ckpt_path, config=config, strict=False
        )
        model.config = config

    trainer = pl.Trainer(
        devices=config.gpus,
        max_epochs=config.max_epoch,
        accelerator="auto",
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        callbacks=[checkpoint_callback],
        strategy="auto",
        logger=logger,
    )
    trainer.fit(model=model, ckpt_path=config.resume_ckpt_path)


if __name__ == "__main__":
    main()
