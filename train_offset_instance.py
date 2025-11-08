import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from tools.cfg import py2cfg
import os
import torch
from torch import nn
import torch.nn.functional as F
import cv2
import numpy as np
import argparse
from pathlib import Path
from pytorch_lightning.loggers import CSVLogger
import random
from tools.uda import prob_2_entropy
from tools.visual import (
    save_tensor_as_png,
    visualize_masks,
    visualize_grayscale_as_pseudocolor,
    visualize_correction_overlay,
    visualize_gt_vs_pred_overlay,
    visualize_emi_vs_pred_overlay,
)


def calculate_mae(preds, targets, conf=None):
    if conf is None:
        conf = torch.ones_like(targets)
    return (torch.abs(preds - targets).sum(dim=1) * conf).sum() / (conf.sum() + 1e-8)


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
        self.net = config.net
        self.loss = config.loss

        self.training_step_outputs = []
        self.validation_step_outputs = []

    def forward(self, image, bboxes_list, centroids_list):
        # 仅在推理时使用。
        # 注意：推理时的输入也必须是 (image, bboxes_list, centroids_list)
        return self.net(image, bboxes_list, centroids_list)

    def training_step(self, batch, batch_idx):
        img = batch["img"]
        bboxes_list = batch["bboxes"]
        centroids_list = batch["centroids"]
        gt_offsets_list = batch["gt_offsets"]
        gt_confidences_list = batch["gt_confidences"]
        # print(
        #     f"{torch.cat(bboxes_list, dim=0).shape=}, {torch.cat(gt_offsets_list, dim=0).shape=}"
        # )

        # 1. 模型预测
        # predicted_offsets: [N_total, 2]
        predicted_offsets = self.net(img, bboxes_list, centroids_list)

        # 2. 准备 Loss 的 GT
        # [N_total, 2]
        gt_offsets = torch.cat(gt_offsets_list, dim=0)
        # [N_total, 1]
        gt_confidences = torch.cat(gt_confidences_list, dim=0)
        # print(
        #     f"{predicted_offsets.shape=}, {gt_offsets.shape=}, {gt_confidences.shape=}"
        # )

        # 3. 计算 Loss
        loss = self.loss(predicted_offsets, gt_offsets, gt_confidences)

        # 4. 记录指标
        # 计算无加权的 MAE 以进行监控
        mae = calculate_mae(predicted_offsets.detach(), gt_offsets, gt_confidences)

        step_out = {"loss": loss, "mae": mae}
        self.training_step_outputs.append(step_out)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_MAE", mae, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_train_epoch_end(self):
        # 我们可以聚合所有 step 的平均值
        avg_loss = torch.stack([x["loss"] for x in self.training_step_outputs]).mean()
        avg_mae = torch.stack([x["mae"] for x in self.training_step_outputs]).mean()
        print(f"Train Epoch End: Avg Loss: {avg_loss}, Avg MAE: {avg_mae}")
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        img = batch["img"]
        bboxes_list = batch["bboxes"]
        centroids_list = batch["centroids"]
        gt_offsets_list = batch["gt_offsets"]
        gt_confidences_list = batch["gt_confidences"]
        mask = batch["mask"]
        gt_mask = batch.get("gt_mask", None)
        img_ids = batch["img_id"]

        predicted_offsets = self.net(img, bboxes_list, centroids_list)
        gt_offsets = torch.cat(gt_offsets_list, dim=0)
        gt_confidences = torch.cat(gt_confidences_list, dim=0)

        loss = self.loss(predicted_offsets, gt_offsets, gt_confidences)
        mae = calculate_mae(predicted_offsets, gt_offsets, gt_confidences)

        step_out = {"val_loss": loss, "val_MAE": mae}
        self.validation_step_outputs.append(step_out)

        # 我们需要将 [N_total, 2] 的预测拆分回 [B, (N_i, 2)] 的列表
        num_instances_per_image = [len(b) for b in bboxes_list]
        pred_offsets_list = torch.split(
            predicted_offsets.detach(), num_instances_per_image, dim=0
        )

        vis_save_path = os.path.join(
            self.config.visualize_name, f"epoch_{self.current_epoch}"
        )
        for i in range(len(img_ids)):
            if len(bboxes_list[i]) == 0:  # 跳过没有实例的图像
                continue

            # 可视化 1: 偏移掩码(蓝) vs 预测修正掩码(红)
            visualize_correction_overlay(
                image_tensor=img[i],
                shifted_mask_tensor=mask[i],
                bboxes_tensor=bboxes_list[i],
                pred_offsets_ratio_tensor=pred_offsets_list[i],
                save_path=vis_save_path,
                file_name=f"{img_ids[i]}_pred_correction",
            )

            # 可视化 2: 预测修正(红) vs 估计修正(蓝)
            visualize_emi_vs_pred_overlay(
                image_tensor=img[i],
                shifted_mask_tensor=mask[i],
                bboxes_tensor=bboxes_list[i],
                pred_offsets_ratio_tensor=pred_offsets_list[i],
                gt_offsets_ratio_tensor=gt_offsets_list[i],
                save_path=vis_save_path,
                file_name=f"{img_ids[i]}_emi_vs_pred",
            )

            # 可视化 3: 预测修正(红) vs GT(绿)
            visualize_gt_vs_pred_overlay(
                image_tensor=img[i],
                shifted_mask_tensor=mask[i],
                gt_mask_tensor=gt_mask[i],
                bboxes_tensor=bboxes_list[i],
                pred_offsets_ratio_tensor=pred_offsets_list[i],
                save_path=vis_save_path,
                file_name=f"{img_ids[i]}_gt_vs_pred",
            )

        return step_out

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(
            [x["val_loss"] for x in self.validation_step_outputs]
        ).mean()
        avg_mae = torch.stack(
            [x["val_MAE"] for x in self.validation_step_outputs]
        ).mean()

        log_dict = {
            "val_loss": avg_loss,
            "val_MAE": avg_mae,  # 这是我们真正监控的指标
        }
        print(f"Validation Epoch End: Avg Loss: {avg_loss}, Avg MAE: {avg_mae}")
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
