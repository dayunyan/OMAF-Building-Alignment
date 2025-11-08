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
from tools.metric import Evaluator
from pytorch_lightning.loggers import CSVLogger
import random
from tools.uda import prob_2_entropy
from tools.visual import (
    save_tensor_as_png,
    visualize_aligned_mask,
    visualize_masks,
    visualize_grayscale_as_pseudocolor,
)


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
        # self.automatic_optimization = False
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
        loss_names = step_outputs[0].keys()
        loss_len = len(step_outputs)
        loss = {}
        for ln in loss_names:
            loss[ln] = (
                sum([o[ln].cpu().detach().numpy() for o in step_outputs]) / loss_len
            )

        return loss

    def training_step(self, batch, batch_idx):
        img, mask = (
            batch["img"],
            batch["gt_semantic_seg"],
        )

        output = self.net(img)
        loss = self.loss(output["logits"], mask)

        pre_mask = nn.Softmax(dim=1)(output["logits"]).argmax(dim=1)
        for i in range(mask.shape[0]):
            self.metrics_train.add_batch(
                mask[i].cpu().numpy(), pre_mask[i].cpu().numpy()
            )

        self.training_step_outputs.append({"loss_tra": loss})
        return loss

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
        img, mask, gt = (batch["img"], batch["gt_semantic_seg"], batch["gt"])

        output = self(img)

        pre_mask = nn.Softmax(dim=1)(output["logits"]).argmax(dim=1)
        for i in range(gt.shape[0]):
            self.metrics_val.add_batch(gt[i].cpu().numpy(), pre_mask[i].cpu().numpy())

        if (self.current_epoch + 1) % self.config.check_val_every_n_epoch == 0:
            for i in range(img.shape[0]):
                visualize_masks(
                    pre_mask[i],
                    gt[i],
                    os.path.join(
                        self.config.visualize_name, f"epoch_{self.current_epoch}"
                    ),
                    f"{batch['img_id'][i]}_PAM",
                )

        loss = self.loss(output["logits"], gt)
        out = {
            "loss_val": loss,
        }
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
