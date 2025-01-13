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

        self.loss_seg = config.loss_seg
        self.loss_emd = config.loss_emd
        self.alpha = config.alpha

        self.xbd_metrics_train = Evaluator(num_class=config.num_classes)
        self.xbd_metrics_val = Evaluator(num_class=config.num_classes)
        self.teq_metrics_train = Evaluator(num_class=config.num_classes)
        self.teq_metrics_val = Evaluator(num_class=config.num_classes)

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
        xbd_img, xbd_mask, teq_img, teq_mask = (
            batch["xbd_img"],
            batch["xbd_gt_semantic_seg"],
            batch["teq_img"],
            batch["teq_gt_semantic_seg"],
        )

        xbd_output = self.net(xbd_img)
        loss_seg = self.loss_seg(
            (xbd_output["logits"], xbd_output["logits_aux"]), xbd_mask
        )
        teq_output = self.net(teq_img)
        loss_emd = self.loss_emd(
            teq_output["logits"],
            teq_mask,
            scale_factor=0.5,
            eps=0.001,
            max_iter=500,
            reduction="mean",
        )
        loss = loss_seg + self.alpha * loss_emd

        xbd_pre_mask = nn.Softmax(dim=1)(xbd_output["logits"]).argmax(dim=1)
        teq_pre_mask = nn.Softmax(dim=1)(teq_output["logits"]).argmax(dim=1)
        for i in range(xbd_mask.shape[0]):
            self.xbd_metrics_train.add_batch(
                xbd_mask[i].cpu().numpy(), xbd_pre_mask[i].cpu().numpy()
            )
            self.teq_metrics_train.add_batch(
                teq_mask[i].cpu().numpy(), teq_pre_mask[i].cpu().numpy()
            )
        out = {
            "loss": loss,
            "ls_seg": loss_seg,
            "ls_emd": loss_emd,
        }
        self.training_step_outputs.append(out)
        return out

    def on_train_epoch_end(self):
        xbd_mIoU = np.nanmean(self.xbd_metrics_train.Intersection_over_Union())
        xbd_F1 = np.nanmean(self.xbd_metrics_train.F1())
        teq_mIoU = np.nanmean(self.teq_metrics_train.Intersection_over_Union())
        teq_F1 = np.nanmean(self.teq_metrics_train.F1())
        xbd_OA = np.nanmean(self.xbd_metrics_train.OA())
        xbd_iou_per_class = self.xbd_metrics_train.Intersection_over_Union()
        teq_OA = np.nanmean(self.teq_metrics_train.OA())
        teq_iou_per_class = self.teq_metrics_train.Intersection_over_Union()
        eval_value = {
            "xbd_mIoU": xbd_mIoU,
            "xbd_F1": xbd_F1,
            "xbd_OA": xbd_OA,
            "teq_mIoU": teq_mIoU,
            "teq_F1": teq_F1,
            "teq_OA": teq_OA,
        }
        print("train:", eval_value)

        xbd_iou_value = {}
        teq_iou_value = {}
        for class_name, xbd_iou, teq_iou in zip(
            self.config.classes, xbd_iou_per_class, teq_iou_per_class
        ):
            xbd_iou_value[class_name] = xbd_iou
            teq_iou_value[class_name] = teq_iou
        print("xbd_iou_value", xbd_iou_value, "teq_iou_value", teq_iou_value)
        self.xbd_metrics_train.reset()
        self.teq_metrics_train.reset()
        log_dict = {
            "x_t_mIoU": xbd_mIoU,
            "x_t_F1": xbd_F1,
            "x_t_OA": xbd_OA,
            "t_t_mIoU": teq_mIoU,
            "t_t_F1": teq_F1,
            "t_t_OA": teq_OA,
            **self.get_avg_loss(self.training_step_outputs),
        }
        self.log_dict(log_dict, prog_bar=True)
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        xbd_img, xbd_mask, teq_img, teq_mask = (
            batch["xbd_img"],
            batch["xbd_gt_semantic_seg"],
            batch["teq_img"],
            batch["teq_gt_semantic_seg"],
        )

        xbd_output = self(xbd_img)
        teq_output = self(teq_img)
        xbd_pre_mask = nn.Softmax(dim=1)(xbd_output["logits"]).argmax(dim=1)
        teq_pre_mask = nn.Softmax(dim=1)(teq_output["logits"]).argmax(dim=1)
        for i in range(xbd_mask.shape[0]):
            self.xbd_metrics_val.add_batch(
                xbd_mask[i].cpu().numpy(), xbd_pre_mask[i].cpu().numpy()
            )
            self.teq_metrics_val.add_batch(
                teq_mask[i].cpu().numpy(), teq_pre_mask[i].cpu().numpy()
            )

        loss_seg_val = self.loss_seg(xbd_output["logits"], xbd_mask)
        loss_emd_val = self.loss_emd(
            teq_output["logits"],
            teq_mask,
            scale_factor=0.5,
            eps=0.001,
            max_iter=500,
            reduction="mean",
        )
        loss_val = loss_seg_val + self.alpha * loss_emd_val

        out = {
            "loss_val": loss_val,
            "ls_seg_v": loss_seg_val,
            "ls_emd_v": loss_emd_val,
        }
        self.validation_step_outputs.append(out)

        return out

    def on_validation_epoch_end(self):
        xbd_mIoU = np.nanmean(self.xbd_metrics_val.Intersection_over_Union())
        xbd_F1 = np.nanmean(self.xbd_metrics_val.F1())
        teq_mIoU = np.nanmean(self.teq_metrics_val.Intersection_over_Union())
        teq_F1 = np.nanmean(self.teq_metrics_val.F1())
        xbd_OA = np.nanmean(self.xbd_metrics_val.OA())
        xbd_iou_per_class = self.xbd_metrics_val.Intersection_over_Union()
        teq_OA = np.nanmean(self.teq_metrics_val.OA())
        teq_iou_per_class = self.teq_metrics_val.Intersection_over_Union()
        eval_value = {
            "xbd_mIoU": xbd_mIoU,
            "xbd_F1": xbd_F1,
            "xbd_OA": xbd_OA,
            "teq_mIoU": teq_mIoU,
            "teq_F1": teq_F1,
            "teq_OA": teq_OA,
        }
        print("val:", eval_value)
        xbd_iou_value = {}
        teq_iou_value = {}
        for class_name, xbd_iou, teq_iou in zip(
            self.config.classes, xbd_iou_per_class, teq_iou_per_class
        ):
            xbd_iou_value[class_name] = xbd_iou
            teq_iou_value[class_name] = teq_iou
        print("xbd_iou_value", xbd_iou_value, "teq_iou_value", teq_iou_value)
        self.xbd_metrics_val.reset()
        self.teq_metrics_val.reset()
        log_dict = {
            "x_v_mIoU": xbd_mIoU,
            "x_v_F1": xbd_F1,
            "x_v_OA": xbd_OA,
            "t_v_mIoU": teq_mIoU,
            "t_v_F1": teq_F1,
            "t_v_OA": teq_OA,
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

    model = Supervision_Train.load_from_checkpoint(
        os.path.join(config.weights_path, config.test_weights_name + ".ckpt"),
        config=config,
    )
    # model = Supervision_Train(config)
    if config.pretrained_ckpt_path:
        model = Supervision_Train.load_from_checkpoint(
            config.pretrained_ckpt_path, config=config
        )

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
