from typing import Any, Optional

import einops
import torch
from lightning import LightningModule
from torch.nn import L1Loss, SmoothL1Loss
from torchmetrics import (
    MaxMetric,
    MeanMetric,
    MinMetric,
    PearsonCorrCoef,
)

from src.utils.unitary_linear_norm import unitary_norm_inv

# ======================== joint regression module for room feature encoder only========================


class JointRegressorModuleEncoder(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        optim_cfg: Optional[dict],
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    ) -> None:
        """Initialize a rirRegressorModule

        :param net_denoiser_path: The path to the pretrained denoiser model.
        :param net_rirRegressor: The rirRegressor model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore="net")

        self.regressor = net

        # loss function
        self.joint_loss_train = SmoothL1Loss()

        self.joint_loss_val = L1Loss()
        self.joint_corrcoef_val = PearsonCorrCoef()

        self.joint_loss_test = L1Loss()
        self.joint_corrcoef_test = PearsonCorrCoef()

        self.sti_weight = self.hparams.optim_cfg.sti_weight
        self.alcons_weight = self.hparams.optim_cfg.alcons_weight
        self.tr_weight = self.hparams.optim_cfg.tr_weight
        self.edt_weight = self.hparams.optim_cfg.edt_weight
        self.c80_weight = self.hparams.optim_cfg.c80_weight
        self.c50_weight = self.hparams.optim_cfg.c50_weight
        self.d50_weight = self.hparams.optim_cfg.d50_weight
        self.ts_weight = self.hparams.optim_cfg.ts_weight
        self.volume_weight = self.hparams.optim_cfg.volume_weight
        self.dist_src_weight = self.hparams.optim_cfg.dist_src_weight

        # for averaging loss across batches
        self.train_loss_sti = MeanMetric()
        self.val_loss_sti = MeanMetric()
        self.test_loss_sti = MeanMetric()

        self.train_loss_alcons = MeanMetric()
        self.val_loss_alcons = MeanMetric()
        self.test_loss_alcons = MeanMetric()

        self.train_loss_tr = MeanMetric()
        self.val_loss_tr = MeanMetric()
        self.test_loss_tr = MeanMetric()

        self.train_loss_edt = MeanMetric()
        self.val_loss_edt = MeanMetric()
        self.test_loss_edt = MeanMetric()

        self.train_loss_c80 = MeanMetric()
        self.val_loss_c80 = MeanMetric()
        self.test_loss_c80 = MeanMetric()

        self.train_loss_c50 = MeanMetric()
        self.val_loss_c50 = MeanMetric()
        self.test_loss_c50 = MeanMetric()

        self.train_loss_d50 = MeanMetric()
        self.val_loss_d50 = MeanMetric()
        self.test_loss_d50 = MeanMetric()

        self.train_loss_ts = MeanMetric()
        self.val_loss_ts = MeanMetric()
        self.test_loss_ts = MeanMetric()

        self.train_loss_volume = MeanMetric()
        self.val_loss_volume = MeanMetric()
        self.test_loss_volume = MeanMetric()

        self.train_loss_dist_src = MeanMetric()
        self.val_loss_dist_src = MeanMetric()
        self.test_loss_dist_src = MeanMetric()

        # for tracking validation correlation coefficient of RAP and RGP
        self.val_corrcoef_sti = MeanMetric()
        self.val_corrcoef_alcons = MeanMetric()
        self.val_corrcoef_tr = MeanMetric()
        self.val_corrcoef_edt = MeanMetric()
        self.val_corrcoef_c80 = MeanMetric()
        self.val_corrcoef_c50 = MeanMetric()
        self.val_corrcoef_d50 = MeanMetric()
        self.val_corrcoef_ts = MeanMetric()
        self.val_corrcoef_volume = MeanMetric()
        self.val_corrcoef_dist_src = MeanMetric()

        # for tracking evaluation correlation coefficient of RAP and RGP
        self.test_corrcoef_sti = MeanMetric()
        self.test_corrcoef_alcons = MeanMetric()
        self.test_corrcoef_tr = MeanMetric()
        self.test_corrcoef_edt = MeanMetric()
        self.test_corrcoef_c80 = MeanMetric()
        self.test_corrcoef_c50 = MeanMetric()
        self.test_corrcoef_d50 = MeanMetric()
        self.test_corrcoef_ts = MeanMetric()
        self.test_corrcoef_volume = MeanMetric()
        self.test_corrcoef_dist_src = MeanMetric()

        # for tracking best so far validation best metrics
        self.val_loss_best_sti = MinMetric()
        self.val_loss_best_alcons = MinMetric()
        self.val_loss_best_tr = MinMetric()
        self.val_loss_best_edt = MinMetric()
        self.val_loss_best_c80 = MinMetric()
        self.val_loss_best_c50 = MinMetric()
        self.val_loss_best_d50 = MinMetric()
        self.val_loss_best_ts = MinMetric()
        self.val_loss_best_volume = MinMetric()
        self.val_loss_best_dist_src = MinMetric()

        # for tracking best of correlation coefficient
        self.val_corrcoef_best_sti = MaxMetric()
        self.val_corrcoef_best_alcons = MaxMetric()
        self.val_corrcoef_best_tr = MaxMetric()
        self.val_corrcoef_best_edt = MaxMetric()
        self.val_corrcoef_best_c80 = MaxMetric()
        self.val_corrcoef_best_c50 = MaxMetric()
        self.val_corrcoef_best_d50 = MaxMetric()
        self.val_corrcoef_best_ts = MaxMetric()
        self.val_corrcoef_best_volume = MaxMetric()
        self.val_corrcoef_best_dist_src = MaxMetric()

    def forward(
        self,
        source: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Perform a forward pass through freezed denoiser and rirRegressor.

        :param x: A tensor of waveform
        :return: A tensor of estimated Th, Tt, volume, distSrc, azimuthSrc, elevationSrc.
        """
        net_output = self.regressor(source, padding_mask)

        return net_output

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss_sti.reset()
        self.val_loss_alcons.reset()
        self.val_loss_tr.reset()
        self.val_loss_edt.reset()
        self.val_loss_c80.reset()
        self.val_loss_c50.reset()
        self.val_loss_d50.reset()
        self.val_loss_ts.reset()
        self.val_loss_volume.reset()
        self.val_loss_dist_src.reset()

        self.val_corrcoef_sti.reset()
        self.val_corrcoef_alcons.reset()
        self.val_corrcoef_tr.reset()
        self.val_corrcoef_edt.reset()
        self.val_corrcoef_c80.reset()
        self.val_corrcoef_c50.reset()
        self.val_corrcoef_d50.reset()
        self.val_corrcoef_ts.reset()
        self.val_corrcoef_volume.reset()
        self.val_corrcoef_dist_src.reset()

        self.val_loss_best_sti.reset()
        self.val_loss_best_alcons.reset()
        self.val_loss_best_tr.reset()
        self.val_loss_best_edt.reset()
        self.val_loss_best_c80.reset()
        self.val_loss_best_c50.reset()
        self.val_loss_best_d50.reset()
        self.val_loss_best_ts.reset()
        self.val_loss_best_volume.reset()
        self.val_loss_best_dist_src.reset()

        self.val_corrcoef_best_sti.reset()
        self.val_corrcoef_best_alcons.reset()
        self.val_corrcoef_best_tr.reset()
        self.val_corrcoef_best_edt.reset()
        self.val_corrcoef_best_c80.reset()
        self.val_corrcoef_best_c50.reset()
        self.val_corrcoef_best_d50.reset()
        self.val_corrcoef_best_ts.reset()
        self.val_corrcoef_best_volume.reset()
        self.val_corrcoef_best_dist_src.reset()

    def model_step(
        self,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a dict) containing the input tensor
            of raw, target Th, target Tt, target volume, target distSrc,
            target azimuthSrc, target elevationSrc.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        # get target labels
        sti = batch["groundtruth"]["sti"]
        alcons = batch["groundtruth"]["alcons"]
        tr = batch["groundtruth"]["tr"]
        edt = batch["groundtruth"]["edt"]
        c80 = batch["groundtruth"]["c80"]
        c50 = batch["groundtruth"]["c50"]
        d50 = batch["groundtruth"]["d50"]
        ts = batch["groundtruth"]["ts"]
        volume = batch["groundtruth"]["volume"]
        dist_src = batch["groundtruth"]["dist_src"]

        # network forward pass
        net_output = self.forward(**batch["net_input"])
        sti_hat = net_output["sti_hat"]
        alcons_hat = net_output["alcons_hat"]
        tr_hat = net_output["tr_hat"]
        edt_hat = net_output["edt_hat"]
        c80_hat = net_output["c80_hat"]
        c50_hat = net_output["c50_hat"]
        d50_hat = net_output["d50_hat"]
        ts_hat = net_output["ts_hat"]
        volume_hat = net_output["volume_hat"]
        dist_src_hat = net_output["dist_src_hat"]
        padding_mask = net_output["padding_mask"]

        if padding_mask is not None and padding_mask.any():
            reverse_padding_mask = padding_mask.logical_not()
            # ---------- padding mask handling ----------
            # Repeat the target labels to match the output shape
            sti = einops.repeat(sti, "b -> b t", t=sti_hat.shape[1])
            alcons = einops.repeat(alcons, "b -> b t", t=alcons_hat.shape[1])
            tr = einops.repeat(tr, "b -> b t", t=tr_hat.shape[1])
            edt = einops.repeat(edt, "b -> b t", t=edt_hat.shape[1])
            c80 = einops.repeat(c80, "b -> b t", t=c80_hat.shape[1])
            c50 = einops.repeat(c50, "b -> b t", t=c50_hat.shape[1])
            d50 = einops.repeat(d50, "b -> b t", t=d50_hat.shape[1])
            ts = einops.repeat(ts, "b -> b t", t=ts_hat.shape[1])
            volume = einops.repeat(volume, "b -> b t", t=volume_hat.shape[1])
            dist_src = einops.repeat(dist_src, "b -> b t", t=dist_src_hat.shape[1])

            # Collapse the time dimension
            sti_hat = sti_hat.masked_select(reverse_padding_mask)
            alcons_hat = alcons_hat.masked_select(reverse_padding_mask)
            tr_hat = tr_hat.masked_select(reverse_padding_mask)
            edt_hat = edt_hat.masked_select(reverse_padding_mask)
            c80_hat = c80_hat.masked_select(reverse_padding_mask)
            c50_hat = c50_hat.masked_select(reverse_padding_mask)
            d50_hat = d50_hat.masked_select(reverse_padding_mask)
            ts_hat = ts_hat.masked_select(reverse_padding_mask)
            volume_hat = volume_hat.masked_select(reverse_padding_mask)
            dist_src_hat = dist_src_hat.masked_select(reverse_padding_mask)

            sti = sti.masked_select(reverse_padding_mask)
            alcons = alcons.masked_select(reverse_padding_mask)
            tr = tr.masked_select(reverse_padding_mask)
            edt = edt.masked_select(reverse_padding_mask)
            c80 = c80.masked_select(reverse_padding_mask)
            c50 = c50.masked_select(reverse_padding_mask)
            d50 = d50.masked_select(reverse_padding_mask)
            ts = ts.masked_select(reverse_padding_mask)
            volume = volume.masked_select(reverse_padding_mask)
            dist_src = dist_src.masked_select(reverse_padding_mask)

        else:
            # Repeat the target labels to match the output shape
            sti = einops.repeat(sti, "b -> b t", t=sti_hat.shape[1])
            alcons = einops.repeat(alcons, "b -> b t", t=alcons_hat.shape[1])
            tr = einops.repeat(tr, "b -> b t", t=tr_hat.shape[1])
            edt = einops.repeat(edt, "b -> b t", t=edt_hat.shape[1])
            c80 = einops.repeat(c80, "b -> b t", t=c80_hat.shape[1])
            c50 = einops.repeat(c50, "b -> b t", t=c50_hat.shape[1])
            d50 = einops.repeat(d50, "b -> b t", t=d50_hat.shape[1])
            ts = einops.repeat(ts, "b -> b t", t=ts_hat.shape[1])
            volume = einops.repeat(volume, "b -> b t", t=volume_hat.shape[1])
            dist_src = einops.repeat(dist_src, "b -> b t", t=dist_src_hat.shape[1])

        loss_sti = self.joint_loss_train(sti_hat, sti)
        loss_alcons = self.joint_loss_train(alcons_hat, alcons)
        loss_tr = self.joint_loss_train(tr_hat, tr)
        loss_edt = self.joint_loss_train(edt_hat, edt)
        loss_c80 = self.joint_loss_train(c80_hat, c80)
        loss_c50 = self.joint_loss_train(c50_hat, c50)
        loss_d50 = self.joint_loss_train(d50_hat, d50)
        loss_ts = self.joint_loss_train(ts_hat, ts)
        loss_volume = self.joint_loss_train(volume_hat, volume)
        loss_dist_src = self.joint_loss_train(dist_src_hat, dist_src)

        return {
            "loss_sti": loss_sti,
            "loss_alcons": loss_alcons,
            "loss_tr": loss_tr,
            "loss_edt": loss_edt,
            "loss_c80": loss_c80,
            "loss_c50": loss_c50,
            "loss_d50": loss_d50,
            "loss_ts": loss_ts,
            "loss_volume": loss_volume,
            "loss_dist_src": loss_dist_src,
        }

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of raw,
            target Th, target Tt, target volume, target distSrc, target azimuthSrc,
            target elevationSrc.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """

        output = self.model_step(batch)

        loss_sti = output["loss_sti"]
        loss_alcons = output["loss_alcons"]
        loss_tr = output["loss_tr"]
        loss_edt = output["loss_edt"]
        loss_c80 = output["loss_c80"]
        loss_c50 = output["loss_c50"]
        loss_d50 = output["loss_d50"]
        loss_ts = output["loss_ts"]
        loss_volume = output["loss_volume"]
        loss_dist_src = output["loss_dist_src"]

        # ------------------- other parameters -------------------

        # update metrics of Th, Tt, volume, distSrc, azimuthSrc, elevationSrc loss
        self.train_loss_sti(loss_sti)
        self.train_loss_alcons(loss_alcons)
        self.train_loss_tr(loss_tr)
        self.train_loss_edt(loss_edt)
        self.train_loss_c80(loss_c80)
        self.train_loss_c50(loss_c50)
        self.train_loss_d50(loss_d50)
        self.train_loss_ts(loss_ts)
        self.train_loss_volume(loss_volume)
        self.train_loss_dist_src(loss_dist_src)
        # self.train_loss_azimuth(loss_azimuth)
        # self.train_loss_elevation(loss_elevation)

        # ------------------- total loss -------------------
        total_loss = (
            self.sti_weight * loss_sti
            + self.alcons_weight * loss_alcons
            + self.tr_weight * loss_tr
            + self.edt_weight * loss_edt
            + self.c80_weight * loss_c80
            + self.c50_weight * loss_c50
            + self.d50_weight * loss_d50
            + self.ts_weight * loss_ts
            + self.volume_weight * loss_volume
            + self.dist_src_weight * loss_dist_src
        )

        # ------------------- log metrics -------------------
        self.log(
            "train/loss/sti",
            self.train_loss_sti,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "train/loss/alcons",
            self.train_loss_alcons,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "train/loss/tr",
            self.train_loss_tr,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "train/loss/edt",
            self.train_loss_edt,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "train/loss/c80",
            self.train_loss_c80,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "train/loss/c50",
            self.train_loss_c50,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "train/loss/d50",
            self.train_loss_d50,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "train/loss/ts",
            self.train_loss_ts,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "train/loss/volume",
            self.train_loss_volume,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "train/loss/dist_src",
            self.train_loss_dist_src,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "train/loss/total",
            total_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        # return loss or backpropagation will fail
        return total_loss

    def on_train_epoch_end(self) -> None:
        """Lightning hook that is called when a training epoch ends."""
        pass

    def validation_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a dict) containing the input tensor of raw,
            target Th, target Tt, target volume, target distSrc, target azimuthSrc,
            target elevationSrc.
        :param batch_idx: The index of the current batch.
        """
        # get target labels
        sti = batch["groundtruth"]["sti"]
        alcons = batch["groundtruth"]["alcons"]
        tr = batch["groundtruth"]["tr"]
        edt = batch["groundtruth"]["edt"]
        c80 = batch["groundtruth"]["c80"]
        c50 = batch["groundtruth"]["c50"]
        d50 = batch["groundtruth"]["d50"]
        ts = batch["groundtruth"]["ts"]
        volume = batch["groundtruth"]["volume"]
        dist_src = batch["groundtruth"]["dist_src"]

        # network forward pass
        net_output = self.forward(**batch["net_input"])
        sti_hat = net_output["sti_hat"]
        alcons_hat = net_output["alcons_hat"]
        tr_hat = net_output["tr_hat"]
        edt_hat = net_output["edt_hat"]
        c80_hat = net_output["c80_hat"]
        c50_hat = net_output["c50_hat"]
        d50_hat = net_output["d50_hat"]
        ts_hat = net_output["ts_hat"]
        volume_hat = net_output["volume_hat"]
        dist_src_hat = net_output["dist_src_hat"]
        padding_mask = net_output["padding_mask"]

        if padding_mask is not None and padding_mask.any():
            reverse_padding_mask = padding_mask.logical_not()
            # ---------- padding mask handling ----------
            # Collapse the time dimension
            sti_hat = (sti_hat * reverse_padding_mask).sum(
                dim=1
            ) / reverse_padding_mask.sum(dim=1)
            alcons_hat = (alcons_hat * reverse_padding_mask).sum(
                dim=1
            ) / reverse_padding_mask.sum(dim=1)
            tr_hat = (tr_hat * reverse_padding_mask).sum(
                dim=1
            ) / reverse_padding_mask.sum(dim=1)
            edt_hat = (edt_hat * reverse_padding_mask).sum(
                dim=1
            ) / reverse_padding_mask.sum(dim=1)
            c80_hat = (c80_hat * reverse_padding_mask).sum(
                dim=1
            ) / reverse_padding_mask.sum(dim=1)
            c50_hat = (c50_hat * reverse_padding_mask).sum(
                dim=1
            ) / reverse_padding_mask.sum(dim=1)
            d50_hat = (d50_hat * reverse_padding_mask).sum(
                dim=1
            ) / reverse_padding_mask.sum(dim=1)
            ts_hat = (ts_hat * reverse_padding_mask).sum(
                dim=1
            ) / reverse_padding_mask.sum(dim=1)
            volume_hat = (volume_hat * reverse_padding_mask).sum(
                dim=1
            ) / reverse_padding_mask.sum(dim=1)
            dist_src_hat = (dist_src_hat * reverse_padding_mask).sum(
                dim=1
            ) / reverse_padding_mask.sum(dim=1)

        else:
            # Collapse the time dimension
            sti_hat = sti_hat.mean(dim=1)
            alcons_hat = alcons_hat.mean(dim=1)
            tr_hat = tr_hat.mean(dim=1)
            edt_hat = edt_hat.mean(dim=1)
            c80_hat = c80_hat.mean(dim=1)
            c50_hat = c50_hat.mean(dim=1)
            d50_hat = d50_hat.mean(dim=1)
            ts_hat = ts_hat.mean(dim=1)
            volume_hat = volume_hat.mean(dim=1)
            dist_src_hat = dist_src_hat.mean(dim=1)

        # update metrics of Th, Tt, volume, distSrc loss and correlation coefficient
        loss_sti = self.joint_loss_val(sti_hat, sti)
        loss_alcons = self.joint_loss_val(alcons_hat, alcons)
        loss_tr = self.joint_loss_val(tr_hat, tr)
        loss_edt = self.joint_loss_val(edt_hat, edt)
        loss_c80 = self.joint_loss_val(c80_hat, c80)
        loss_c50 = self.joint_loss_val(c50_hat, c50)
        loss_d50 = self.joint_loss_val(d50_hat, d50)
        loss_ts = self.joint_loss_val(ts_hat, ts)
        loss_volume = self.joint_loss_val(volume_hat, volume)
        loss_dist_src = self.joint_loss_val(dist_src_hat, dist_src)

        corrcoef_sti = self.joint_corrcoef_val(sti_hat, sti)
        corrcoef_alcons = self.joint_corrcoef_val(alcons_hat, alcons)
        corrcoef_tr = self.joint_corrcoef_val(tr_hat, tr)
        corrcoef_edt = self.joint_corrcoef_val(edt_hat, edt)
        corrcoef_c80 = self.joint_corrcoef_val(c80_hat, c80)
        corrcoef_c50 = self.joint_corrcoef_val(c50_hat, c50)
        corrcoef_d50 = self.joint_corrcoef_val(d50_hat, d50)
        corrcoef_ts = self.joint_corrcoef_val(ts_hat, ts)
        corrcoef_volume = self.joint_corrcoef_val(volume_hat, volume)
        corrcoef_dist_src = self.joint_corrcoef_val(dist_src_hat, dist_src)

        self.val_loss_sti(loss_sti)
        self.val_loss_alcons(loss_alcons)
        self.val_loss_tr(loss_tr)
        self.val_loss_edt(loss_edt)
        self.val_loss_c80(loss_c80)
        self.val_loss_c50(loss_c50)
        self.val_loss_d50(loss_d50)
        self.val_loss_ts(loss_ts)
        self.val_loss_volume(loss_volume)
        self.val_loss_dist_src(loss_dist_src)

        self.val_corrcoef_sti(corrcoef_sti)
        self.val_corrcoef_alcons(corrcoef_alcons)
        self.val_corrcoef_tr(corrcoef_tr)
        self.val_corrcoef_edt(corrcoef_edt)
        self.val_corrcoef_c80(corrcoef_c80)
        self.val_corrcoef_c50(corrcoef_c50)
        self.val_corrcoef_d50(corrcoef_d50)
        self.val_corrcoef_ts(corrcoef_ts)
        self.val_corrcoef_volume(corrcoef_volume)
        self.val_corrcoef_dist_src(corrcoef_dist_src)

        # ------------------- total loss -------------------

        total_loss = (
            self.sti_weight * loss_sti
            + self.alcons_weight * loss_alcons
            + self.tr_weight * loss_tr
            + self.edt_weight * loss_edt
            + self.c80_weight * loss_c80
            + self.c50_weight * loss_c50
            + self.d50_weight * loss_d50
            + self.ts_weight * loss_ts
            + self.volume_weight * loss_volume
            + self.dist_src_weight * loss_dist_src
        )

        # ------------------- log metrics -------------------
        self.log(
            "val/loss/sti",
            self.val_loss_sti,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "val/corrcoef/sti",
            self.val_corrcoef_sti,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "val/loss/alcons",
            self.val_loss_alcons,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "val/corrcoef/alcons",
            self.val_corrcoef_alcons,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "val/loss/tr",
            self.val_loss_tr,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "val/corrcoef/tr",
            self.val_corrcoef_tr,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "val/loss/edt",
            self.val_loss_edt,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "val/corrcoef/edt",
            self.val_corrcoef_edt,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "val/loss/c80",
            self.val_loss_c80,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "val/corrcoef/c80",
            self.val_corrcoef_c80,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "val/loss/c50",
            self.val_loss_c50,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "val/corrcoef/c50",
            self.val_corrcoef_c50,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "val/loss/d50",
            self.val_loss_d50,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "val/corrcoef/d50",
            self.val_corrcoef_d50,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "val/loss/ts",
            self.val_loss_ts,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "val/corrcoef/ts",
            self.val_corrcoef_ts,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "val/loss/volume",
            self.val_loss_volume,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "val/corrcoef/volume",
            self.val_corrcoef_volume,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "val/loss/dist_src",
            self.val_loss_dist_src,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "val/corrcoef/dist_src",
            self.val_corrcoef_dist_src,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "val/loss/total",
            total_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        # ------------------- best so far validation loss -------------------
        metric_sti = self.val_loss_sti.compute()  # get current val metric
        metric_alcons = self.val_loss_alcons.compute()  # get current val metric
        metric_tr = self.val_loss_tr.compute()  # get current val metric
        metric_edt = self.val_loss_edt.compute()  # get current val metric
        metric_c80 = self.val_loss_c80.compute()  # get current val metric
        metric_c50 = self.val_loss_c50.compute()  # get current val metric
        metric_d50 = self.val_loss_d50.compute()  # get current val metric
        metric_ts = self.val_loss_ts.compute()  # get current val metric
        metric_volume = self.val_loss_volume.compute()  # get current val metric
        metric_dist_src = self.val_loss_dist_src.compute()  # get current val metric

        # ------------- best so far validation correlation coefficient ------------
        metric_sti_corrcoef = self.val_corrcoef_sti.compute()
        metric_alcons_corrcoef = self.val_corrcoef_alcons.compute()
        metric_tr_corrcoef = self.val_corrcoef_tr.compute()
        metric_edt_corrcoef = self.val_corrcoef_edt.compute()
        metric_c80_corrcoef = self.val_corrcoef_c80.compute()
        metric_c50_corrcoef = self.val_corrcoef_c50.compute()
        metric_d50_corrcoef = self.val_corrcoef_d50.compute()
        metric_ts_corrcoef = self.val_corrcoef_ts.compute()
        metric_volume_corrcoef = self.val_corrcoef_volume.compute()
        metric_dist_src_corrcoef = self.val_corrcoef_dist_src.compute()
        # metric_azimuth_corrcoef = self.val_corrcoef_azimuth.compute()
        # metric_elevation_corrcoef = self.val_corrcoef_elevation.compute()

        # update best so far val metric of Th, Tt, volume, distSrc, azimuthSrc, elevationSrc
        # update best so far val metric of loss
        self.val_loss_best_sti(metric_sti)  # update best so far val metric
        self.val_loss_best_alcons(metric_alcons)  # update best so far val metric
        self.val_loss_best_tr(metric_tr)  # update best so far val metric
        self.val_loss_best_edt(metric_edt)  # update best so far val metric
        self.val_loss_best_c80(metric_c80)  # update best so far val metric
        self.val_loss_best_c50(metric_c50)  # update best so far val metric
        self.val_loss_best_d50(metric_d50)  # update best so far val metric
        self.val_loss_best_ts(metric_ts)  # update best so far val metric
        self.val_loss_best_volume(metric_volume)  # update best so far val metric
        self.val_loss_best_dist_src(metric_dist_src)  # update best so far val metric

        # update best so far val metric of correlation coefficient
        self.val_corrcoef_best_sti(metric_sti_corrcoef)
        self.val_corrcoef_best_alcons(metric_alcons_corrcoef)
        self.val_corrcoef_best_tr(metric_tr_corrcoef)
        self.val_corrcoef_best_edt(metric_edt_corrcoef)
        self.val_corrcoef_best_c80(metric_c80_corrcoef)
        self.val_corrcoef_best_c50(metric_c50_corrcoef)
        self.val_corrcoef_best_d50(metric_d50_corrcoef)
        self.val_corrcoef_best_ts(metric_ts_corrcoef)
        self.val_corrcoef_best_volume(metric_volume_corrcoef)
        self.val_corrcoef_best_dist_src(metric_dist_src_corrcoef)

        # ------------------- total best so far validation loss -------------------

        total_loss_best = (
            self.sti_weight * self.val_loss_best_sti
            + self.alcons_weight * self.val_loss_best_alcons
            + self.tr_weight * self.val_loss_best_tr
            + self.edt_weight * self.val_loss_best_edt
            + self.c80_weight * self.val_loss_best_c80
            + self.c50_weight * self.val_loss_best_c50
            + self.d50_weight * self.val_loss_best_d50
            + self.ts_weight * self.val_loss_best_ts
            + self.volume_weight * self.val_loss_best_volume
            + self.dist_src_weight * self.val_loss_best_dist_src
        )

        # log `val_loss_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch

        # ------------------- log best so far validation loss -------------------

        self.log(
            "val/loss_best/sti",
            self.val_loss_best_sti.compute(),
            sync_dist=True,
            prog_bar=True,
        )

        self.log(
            "val/loss_best/alcons",
            self.val_loss_best_alcons.compute(),
            sync_dist=True,
            prog_bar=True,
        )

        self.log(
            "val/loss_best/tr",
            self.val_loss_best_tr.compute(),
            sync_dist=True,
            prog_bar=True,
        )

        self.log(
            "val/loss_best/edt",
            self.val_loss_best_edt.compute(),
            sync_dist=True,
            prog_bar=True,
        )

        self.log(
            "val/loss_best/c80",
            self.val_loss_best_c80.compute(),
            sync_dist=True,
            prog_bar=True,
        )

        self.log(
            "val/loss_best/c50",
            self.val_loss_best_c50.compute(),
            sync_dist=True,
            prog_bar=True,
        )

        self.log(
            "val/loss_best/d50",
            self.val_loss_best_d50.compute(),
            sync_dist=True,
            prog_bar=True,
        )

        self.log(
            "val/loss_best/ts",
            self.val_loss_best_ts.compute(),
            sync_dist=True,
            prog_bar=True,
        )

        self.log(
            "val/loss_best/volume",
            self.val_loss_best_volume.compute(),
            sync_dist=True,
            prog_bar=True,
        )

        self.log(
            "val/loss_best/dist_src",
            self.val_loss_best_dist_src.compute(),
            sync_dist=True,
            prog_bar=True,
        )

        self.log(
            "val/loss_best/total",
            total_loss_best.compute(),
            sync_dist=True,
            prog_bar=True,
        )

        # ------------------- log best so far validation correlation coefficient -------------------
        # log `val_loss_best` as a value through `.compute()` method, instead of as a metric object
        self.log(
            "val/corrcoef_best/sti",
            self.val_corrcoef_best_sti.compute(),
            sync_dist=True,
            prog_bar=True,
        )

        self.log(
            "val/corrcoef_best/alcons",
            self.val_corrcoef_best_alcons.compute(),
            sync_dist=True,
            prog_bar=True,
        )

        self.log(
            "val/corrcoef_best/tr",
            self.val_corrcoef_best_tr.compute(),
            sync_dist=True,
            prog_bar=True,
        )

        self.log(
            "val/corrcoef_best/edt",
            self.val_corrcoef_best_edt.compute(),
            sync_dist=True,
            prog_bar=True,
        )

        self.log(
            "val/corrcoef_best/c80",
            self.val_corrcoef_best_c80.compute(),
            sync_dist=True,
            prog_bar=True,
        )

        self.log(
            "val/corrcoef_best/c50",
            self.val_corrcoef_best_c50.compute(),
            sync_dist=True,
            prog_bar=True,
        )

        self.log(
            "val/corrcoef_best/d50",
            self.val_corrcoef_best_d50.compute(),
            sync_dist=True,
            prog_bar=True,
        )

        self.log(
            "val/corrcoef_best/ts",
            self.val_corrcoef_best_ts.compute(),
            sync_dist=True,
            prog_bar=True,
        )

        self.log(
            "val/corrcoef_best/volume",
            self.val_corrcoef_best_volume.compute(),
            sync_dist=True,
            prog_bar=True,
        )

        self.log(
            "val/corrcoef_best/dist_src",
            self.val_corrcoef_best_dist_src.compute(),
            sync_dist=True,
            prog_bar=True,
        )

    def test_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        sti = batch["groundtruth"]["sti"]
        alcons = batch["groundtruth"]["alcons"]
        tr = batch["groundtruth"]["tr"]
        edt = batch["groundtruth"]["edt"]
        c80 = batch["groundtruth"]["c80"]
        c50 = batch["groundtruth"]["c50"]
        d50 = batch["groundtruth"]["d50"]
        ts = batch["groundtruth"]["ts"]
        volume = batch["groundtruth"]["volume"]
        dist_src = batch["groundtruth"]["dist_src"]

        net_output = self.forward(**batch["net_input"])
        sti_hat = net_output["sti_hat"]
        alcons_hat = net_output["alcons_hat"]
        tr_hat = net_output["tr_hat"]
        edt_hat = net_output["edt_hat"]
        c80_hat = net_output["c80_hat"]
        c50_hat = net_output["c50_hat"]
        d50_hat = net_output["d50_hat"]
        ts_hat = net_output["ts_hat"]
        volume_hat = net_output["volume_hat"]
        dist_src_hat = net_output["dist_src_hat"]
        padding_mask = net_output["padding_mask"]

        if padding_mask is not None and padding_mask.any():
            reverse_padding_mask = padding_mask.logical_not()
            # ---------- padding mask handling ----------
            # Collapse the time dimension
            sti_hat = (sti_hat * reverse_padding_mask).sum(
                dim=1
            ) / reverse_padding_mask.sum(dim=1)
            alcons_hat = (alcons_hat * reverse_padding_mask).sum(
                dim=1
            ) / reverse_padding_mask.sum(dim=1)
            tr_hat = (tr_hat * reverse_padding_mask).sum(
                dim=1
            ) / reverse_padding_mask.sum(dim=1)
            edt_hat = (edt_hat * reverse_padding_mask).sum(
                dim=1
            ) / reverse_padding_mask.sum(dim=1)
            c80_hat = (c80_hat * reverse_padding_mask).sum(
                dim=1
            ) / reverse_padding_mask.sum(dim=1)
            c50_hat = (c50_hat * reverse_padding_mask).sum(
                dim=1
            ) / reverse_padding_mask.sum(dim=1)
            d50_hat = (d50_hat * reverse_padding_mask).sum(
                dim=1
            ) / reverse_padding_mask.sum(dim=1)
            ts_hat = (ts_hat * reverse_padding_mask).sum(
                dim=1
            ) / reverse_padding_mask.sum(dim=1)
            volume_hat = (volume_hat * reverse_padding_mask).sum(
                dim=1
            ) / reverse_padding_mask.sum(dim=1)
            dist_src_hat = (dist_src_hat * reverse_padding_mask).sum(
                dim=1
            ) / reverse_padding_mask.sum(dim=1)

        else:
            # Collapse the time dimension
            sti_hat = sti_hat.mean(dim=1)
            alcons_hat = alcons_hat.mean(dim=1)
            tr_hat = tr_hat.mean(dim=1)
            edt_hat = edt_hat.mean(dim=1)
            c80_hat = c80_hat.mean(dim=1)
            c50_hat = c50_hat.mean(dim=1)
            d50_hat = d50_hat.mean(dim=1)
            ts_hat = ts_hat.mean(dim=1)
            volume_hat = volume_hat.mean(dim=1)
            dist_src_hat = dist_src_hat.mean(dim=1)

        # MAE loss
        loss_sti = self.joint_loss_test(sti_hat, sti)
        loss_alcons = self.joint_loss_test(alcons_hat, alcons)
        loss_tr = self.joint_loss_test(tr_hat, tr)
        loss_edt = self.joint_loss_test(edt_hat, edt)
        loss_c80 = self.joint_loss_test(c80_hat, c80)
        loss_c50 = self.joint_loss_test(c50_hat, c50)
        loss_d50 = self.joint_loss_test(d50_hat, d50)
        loss_ts = self.joint_loss_test(ts_hat, ts)
        loss_volume = self.joint_loss_test(volume_hat, volume)
        loss_dist_src = self.joint_loss_test(dist_src_hat, dist_src)

        # correlation coefficient loss
        loss_sti_corrcoef = self.joint_corrcoef_test(sti_hat, sti)
        loss_alcons_corrcoef = self.joint_corrcoef_test(alcons_hat, alcons)
        loss_tr_corrcoef = self.joint_corrcoef_test(tr_hat, tr)
        loss_edt_corrcoef = self.joint_corrcoef_test(edt_hat, edt)
        loss_c80_corrcoef = self.joint_corrcoef_test(c80_hat, c80)
        loss_c50_corrcoef = self.joint_corrcoef_test(c50_hat, c50)
        loss_d50_corrcoef = self.joint_corrcoef_test(d50_hat, d50)
        loss_ts_corrcoef = self.joint_corrcoef_test(ts_hat, ts)
        loss_volume_corrcoef = self.joint_corrcoef_test(volume_hat, volume)
        loss_dist_src_corrcoef = self.joint_corrcoef_test(dist_src_hat, dist_src)

        # update and log metrics
        self.test_loss_sti(loss_sti)
        self.test_loss_alcons(loss_alcons)
        self.test_loss_tr(loss_tr)
        self.test_loss_edt(loss_edt)
        self.test_loss_c80(loss_c80)
        self.test_loss_c50(loss_c50)
        self.test_loss_d50(loss_d50)
        self.test_loss_ts(loss_ts)
        self.test_loss_volume(loss_volume)
        self.test_loss_dist_src(loss_dist_src)

        self.test_corrcoef_sti(loss_sti_corrcoef)
        self.test_corrcoef_alcons(loss_alcons_corrcoef)
        self.test_corrcoef_tr(loss_tr_corrcoef)
        self.test_corrcoef_edt(loss_edt_corrcoef)
        self.test_corrcoef_c80(loss_c80_corrcoef)
        self.test_corrcoef_c50(loss_c50_corrcoef)
        self.test_corrcoef_d50(loss_d50_corrcoef)
        self.test_corrcoef_ts(loss_ts_corrcoef)
        self.test_corrcoef_volume(loss_volume_corrcoef)
        self.test_corrcoef_dist_src(loss_dist_src_corrcoef)

        self.log(
            "test/loss/sti",
            self.test_loss_sti,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "test/corrcoef/sti",
            self.test_corrcoef_sti,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "test/loss/alcons",
            self.test_loss_alcons,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "test/corrcoef/alcons",
            self.test_corrcoef_alcons,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "test/loss/tr",
            self.test_loss_tr,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "test/corrcoef/tr",
            self.test_corrcoef_tr,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "test/loss/edt",
            self.test_loss_edt,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "test/corrcoef/edt",
            self.test_corrcoef_edt,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "test/loss/c80",
            self.test_loss_c80,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "test/corrcoef/c80",
            self.test_corrcoef_c80,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "test/loss/c50",
            self.test_loss_c50,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "test/corrcoef/c50",
            self.test_corrcoef_c50,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "test/loss/d50",
            self.test_loss_d50,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "test/corrcoef/d50",
            self.test_corrcoef_d50,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "test/loss/ts",
            self.test_loss_ts,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "test/corrcoef/ts",
            self.test_corrcoef_ts,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "test/loss/volume",
            self.test_loss_volume,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "test/corrcoef/volume",
            self.test_corrcoef_volume,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "test/loss/dist_src",
            self.test_loss_dist_src,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "test/corrcoef/dist_src",
            self.test_corrcoef_dist_src,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        # Th_pred = self.model_step(batch)
        pass

    def configure_optimizers(self) -> dict[str, Any]:
        """Configures optimizers and learning-rate schedulers to be used for training.

        Normally you'd need one, but in the case of GANs or similar you might need multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(self.regressor.parameters())

        if self.hparams.scheduler is not None:
            T_max = self.hparams.optim_cfg.T_max
            eta_min = 1e-8

            scheduler = self.hparams.scheduler(
                optimizer=optimizer,
                T_max=T_max,
                eta_min=eta_min,
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss/total",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    import hydra
    import omegaconf
    import rootutils

    root = rootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "network.yaml")
    _ = hydra.utils.instantiate(cfg.model.jointRegressorModule)

    _ = JointRegressorModuleEncoder(cfg.model.jointRegressorModule)
