from typing import Any, Dict, Optional

import torch
from lightning import LightningModule
from torchmetrics import (
    MinMetric,
    MaxMetric,
    MeanMetric,
    PearsonCorrCoef,
)
import einops

from src.utils.unitary_linear_norm import unitary_norm_inv
from torch.nn import SmoothL1Loss, L1Loss

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

        self.rirRegressor = net

        # loss function
        self.joint_loss_train = SmoothL1Loss()

        self.joint_loss_val = L1Loss()
        self.joint_corrcoef_val = PearsonCorrCoef()

        self.joint_loss_test = L1Loss()
        self.joint_corrcoef_test = PearsonCorrCoef()

        self.Th_weight = self.hparams.optim_cfg.Th_weight
        self.Tt_weight = self.hparams.optim_cfg.Tt_weight
        self.volume_weight = self.hparams.optim_cfg.volume_weight
        self.dist_src_weight = self.hparams.optim_cfg.dist_src_weight
        self.azimuth_weight = self.hparams.optim_cfg.azimuth_weight
        self.elevation_weight = self.hparams.optim_cfg.elevation_weight

        # for averaging loss across batches
        self.train_loss_Th = MeanMetric()
        self.val_loss_Th = MeanMetric()
        self.test_loss_Th = MeanMetric()

        self.train_loss_Tt = MeanMetric()
        self.val_loss_Tt = MeanMetric()
        self.test_loss_Tt = MeanMetric()

        self.train_loss_volume = MeanMetric()
        self.val_loss_volume = MeanMetric()
        self.test_loss_volume = MeanMetric()

        self.train_loss_dist_src = MeanMetric()
        self.val_loss_dist_src = MeanMetric()
        self.test_loss_dist_src = MeanMetric()

        # self.train_loss_azimuth = MeanMetric()
        # self.val_loss_azimuth = MeanMetric()
        # self.test_loss_azimuth = MeanMetric()

        # self.train_loss_elevation = MeanMetric()
        # self.val_loss_elevation = MeanMetric()
        # self.test_loss_elevation = MeanMetric()

        # for tracking validation correlation coefficient of Th, Tt, volume, distSrc, azimuthSrc, elevationSrc
        self.val_corrcoef_Th = MeanMetric()
        self.val_corrcoef_Tt = MeanMetric()
        self.val_corrcoef_volume = MeanMetric()
        self.val_corrcoef_dist_src = MeanMetric()
        # self.val_corrcoef_azimuth = MeanMetric()
        # self.val_corrcoef_elevation = MeanMetric()

        # for tracking evaluation correlation coefficient of Th, Tt, volume, distSrc, azimuthSrc, elevationSrc
        self.test_corrcoef_Th = MeanMetric()
        self.test_corrcoef_Tt = MeanMetric()
        self.test_corrcoef_volume = MeanMetric()
        self.test_corrcoef_dist_src = MeanMetric()
        # self.test_corrcoef_azimuth = MeanMetric()
        # self.test_corrcoef_elevation = MeanMetric()

        # for tracking best so far validation best metrics
        self.val_loss_best_Th = MinMetric()
        self.val_loss_best_Tt = MinMetric()
        self.val_loss_best_volume = MinMetric()
        self.val_loss_best_dist_src = MinMetric()
        # self.val_loss_best_azimuth = MinMetric()
        # self.val_loss_best_elevation = MinMetric()

        # for tracking best of correlation coefficient
        self.val_corrcoef_best_Th = MaxMetric()
        self.val_corrcoef_best_Tt = MaxMetric()
        self.val_corrcoef_best_volume = MaxMetric()
        self.val_corrcoef_best_dist_src = MaxMetric()
        # self.val_corrcoef_best_azimuth = MaxMetric()
        # self.val_corrcoef_best_elevation = MaxMetric()

    def forward(
        self,
        source: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Perform a forward pass through freezed denoiser and rirRegressor.

        :param x: A tensor of waveform
        :return: A tensor of estimated Th, Tt, volume, distSrc, azimuthSrc, elevationSrc.
        """
        net_output = self.rirRegressor(source, padding_mask)

        return net_output

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss_Th.reset()
        self.val_loss_Tt.reset()
        self.val_loss_volume.reset()
        self.val_loss_dist_src.reset()
        # self.val_loss_azimuth.reset()
        # self.val_loss_elevation.reset()

        self.val_corrcoef_Th.reset()
        self.val_corrcoef_Tt.reset()
        self.val_corrcoef_volume.reset()
        self.val_corrcoef_dist_src.reset()
        # self.val_corrcoef_azimuth.reset()
        # self.val_corrcoef_elevation.reset()

        self.val_loss_best_Th.reset()
        self.val_loss_best_Tt.reset()
        self.val_loss_best_volume.reset()
        self.val_loss_best_dist_src.reset()
        # self.val_loss_best_azimuth.reset()
        # self.val_loss_best_elevation.reset()

        self.val_corrcoef_best_Th.reset()
        self.val_corrcoef_best_Tt.reset()
        self.val_corrcoef_best_volume.reset()
        self.val_corrcoef_best_dist_src.reset()
        # self.val_corrcoef_best_azimuth.reset()
        # self.val_corrcoef_best_elevation.reset()

    def model_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a dict) containing the input tensor
            of raw, target Th, target Tt, target volume, target distSrc,
            target azimuthSrc, target elevationSrc.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        Th = batch["groundtruth"]["Th"]
        Tt = batch["groundtruth"]["Tt"]
        volume = batch["groundtruth"]["volume"]
        dist_src = batch["groundtruth"]["dist_src"]
        # azimuth = batch["groundtruth"]["azimuth"]
        # elevation = batch["groundtruth"]["elevation"]

        # network forward pass
        net_output = self.forward(**batch["net_input"])
        Th_hat = net_output["Th_hat"]
        Tt_hat = net_output["Tt_hat"]
        volume_hat = net_output["volume_hat"]
        dist_src_hat = net_output["dist_src_hat"]
        # azimuth_hat = net_output["azimuth_hat"]
        # elevation_hat = net_output["elevation_hat"]
        padding_mask = net_output["padding_mask"]

        if padding_mask is not None and padding_mask.any():
            reverse_padding_mask = padding_mask.logical_not()
            # ---------- padding mask handling ----------
            # Repeat the target labels to match the output shape
            Th = einops.repeat(Th, "b -> b t", t=Th_hat.shape[1])
            Tt = einops.repeat(Tt, "b -> b t", t=Tt_hat.shape[1])
            volume = einops.repeat(volume, "b -> b t", t=volume_hat.shape[1])
            dist_src = einops.repeat(dist_src, "b -> b t", t=dist_src_hat.shape[1])
            # azimuth = einops.repeat(azimuth, "b -> b t", t=azimuth_hat.shape[1])
            # elevation = einops.repeat(elevation, "b -> b t", t=elevation_hat.shape[1])

            Th_hat = Th_hat.masked_select(reverse_padding_mask)
            Tt_hat = Tt_hat.masked_select(reverse_padding_mask)
            volume_hat = volume_hat.masked_select(reverse_padding_mask)
            dist_src_hat = dist_src_hat.masked_select(reverse_padding_mask)
            # azimuth_hat = azimuth_hat.masked_select(reverse_padding_mask)
            # elevation_hat = elevation_hat.masked_select(reverse_padding_mask)

            Th = Th.masked_select(reverse_padding_mask)
            Tt = Tt.masked_select(reverse_padding_mask)
            volume = volume.masked_select(reverse_padding_mask)
            dist_src = dist_src.masked_select(reverse_padding_mask)
            # azimuth = azimuth.masked_select(reverse_padding_mask)
            # elevation = elevation.masked_select(reverse_padding_mask)

        else:
            # Repeat the target labels to match the output shape
            Th = einops.repeat(Th, "b -> b t", t=Th_hat.shape[1])
            Tt = einops.repeat(Tt, "b -> b t", t=Tt_hat.shape[1])
            volume = einops.repeat(volume, "b -> b t", t=volume_hat.shape[1])
            dist_src = einops.repeat(dist_src, "b -> b t", t=dist_src_hat.shape[1])
            # azimuth = einops.repeat(azimuth, "b -> b t", t=azimuth_hat.shape[1])
            # elevation = einops.repeat(elevation, "b -> b t", t=elevation_hat.shape[1])

        loss_Th = self.joint_loss_train(Th_hat, Th)
        loss_Tt = self.joint_loss_train(Tt_hat, Tt)
        loss_volume = self.joint_loss_train(volume_hat, volume)
        loss_dist_src = self.joint_loss_train(dist_src_hat, dist_src)
        # loss_azimuth = self.joint_loss_train(azimuth_hat, azimuth)
        # loss_elevation = self.joint_loss_train(elevation_hat, elevation)

        return {
            "loss_Th": loss_Th,
            "loss_Tt": loss_Tt,
            "loss_volume": loss_volume,
            "loss_dist_src": loss_dist_src,
            # "loss_azimuth": loss_azimuth,
            # "loss_elevation": loss_elevation,
        }

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
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

        loss_Th = output["loss_Th"]
        loss_Tt = output["loss_Tt"]
        loss_volume = output["loss_volume"]
        loss_dist_src = output["loss_dist_src"]
        # loss_azimuth = output["loss_azimuth"]
        # loss_elevation = output["loss_elevation"]

        # ------------------- other parameters -------------------

        # update metrics of Th, Tt, volume, distSrc, azimuthSrc, elevationSrc loss
        self.train_loss_Th(loss_Th)
        self.train_loss_Tt(loss_Tt)
        self.train_loss_volume(loss_volume)
        self.train_loss_dist_src(loss_dist_src)
        # self.train_loss_azimuth(loss_azimuth)
        # self.train_loss_elevation(loss_elevation)

        # ------------------- total loss -------------------
        total_loss = (
            self.Th_weight * loss_Th
            + self.Tt_weight * loss_Tt
            + self.volume_weight * loss_volume
            + self.dist_src_weight * loss_dist_src
            # + self.azimuth_weight * loss_azimuth
            # + self.elevation_weight * loss_elevation
        )

        # ------------------- log metrics -------------------
        self.log(
            "train/loss/Th",
            self.train_loss_Th,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "train/loss/Tt",
            self.train_loss_Tt,
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

        # self.log(
        #     "train/loss/azimuth_src",
        #     self.train_loss_azimuth,
        #     on_step=True,
        #     on_epoch=True,
        #     prog_bar=True,
        #     sync_dist=True,
        # )

        # self.log(
        #     "train/loss/elevation_src",
        #     self.train_loss_elevation,
        #     on_step=True,
        #     on_epoch=True,
        #     prog_bar=True,
        #     sync_dist=True,
        # )

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
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a dict) containing the input tensor of raw,
            target Th, target Tt, target volume, target distSrc, target azimuthSrc,
            target elevationSrc.
        :param batch_idx: The index of the current batch.
        """
        Th = batch["groundtruth"]["Th"]
        Tt = batch["groundtruth"]["Tt"]
        volume = batch["groundtruth"]["volume"]
        dist_src = batch["groundtruth"]["dist_src"]
        # azimuth = batch["groundtruth"]["azimuth"]
        # elevation = batch["groundtruth"]["elevation"]

        # network forward pass
        net_output = self.forward(**batch["net_input"])
        Th_hat = net_output["Th_hat"]
        Tt_hat = net_output["Tt_hat"]
        volume_hat = net_output["volume_hat"]
        dist_src_hat = net_output["dist_src_hat"]
        # azimuth_hat = net_output["azimuth_hat"]
        # elevation_hat = net_output["elevation_hat"]
        padding_mask = net_output["padding_mask"]

        if padding_mask is not None and padding_mask.any():
            reverse_padding_mask = padding_mask.logical_not()
            # ---------- padding mask handling ----------
            # Collapse the time dimension
            Th_hat = (Th_hat * reverse_padding_mask).sum(
                dim=1
            ) / reverse_padding_mask.sum(dim=1)
            Tt_hat = (Tt_hat * reverse_padding_mask).sum(
                dim=1
            ) / reverse_padding_mask.sum(dim=1)
            volume_hat = (volume_hat * reverse_padding_mask).sum(
                dim=1
            ) / reverse_padding_mask.sum(dim=1)
            dist_src_hat = (dist_src_hat * reverse_padding_mask).sum(
                dim=1
            ) / reverse_padding_mask.sum(dim=1)
            # azimuth_hat = (azimuth_hat * reverse_padding_mask).sum(
            #     dim=1
            # ) / reverse_padding_mask.sum(dim=1)
            # elevation_hat = (elevation_hat * reverse_padding_mask).sum(
            #     dim=1
            # ) / reverse_padding_mask.sum(dim=1)

        else:
            # Collapse the time dimension
            Th_hat = Th_hat.mean(dim=1)
            Tt_hat = Tt_hat.mean(dim=1)
            volume_hat = volume_hat.mean(dim=1)
            dist_src_hat = dist_src_hat.mean(dim=1)
            # azimuth_hat = azimuth_hat.mean(dim=1)
            # elevation_hat = elevation_hat.mean(dim=1)

        # update metrics of Th, Tt, volume, distSrc loss and correlation coefficient
        loss_Th = self.joint_loss_val(Th_hat, Th)
        loss_Tt = self.joint_loss_val(Tt_hat, Tt)
        loss_volume = self.joint_loss_val(volume_hat, volume)
        loss_dist_src = self.joint_loss_val(dist_src_hat, dist_src)
        # loss_azimuth_src = self.joint_loss_val(azimuth_hat, azimuth)
        # loss_elevation_src = self.joint_loss_val(elevation_hat, elevation)

        corrcoef_Th = self.joint_corrcoef_val(Th_hat, Th)
        corrcoef_Tt = self.joint_corrcoef_val(Tt_hat, Tt)
        corrcoef_volume = self.joint_corrcoef_val(volume_hat, volume)
        corrcoef_dist_src = self.joint_corrcoef_val(dist_src_hat, dist_src)
        # corrcoef_azimuth_src = self.joint_corrcoef_val(azimuth_hat, azimuth)
        # corrcoef_elevation_src = self.joint_corrcoef_val(elevation_hat, elevation)

        self.val_loss_Th(loss_Th)
        self.val_loss_Tt(loss_Tt)
        self.val_loss_volume(loss_volume)
        self.val_loss_dist_src(loss_dist_src)
        # self.val_loss_azimuth(loss_azimuth_src)
        # self.val_loss_elevation(loss_elevation_src)

        self.val_corrcoef_Th(corrcoef_Th)
        self.val_corrcoef_Tt(corrcoef_Tt)
        self.val_corrcoef_volume(corrcoef_volume)
        self.val_corrcoef_dist_src(corrcoef_dist_src)
        # self.val_corrcoef_azimuth(corrcoef_azimuth_src)
        # self.val_corrcoef_elevation(corrcoef_elevation_src)

        # ------------------- total loss -------------------

        total_loss = (
            self.Th_weight * loss_Th
            + self.Tt_weight * loss_Tt
            + self.volume_weight * loss_volume
            + self.dist_src_weight * loss_dist_src
            # + self.azimuth_weight * loss_azimuth_src
            # + self.elevation_weight * loss_elevation_src
        )

        # ------------------- log metrics -------------------
        self.log(
            "val/loss/Th",
            self.val_loss_Th,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "val/corrcoef/Th",
            self.val_corrcoef_Th,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "val/loss/Tt",
            self.val_loss_Tt,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "val/corrcoef/Tt",
            self.val_corrcoef_Tt,
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

        # self.log(
        #     "val/loss/azimuth_src",
        #     self.val_loss_azimuth,
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=True,
        #     sync_dist=True,
        # )

        # self.log(
        #     "val/corrcoef/azimuth",
        #     self.val_corrcoef_azimuth,
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=True,
        #     sync_dist=True,
        # )

        # self.log(
        #     "val/loss/elevation_src",
        #     self.val_loss_elevation,
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=True,
        #     sync_dist=True,
        # )

        # self.log(
        #     "val/corrcoef/elevation",
        #     self.val_corrcoef_elevation,
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=True,
        #     sync_dist=True,
        # )

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
        metric_Th = self.val_loss_Th.compute()  # get current val metric
        metric_Tt = self.val_loss_Tt.compute()  # get current val metric
        metric_volume = self.val_loss_volume.compute()  # get current val metric
        metric_dist_src = self.val_loss_dist_src.compute()  # get current val metric
        # metric_azimuth = self.val_loss_azimuth.compute()  # get current val metric
        # metric_elevation = self.val_loss_elevation.compute()  # get current val metric

        # ------------- best so far validation correlation coefficient ------------
        metric_Th_corrcoef = self.val_corrcoef_Th.compute()
        metric_Tt_corrcoef = self.val_corrcoef_Tt.compute()
        metric_volume_corrcoef = self.val_corrcoef_volume.compute()
        metric_dist_src_corrcoef = self.val_corrcoef_dist_src.compute()
        # metric_azimuth_corrcoef = self.val_corrcoef_azimuth.compute()
        # metric_elevation_corrcoef = self.val_corrcoef_elevation.compute()

        # update best so far val metric of Th, Tt, volume, distSrc, azimuthSrc, elevationSrc
        # update best so far val metric of loss
        self.val_loss_best_Th(metric_Th)  # update best so far val metric
        self.val_loss_best_Tt(metric_Tt)  # update best so far val metric
        self.val_loss_best_volume(metric_volume)  # update best so far val metric
        self.val_loss_best_dist_src(metric_dist_src)  # update best so far val metric
        # self.val_loss_best_azimuth(metric_azimuth)  # update best so far val metric
        # self.val_loss_best_elevation(metric_elevation)  # update best so far val metric

        # update best so far val metric of correlation coefficient
        self.val_corrcoef_best_Th(metric_Th_corrcoef)
        self.val_corrcoef_best_Tt(metric_Tt_corrcoef)
        self.val_corrcoef_best_volume(metric_volume_corrcoef)
        self.val_corrcoef_best_dist_src(metric_dist_src_corrcoef)
        # self.val_corrcoef_best_azimuth(metric_azimuth_corrcoef)
        # self.val_corrcoef_best_elevation(metric_elevation_corrcoef)

        # ------------------- total best so far validation loss -------------------

        total_loss_best = (
            self.Th_weight * self.val_loss_best_Th
            + self.Tt_weight * self.val_loss_best_Tt
            + self.volume_weight * self.val_loss_best_volume
            + self.dist_src_weight * self.val_loss_best_dist_src
            # + self.azimuth_weight * self.val_loss_best_azimuth
            # + self.elevation_weight * self.val_loss_best_elevation
        )

        # log `val_loss_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch

        # ------------------- log best so far validation loss -------------------

        self.log(
            "val/loss_best/Th",
            self.val_loss_best_Th.compute(),
            sync_dist=True,
            prog_bar=True,
        )

        self.log(
            "val/loss_best/Tt",
            self.val_loss_best_Tt.compute(),
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

        # self.log(
        #     "val/loss_best/azimuth_src",
        #     self.val_loss_best_azimuth.compute(),
        #     sync_dist=True,
        #     prog_bar=True,
        # )

        # self.log(
        #     "val/loss_best/elevation_src",
        #     self.val_loss_best_elevation.compute(),
        #     sync_dist=True,
        #     prog_bar=True,
        # )

        self.log(
            "val/loss_best/total",
            total_loss_best.compute(),
            sync_dist=True,
            prog_bar=True,
        )

        # ------------------- log best so far validation correlation coefficient -------------------
        # log `val_loss_best` as a value through `.compute()` method, instead of as a metric object
        self.log(
            "val/corrcoef_best/Th",
            self.val_corrcoef_best_Th.compute(),
            sync_dist=True,
            prog_bar=True,
        )

        self.log(
            "val/corrcoef_best/Tt",
            self.val_corrcoef_best_Tt.compute(),
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

        # self.log(
        #     "val/corrcoef_best/azimuth_src",
        #     self.val_corrcoef_best_azimuth.compute(),
        #     sync_dist=True,
        #     prog_bar=True,
        # )

        # self.log(
        #     "val/corrcoef_best/elevation_src",
        #     self.val_corrcoef_best_elevation.compute(),
        #     sync_dist=True,
        #     prog_bar=True,
        # )

    def test_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        Th = batch["groundtruth"]["Th"]
        Tt = batch["groundtruth"]["Tt"]
        volume = batch["groundtruth"]["volume"]
        dist_src = batch["groundtruth"]["dist_src"]
        # azimuth = batch["groundtruth"]["azimuth"]
        # elevation = batch["groundtruth"]["elevation"]

        net_output = self.forward(**batch["net_input"])
        Th_hat = net_output["Th_hat"]
        Tt_hat = net_output["Tt_hat"]
        volume_hat = net_output["volume_hat"]
        dist_src_hat = net_output["dist_src_hat"]
        # azimuth_hat = net_output["azimuth_hat"]
        # elevation_hat = net_output["elevation_hat"]
        padding_mask = net_output["padding_mask"]

        if padding_mask is not None and padding_mask.any():
            reverse_padding_mask = padding_mask.logical_not()
            # ---------- padding mask handling ----------
            # Collapse the time dimension
            Th_hat = (Th_hat * reverse_padding_mask).sum(
                dim=1
            ) / reverse_padding_mask.sum(dim=1)
            Tt_hat = (Tt_hat * reverse_padding_mask).sum(
                dim=1
            ) / reverse_padding_mask.sum(dim=1)
            volume_hat = (volume_hat * reverse_padding_mask).sum(
                dim=1
            ) / reverse_padding_mask.sum(dim=1)
            dist_src_hat = (dist_src_hat * reverse_padding_mask).sum(
                dim=1
            ) / reverse_padding_mask.sum(dim=1)
            # azimuth_hat = (azimuth_hat * reverse_padding_mask).sum(
            #     dim=1
            # ) / reverse_padding_mask.sum(dim=1)
            # elevation_hat = (elevation_hat * reverse_padding_mask).sum(
            #     dim=1
            # ) / reverse_padding_mask.sum(dim=1)

        else:
            # Collapse the time dimension
            Th_hat = Th_hat.mean(dim=1)
            Tt_hat = Tt_hat.mean(dim=1)
            volume_hat = volume_hat.mean(dim=1)
            dist_src_hat = dist_src_hat.mean(dim=1)
            # azimuth_hat = azimuth_hat.mean(dim=1)
            # elevation_hat = elevation_hat.mean(dim=1)

        # ------------------- inverse unitary normalization -------------------
        Th_hat = unitary_norm_inv(Th_hat, lb=0.005, ub=0.276)
        Th = unitary_norm_inv(Th, lb=0.005, ub=0.276)
        volume_hat = unitary_norm_inv(volume_hat, lb=1.5051, ub=3.9542)
        volume = unitary_norm_inv(volume, lb=1.5051, ub=3.9542)
        dist_src_hat = unitary_norm_inv(dist_src_hat, lb=0.191, ub=28.350)
        dist_src = unitary_norm_inv(dist_src, lb=0.191, ub=28.350)
        # azimuth_hat = unitary_norm_inv(azimuth_hat, lb=-1.000, ub=1.000)
        # azimuth = unitary_norm_inv(azimuth, lb=-1.000, ub=1.000)
        # elevation_hat = unitary_norm_inv(elevation_hat, lb=-0.733, ub=0.486)
        # elevation = unitary_norm_inv(elevation, lb=-0.733, ub=0.486)

        # MAE loss
        loss_Th = self.joint_loss_test(Th_hat, Th)
        loss_Tt = self.joint_loss_test(Tt_hat, Tt)
        loss_volume = self.joint_loss_test(volume_hat, volume)
        loss_dist_src = self.joint_loss_test(dist_src_hat, dist_src)
        # loss_azimuth = self.joint_loss_test(azimuth_hat, azimuth)
        # loss_elevation = self.joint_loss_test(elevation_hat, elevation)

        # correlation coefficient loss
        loss_Th_corrcoef = self.joint_corrcoef_test(Th_hat, Th)
        loss_Tt_corrcoef = self.joint_corrcoef_test(Tt_hat, Tt)
        loss_volume_corrcoef = self.joint_corrcoef_test(volume_hat, volume)
        loss_dist_src_corrcoef = self.joint_corrcoef_test(dist_src_hat, dist_src)
        # loss_azimuth_src_corrcoef = self.joint_corrcoef_test(azimuth_hat, azimuth)
        # loss_elevation_src_corrcoef = self.joint_corrcoef_test(
        #     elevation_hat,
        #     elevation,
        # )

        # update and log metrics
        self.test_loss_Th(loss_Th)
        self.test_loss_Tt(loss_Tt)
        self.test_loss_volume(loss_volume)
        self.test_loss_dist_src(loss_dist_src)
        # self.test_loss_azimuth(loss_azimuth)
        # self.test_loss_elevation(loss_elevation)

        self.test_corrcoef_Th(loss_Th_corrcoef)
        self.test_corrcoef_Tt(loss_Tt_corrcoef)
        self.test_corrcoef_volume(loss_volume_corrcoef)
        self.test_corrcoef_dist_src(loss_dist_src_corrcoef)
        # self.test_corrcoef_azimuth(loss_azimuth_src_corrcoef)
        # self.test_corrcoef_elevation(loss_elevation_src_corrcoef)

        self.log(
            "test/loss/Th",
            self.test_loss_Th,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "test/corrcoef/Th",
            self.test_corrcoef_Th,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "test/loss/Tt",
            self.test_loss_Tt,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "test/corrcoef/Tt",
            self.test_corrcoef_Tt,
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

        # self.log(
        #     "test/loss/azimuth",
        #     self.test_loss_azimuth,
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=True,
        #     sync_dist=True,
        # )

        # self.log(
        #     "test/corrcoef/azimuth",
        #     self.test_corrcoef_azimuth,
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=True,
        #     sync_dist=True,
        # )

        # self.log(
        #     "test/loss/elevation",
        #     self.test_loss_elevation,
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=True,
        #     sync_dist=True,
        # )

        # self.log(
        #     "test/corrcoef/elevation",
        #     self.test_corrcoef_elevation,
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=True,
        #     sync_dist=True,
        # )

        total_loss = (
            loss_Th
            + loss_Tt
            + loss_volume
            + loss_dist_src
            # + loss_azimuth
            # + loss_elevation
        )

        self.log(
            "test/loss/total",
            total_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        # Th_pred = self.model_step(batch)
        pass

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configures optimizers and learning-rate schedulers to be used for training.

        Normally you'd need one, but in the case of GANs or similar you might need multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(self.rirRegressor.parameters())

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
