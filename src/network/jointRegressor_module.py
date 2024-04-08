from typing import Any, Dict, Optional

import torch
from lightning import LightningModule
from torchmetrics import (
    MinMetric,
    MaxMetric,
    MeanMetric,
)
from torchmetrics.classification.accuracy import BinaryAccuracy

from src.utils.unitary_linear_norm import unitary_norm_inv
from src.criterions.polynomial_see_saw_loss import PolynomialSeeSawLoss

# ======================== joint regression module ========================


class JointRegressorModule(LightningModule):
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

        self.jointRegressor = net

        # loss function
        self.criterion_train = PolynomialSeeSawLoss(phase="train")
        self.criterion_val = PolynomialSeeSawLoss(phase="val")
        self.criterion_test = PolynomialSeeSawLoss(phase="test")

        self.Th_weight = self.hparams.optim_cfg.Th_weight
        self.Tt_weight = self.hparams.optim_cfg.Tt_weight
        self.volume_weight = self.hparams.optim_cfg.volume_weight
        self.dist_src_weight = self.hparams.optim_cfg.dist_src_weight
        self.ori_src_weight = self.hparams.optim_cfg.ori_src_weight
        self.azimuth_weight = self.hparams.optim_cfg.azimuth_weight
        self.elevation_weight = self.hparams.optim_cfg.elevation_weight
        self.bc_ori_src_weight = self.hparams.optim_cfg.bc_ori_src_weight
        self.bc_ori_src_weight_alt = self.hparams.optim_cfg.bc_ori_src_weight_alt

        # for averaging loss across batches
        self.train_huber_Th = MeanMetric()
        self.val_l1_Th = MeanMetric()
        self.test_l1_Th = MeanMetric()

        self.train_huber_Tt = MeanMetric()
        self.val_l1_Tt = MeanMetric()
        self.test_l1_Tt = MeanMetric()

        self.train_huber_volume = MeanMetric()
        self.val_l1_volume = MeanMetric()
        self.test_l1_volume = MeanMetric()

        self.train_huber_dist_src = MeanMetric()
        self.val_l1_dist_src = MeanMetric()
        self.test_l1_dist_src = MeanMetric()

        self.train_huber_azimuth = MeanMetric()
        self.val_l1_azimuth = MeanMetric()
        self.test_l1_azimuth = MeanMetric()

        self.train_huber_elevation = MeanMetric()
        self.val_l1_elevation = MeanMetric()
        self.test_l1_elevation = MeanMetric()

        # for tracking bias correction loss of azimuth and elevation
        self.train_bce_azimuth_bc = MeanMetric()
        self.val_bce_azimuth_bc = MeanMetric()
        self.test_bce_azimuth_bc = MeanMetric()

        self.train_bce_elevation_bc = MeanMetric()
        self.val_bce_elevation_bc = MeanMetric()
        self.test_bce_elevation_bc = MeanMetric()

        # for tracking bias correction accuracy of azimuth and elevation
        self.train_acc_azimuth_bc = BinaryAccuracy()
        self.val_acc_azimuth_bc = BinaryAccuracy()
        self.test_acc_azimuth_bc = BinaryAccuracy()

        self.train_acc_elevation_bc = BinaryAccuracy()
        self.val_acc_elevation_bc = BinaryAccuracy()
        self.test_acc_elevation_bc = BinaryAccuracy()

        # for tracking validation correlation coefficient of Th, Tt, volume, distSrc, azimuth, elevationSrc
        self.val_corrcoef_Th = MeanMetric()
        self.val_corrcoef_Tt = MeanMetric()
        self.val_corrcoef_volume = MeanMetric()
        self.val_corrcoef_dist_src = MeanMetric()
        self.val_corrcoef_azimuth = MeanMetric()
        self.val_corrcoef_elevation = MeanMetric()

        # for tracking evaluation correlation coefficient of Th, Tt, volume, distSrc, azimuth, elevationSrc
        self.test_corrcoef_Th = MeanMetric()
        self.test_corrcoef_Tt = MeanMetric()
        self.test_corrcoef_volume = MeanMetric()
        self.test_corrcoef_dist_src = MeanMetric()
        self.test_corrcoef_azimuth = MeanMetric()
        self.test_corrcoef_elevation = MeanMetric()

        # for tracking best so far validation best metrics
        self.val_l1_best_Th = MinMetric()
        self.val_l1_best_Tt = MinMetric()
        self.val_l1_best_volume = MinMetric()
        self.val_l1_best_dist_src = MinMetric()
        self.val_l1_best_azimuth = MinMetric()
        self.val_l1_best_elevation = MinMetric()
        self.val_bce_best_azimuth_bc = MinMetric()
        self.val_bce_best_elevation_bc = MinMetric()

        self.val_acc_best_azimuth_bc = MaxMetric()
        self.val_acc_best_elevation_bc = MaxMetric()

        # for tracking best of correlation coefficient
        self.val_corrcoef_best_Th = MaxMetric()
        self.val_corrcoef_best_Tt = MaxMetric()
        self.val_corrcoef_best_volume = MaxMetric()
        self.val_corrcoef_best_dist_src = MaxMetric()
        self.val_corrcoef_best_azimuth = MaxMetric()
        self.val_corrcoef_best_elevation = MaxMetric()

        self.predict_tools = PolynomialSeeSawLoss(phase="infer")

    def forward(
        self,
        source: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Perform a forward pass through freezed denoiser and rirRegressor.

        :param x: A tensor of waveform
        :return: A tensor of estimated Th, Tt, volume, distSrc, azimuth, elevationSrc.
        """

        net_output = self.jointRegressor(source, padding_mask)

        return net_output

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_l1_Th.reset()
        self.val_l1_Tt.reset()
        self.val_l1_volume.reset()
        self.val_l1_dist_src.reset()
        self.val_l1_azimuth.reset()
        self.val_l1_elevation.reset()
        self.val_bce_azimuth_bc.reset()
        self.val_bce_azimuth_bc.reset()

        self.val_corrcoef_Th.reset()
        self.val_corrcoef_Tt.reset()
        self.val_corrcoef_volume.reset()
        self.val_corrcoef_dist_src.reset()
        self.val_corrcoef_azimuth.reset()
        self.val_corrcoef_elevation.reset()

        self.val_acc_azimuth_bc.reset()
        self.val_acc_elevation_bc.reset()

        self.val_l1_best_Th.reset()
        self.val_l1_best_Tt.reset()
        self.val_l1_best_volume.reset()
        self.val_l1_best_dist_src.reset()
        self.val_l1_best_azimuth.reset()
        self.val_l1_best_elevation.reset()
        self.val_bce_best_azimuth_bc.reset()
        self.val_bce_best_azimuth_bc.reset()

        self.val_corrcoef_best_Th.reset()
        self.val_corrcoef_best_Tt.reset()
        self.val_corrcoef_best_volume.reset()
        self.val_corrcoef_best_dist_src.reset()
        self.val_corrcoef_best_azimuth.reset()
        self.val_corrcoef_best_elevation.reset()

        self.val_acc_best_azimuth_bc.reset()
        self.val_acc_best_elevation_bc.reset()

    def model_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a dict) containing the input tensor
            of raw, target Th, target Tt, target volume, target distSrc,
            target azimuth, target elevationSrc.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """

        # network forward pass
        net_output = self.forward(**batch["net_input"])

        loss = self.criterion_train(net_output, batch["groundtruth"])

        loss_Th = loss["loss_Th"]
        loss_Tt = loss["loss_Tt"]
        loss_volume = loss["loss_volume"]
        loss_dist_src = loss["loss_dist_src"]
        loss_ori_src = loss["loss_ori_src"]
        pred_label_ori_src = loss["pred_label_ori_src"]

        return (
            loss_Th,
            loss_Tt,
            loss_volume,
            loss_dist_src,
            loss_ori_src,
            pred_label_ori_src,
        )

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of raw,
            target Th, target Tt, target volume, target distSrc, target azimuth,
            target elevationSrc.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        (
            loss_Th,
            loss_Tt,
            loss_volume,
            loss_dist_src,
            loss_ori_src,
            pred_label_ori_src,
        ) = self.model_step(batch)

        # -------------------azimuth and elevation bias correction-------------------

        # binary cross entropy loss of azimuth and elevation bias corrector
        bce_loss_azimuth = loss_ori_src["loss_azimuth_bc"][0]
        bce_loss_elevation = loss_ori_src["loss_elevation_bc"][0]

        # update bias corrector bce loss
        self.train_bce_azimuth_bc(bce_loss_azimuth)
        self.train_bce_elevation_bc(bce_loss_elevation)

        # accuracy of azimuth and elevation bias correction
        target_azimuth_label = batch["groundtruth"]["azimuth_classif"].long()
        target_elevation_label = batch["groundtruth"]["elevation_classif"].long()

        # update bias correction accuracy
        self.train_acc_azimuth_bc(
            pred_label_ori_src["pred_azimuth_label"], target_azimuth_label
        )
        self.train_acc_elevation_bc(
            pred_label_ori_src["pred_elevation_label"], target_elevation_label
        )

        # ----------------- azimuth and elevation estimation -----------------
        # determine whether to use azimuth and elevation estimation
        # if judge_prob_azimuth >= 0.4, use azimuth estimation
        if len(loss_ori_src["loss_azimuth_pp"]) > 0:
            huber_loss_azimuth = loss_ori_src["loss_azimuth_pp"][0]

            # update azimuth huber loss
            self.train_huber_azimuth(huber_loss_azimuth)

        else:
            huber_loss_azimuth = torch.tensor(1e-6).to(bce_loss_azimuth.device)

        # if judge_prob_elevation >= 0.4, use elevation estimation
        if len(loss_ori_src["loss_elevation_pp"]) > 0:
            huber_loss_elevation = loss_ori_src["loss_elevation_pp"][0]

            # update elevation huber loss
            self.train_huber_elevation(huber_loss_elevation)

        else:
            huber_loss_elevation = torch.tensor(1e-6).to(bce_loss_elevation.device)

        bce_loss_ori_src = self.bc_ori_src_weight * (
            bce_loss_azimuth + bce_loss_elevation
        )

        loss_ori_src = bce_loss_ori_src + (
            (
                self.ori_src_weight
                * (
                    self.azimuth_weight * huber_loss_azimuth
                    + self.elevation_weight * huber_loss_elevation
                )
            )
            / (1 + self.bc_ori_src_weight_alt * bce_loss_ori_src)
        )  # see-saw loss for balancing the four losses of orientation estimation

        # ------------------- other parameters -------------------

        # update huber loss of Th, Tt, volume, distSrc
        self.train_huber_Th(loss_Th)
        self.train_huber_Tt(loss_Tt)
        self.train_huber_volume(loss_volume)
        self.train_huber_dist_src(loss_dist_src)

        # ------------------- total loss -------------------
        total_loss = (
            self.Th_weight * loss_Th
            + self.Tt_weight * loss_Tt
            + self.volume_weight * loss_volume
            + self.dist_src_weight * loss_dist_src
            + loss_ori_src
        )

        # ------------------- log metrics -------------------
        self.log(
            "train/loss/Th",
            self.train_huber_Th,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "train/loss/Tt",
            self.train_huber_Tt,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "train/loss/volume",
            self.train_huber_volume,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "train/loss/dist_src",
            self.train_huber_dist_src,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "train/loss/azimuth",
            self.train_huber_azimuth,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "train/loss/elevation",
            self.train_huber_elevation,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "train/loss/BC_loss_azimuth",
            self.train_bce_azimuth_bc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "train/loss/BC_loss_elevation",
            self.train_bce_elevation_bc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "train/acc/BC_azimuth",
            self.train_acc_azimuth_bc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "train/acc/BC_elevation",
            self.train_acc_elevation_bc,
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
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a dict) containing the input tensor of raw,
            target Th, target Tt, target volume, target distSrc, target azimuth,
            target elevationSrc.
        :param batch_idx: The index of the current batch.
        """

        # network forward pass
        net_output = self.forward(**batch["net_input"])

        loss = self.criterion_val(net_output, batch["groundtruth"])

        loss_Th = loss["loss_Th"]
        loss_Tt = loss["loss_Tt"]
        loss_volume = loss["loss_volume"]
        loss_dist_src = loss["loss_dist_src"]
        loss_ori_src = loss["loss_ori_src"]
        pred_label_ori_src = loss["pred_label_ori_src"]
        corr_Th = loss["corr_Th"]
        corr_Tt = loss["corr_Tt"]
        corr_volume = loss["corr_volume"]
        corr_dist_src = loss["corr_dist_src"]
        corr_ori_src = loss["corr_ori_src"]

        # -------------------azimuth and elevation bias correction-------------------
        bce_loss_azimuth = loss_ori_src["loss_azimuth_bc"][0]
        bce_loss_elevation = loss_ori_src["loss_elevation_bc"][0]

        # update and log metrics of bias correction
        self.val_bce_azimuth_bc(bce_loss_azimuth)
        self.val_bce_elevation_bc(bce_loss_elevation)

        # accuracy of azimuth and elevation bias correction
        target_azimuth_label = batch["groundtruth"]["azimuth_classif"].long()
        target_elevation_label = batch["groundtruth"]["elevation_classif"].long()

        pred_azimuth_label = pred_label_ori_src["pred_azimuth_label"]
        pred_elevation_label = pred_label_ori_src["pred_elevation_label"]

        # update bias correction accuracy of azimuth and elevation
        self.val_acc_azimuth_bc(pred_azimuth_label, target_azimuth_label)
        self.val_acc_elevation_bc(pred_elevation_label, target_elevation_label)

        # ----------------- azimuth and elevation estimation -----------------

        # determine whether to use azimuth and elevation estimation
        if len(loss_ori_src["loss_azimuth_pp"]) > 0:

            l1_loss_azimuth = loss_ori_src["loss_azimuth_pp"][0]

            self.val_l1_azimuth(l1_loss_azimuth)

        else:
            l1_loss_azimuth = torch.tensor(1e-6).to(bce_loss_azimuth.device)

        if len(corr_ori_src["corr_azimuth_pp_val"]) > 0:

            corr_azimuth = corr_ori_src["corr_azimuth_pp_val"][0]

            self.val_corrcoef_azimuth(corr_azimuth)

        # if judge_prob_elevation > 0.5, pass through elevation regressor
        if len(loss_ori_src["loss_elevation_pp"]) > 0:

            l1_loss_elevation = loss_ori_src["loss_elevation_pp"][0]

            self.val_l1_elevation(l1_loss_elevation)

        else:
            l1_loss_elevation = torch.tensor(1e-6).to(bce_loss_elevation.device)

        if len(corr_ori_src["corr_elevation_pp_val"]) > 0:

            corr_elevation = corr_ori_src["corr_elevation_pp_val"][0]

            self.val_corrcoef_elevation(corr_elevation)

        bce_loss_ori_src = self.bc_ori_src_weight * (
            bce_loss_azimuth + bce_loss_elevation
        )

        # see-saw loss of azimuth and elevation estimation
        loss_ori_src = bce_loss_ori_src + (
            self.ori_src_weight
            * (
                self.azimuth_weight * l1_loss_azimuth
                + self.elevation_weight * l1_loss_elevation
            )
        ) / (
            1 + self.bc_ori_src_weight_alt * bce_loss_ori_src
        )  # see-saw loss for balancing the four losses of azimuth and elevation estimation

        # ------------------- other parameters -------------------

        self.val_l1_Th(loss_Th)
        self.val_l1_Tt(loss_Tt)
        self.val_l1_volume(loss_volume)
        self.val_l1_dist_src(loss_dist_src)

        self.val_corrcoef_Th(corr_Th)
        self.val_corrcoef_Tt(corr_Tt)
        self.val_corrcoef_volume(corr_volume)
        self.val_corrcoef_dist_src(corr_dist_src)

        # ------------------- total loss -------------------

        total_loss = (
            self.Th_weight * loss_Th
            + self.Tt_weight * loss_Tt
            + self.volume_weight * loss_volume
            + self.dist_src_weight * loss_dist_src
            + loss_ori_src
        )

        # ------------------- log metrics -------------------
        self.log(
            "val/loss/Th",
            self.val_l1_Th,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "val/corrcoef/Th_corrcoef",
            self.val_corrcoef_Th,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "val/loss/Tt",
            self.val_l1_Tt,
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
            self.val_l1_volume,
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
            self.val_l1_dist_src,
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
            "val/loss/azimuth",
            self.val_l1_azimuth,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "val/corrcoef/azimuth",
            self.val_corrcoef_azimuth,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "val/loss/elevation",
            self.val_l1_elevation,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "val/corrcoef/elevation",
            self.val_corrcoef_elevation,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "val/loss/BC_azimuth",
            self.val_bce_azimuth_bc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "val/loss/BC_elevation",
            self.val_bce_elevation_bc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "val/acc/BC_azimuth",
            self.val_acc_azimuth_bc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "val/acc/BC_elevation",
            self.val_acc_elevation_bc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "val/loss/loss_ori_src",
            loss_ori_src,
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
        metric_l1_Th = self.val_l1_Th.compute()  # get current val metric
        metric_l1_Tt = self.val_l1_Tt.compute()  # get current val metric
        metric_l1_volume = self.val_l1_volume.compute()  # get current val metric
        metric_l1_distSrc = self.val_l1_dist_src.compute()  # get current val metric

        # ------------- best so far validation correlation coefficient ------------
        metric_Th_corrcoef = self.val_corrcoef_Th.compute()
        metric_Tt_corrcoef = self.val_corrcoef_Tt.compute()
        metric_volume_corrcoef = self.val_corrcoef_volume.compute()
        metric_dist_src_corrcoef = self.val_corrcoef_dist_src.compute()

        # -------------------azimuth and elevation bias correction-------------------
        metric_acc_azimuth_bc = (
            self.val_acc_azimuth_bc.compute()
        )  # get current val metric of binary classifier (bias correction)
        metric_acc_elevation_bc = (
            self.val_acc_elevation_bc.compute()
        )  # get current val metric of binary classifier (bias correction)

        metric_loss_azimuth_bc = (
            self.val_bce_azimuth_bc.compute()
        )  # get current val metric of bias correction (azimuth)
        metric_loss_elevation_bc = (
            self.val_bce_elevation_bc.compute()
        )  # get current val metric of bias correction (elevationSrc)

        # -------------- azimuth and elevation estimation --------------
        metric_l1_azimuth = (
            self.val_l1_azimuth.compute()
        )  # get current val metric of azimuth
        metric_l1_elevation = (
            self.val_l1_elevation.compute()
        )  # get current val metric of elevationSrc

        metric_corrcoef_azimuth = (
            self.val_corrcoef_azimuth.compute()
        )  # get current val correlation coefficient of azimuth
        metric_corrcoef_elevation = (
            self.val_corrcoef_elevation.compute()
        )  # get current val correlation coefficient of elevationSrc

        # ------- update best so far val metric of Th, Tt, volume, distSrc -------
        # update best so far val metric of loss
        self.val_l1_best_Th(metric_l1_Th)  # update best so far val metric
        self.val_l1_best_Tt(metric_l1_Tt)  # update best so far val metric
        self.val_l1_best_volume(metric_l1_volume)  # update best so far val metric
        self.val_l1_best_dist_src(metric_l1_distSrc)  # update best so far val metric

        # update best so far val metric of correlation coefficient
        self.val_corrcoef_best_Th(metric_Th_corrcoef)
        self.val_corrcoef_best_Tt(metric_Tt_corrcoef)
        self.val_corrcoef_best_volume(metric_volume_corrcoef)
        self.val_corrcoef_best_dist_src(metric_dist_src_corrcoef)

        # ------ update best so far val metric of bias correction -------
        self.val_acc_best_azimuth_bc(
            metric_acc_azimuth_bc
        )  # update best so far acc metric of bias correction (azimuth)
        self.val_acc_best_elevation_bc(
            metric_acc_elevation_bc
        )  # update best so far acc metric of bias correction (elevationSrc)

        self.val_bce_best_azimuth_bc(
            metric_loss_azimuth_bc
        )  # update best so far val metric of bias correction (azimuth)
        self.val_bce_best_elevation_bc(
            metric_loss_elevation_bc
        )  # update best so far val metric of bias correction (elevationSrc)

        # ------ update best so far val metric of azimuth and elevation estimation -------
        self.val_l1_best_azimuth(
            metric_l1_azimuth
        )  # update best so far val metric of azimuth
        self.val_l1_best_elevation(
            metric_l1_elevation
        )  # update best so far val metric of elevationSrc

        self.val_corrcoef_best_azimuth(
            metric_corrcoef_azimuth
        )  # update best so far val correlation coefficient of azimuth
        self.val_corrcoef_best_elevation(
            metric_corrcoef_elevation
        )  # update best so far val correlation coefficient of elevationSrc

        val_loss_ori_src_bc_best = self.bc_ori_src_weight * (
            self.val_bce_best_azimuth_bc + self.val_bce_best_elevation_bc
        )

        val_loss_ori_src_best = self.ori_src_weight * (
            self.azimuth_weight * self.val_l1_best_azimuth
            + self.elevation_weight * self.val_l1_best_elevation
        )

        val_loss_best_oriSrc = val_loss_ori_src_bc_best + (
            val_loss_ori_src_best
            / (1 + self.bc_ori_src_weight_alt * val_loss_ori_src_bc_best)
        )

        # ------------------- total best so far validation loss -------------------

        total_loss_best = (
            self.Th_weight * self.val_l1_best_Th
            + self.Tt_weight * self.val_l1_best_Tt
            + self.volume_weight * self.val_l1_best_volume
            + self.dist_src_weight * self.val_l1_best_dist_src
            + val_loss_best_oriSrc
        )

        # log `val_loss_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch

        # ------------------- log best so far validation loss -------------------

        self.log(
            "val/loss_best/Th",
            self.val_l1_best_Th.compute(),
            sync_dist=True,
            prog_bar=True,
        )

        self.log(
            "val/loss_best/Tt",
            self.val_l1_best_Tt.compute(),
            sync_dist=True,
            prog_bar=True,
        )

        self.log(
            "val/loss_best/volume",
            self.val_l1_best_volume.compute(),
            sync_dist=True,
            prog_bar=True,
        )

        self.log(
            "val/loss_best/dist_src",
            self.val_l1_best_dist_src.compute(),
            sync_dist=True,
            prog_bar=True,
        )

        self.log(
            "val/loss_best/azimuth",
            self.val_l1_best_azimuth.compute(),
            sync_dist=True,
            prog_bar=True,
        )

        self.log(
            "val/loss_best/elevation",
            self.val_l1_best_elevation.compute(),
            sync_dist=True,
            prog_bar=True,
        )

        self.log(
            "val/loss_best/BC_azimuth",
            self.val_bce_best_azimuth_bc.compute(),
            sync_dist=True,
            prog_bar=True,
        )

        self.log(
            "val/loss_best/BC_elevation",
            self.val_bce_best_elevation_bc.compute(),
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

        self.log(
            "val/corrcoef_best/azimuth",
            self.val_corrcoef_best_azimuth.compute(),
            sync_dist=True,
            prog_bar=True,
        )

        self.log(
            "val/corrcoef_best/elevation",
            self.val_corrcoef_best_elevation.compute(),
            sync_dist=True,
            prog_bar=True,
        )

        # ------------------- log best so far validation bias correction ------------------
        self.log(
            "val/acc_best/BC_azimuth",
            self.val_acc_best_azimuth_bc.compute(),
            sync_dist=True,
            prog_bar=True,
        )

        self.log(
            "val/acc_best/BC_elevation",
            self.val_acc_best_elevation_bc.compute(),
            sync_dist=True,
            prog_bar=True,
        )

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

        # network forward pass
        net_output = self.forward(**batch["net_input"])

        loss = self.criterion_test(net_output, batch["groundtruth"])

        loss_Th = loss["loss_Th"]
        loss_Tt = loss["loss_Tt"]
        loss_volume = loss["loss_volume"]
        loss_dist_src = loss["loss_dist_src"]
        loss_azimuth = loss["loss_azimuth"]
        loss_elevation = loss["loss_elevation"]
        corrcoef_Th = loss["corr_Th"]
        corrcoef_Tt = loss["corr_Tt"]
        corrcoef_volume = loss["corr_volume"]
        corrcoef_dist_src = loss["corr_dist_src"]
        corrcoef_azimuth = loss["corr_azimuth"]
        corrcoef_elevation = loss["corr_elevation"]

        # update and log metrics
        self.test_l1_Th(loss_Th)
        self.test_l1_Tt(loss_Tt)
        self.test_l1_volume(loss_volume)
        self.test_l1_dist_src(loss_dist_src)
        self.test_l1_azimuth(loss_azimuth)
        self.test_l1_elevation(loss_elevation)

        self.test_corrcoef_Th(corrcoef_Th)
        self.test_corrcoef_Tt(corrcoef_Tt)
        self.test_corrcoef_volume(corrcoef_volume)
        self.test_corrcoef_dist_src(corrcoef_dist_src)
        self.test_corrcoef_azimuth(corrcoef_azimuth)
        self.test_corrcoef_elevation(corrcoef_elevation)

        self.log(
            "test/loss/Th",
            self.test_l1_Th,
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
            self.test_l1_Tt,
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
            self.test_l1_volume,
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
            self.test_l1_dist_src,
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

        self.log(
            "test/loss/azimuth",
            self.test_l1_azimuth,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "test/corrcoef/azimuth",
            self.test_corrcoef_azimuth,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "test/loss/elevation",
            self.test_l1_elevation,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "test/corrcoef/elevation",
            self.test_corrcoef_elevation,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        total_loss = (
            loss_Th
            + loss_Tt
            + loss_volume
            + loss_dist_src
            + loss_azimuth
            + loss_elevation
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

    def predict_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """Perform a single prediction step on a batch of data from the test set.

        :param batch: A batch of data (a dict) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.

        :return: A dict containing the model predictions.
        """
        net_output = self.forward(**batch["net_input"])

        Th_hat = net_output["Th_hat"]
        Tt_hat = net_output["Tt_hat"]
        volume_hat = net_output["volume_hat"]
        dist_src_hat = net_output["dist_src_hat"]
        azimuth_hat = net_output["azimuth_hat"]
        elevation_hat = net_output["elevation_hat"]
        padding_mask = net_output["padding_mask"]

        assert (
            Th_hat.shape
            == Tt_hat.shape
            == volume_hat.shape
            == dist_src_hat.shape
            == azimuth_hat.shape
            == elevation_hat.shape
        )

        if padding_mask is not None and padding_mask.any():
            # B x T
            input_lengths = (1 - padding_mask.long()).sum(-1)
            # apply conv formula to get real output_lengths
            output_lengths = self.predict_tools._get_param_pred_output_lengths(
                input_lengths=input_lengths
            )

            padding_mask = torch.zeros(
                Tt_hat.shape[:2], dtype=Tt_hat.dtype, device=Tt_hat.device
            )  # B x T

            # these two operations makes sure that all values
            # before the output lengths indices are attended to
            padding_mask[
                (
                    torch.arange(padding_mask.shape[0], device=padding_mask.device),
                    output_lengths - 1,
                )
            ] = 1
            padding_mask = (
                1 - padding_mask.flip([-1]).cumsum(-1).flip([-1])
            ).bool()  # for padded values, set to True

            reverse_padding_mask = padding_mask.logical_not()

        # ------------------- all predictions -------------------
        # collapse as a single value intead of a straitght-line prediction
        if padding_mask is not None and padding_mask.any():
            # ---------------------- padding mask handling ----------------------
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

            azimuth_hat = (azimuth_hat * reverse_padding_mask).sum(
                dim=1
            ) / reverse_padding_mask.sum(dim=1)

            elevation_hat = (elevation_hat * reverse_padding_mask).sum(
                dim=1
            ) / reverse_padding_mask.sum(dim=1)

        else:
            Th_hat = Th_hat.mean(dim=1)
            Tt_hat = Tt_hat.mean(dim=1)
            volume_hat = volume_hat.mean(dim=1)
            dist_src_hat = dist_src_hat.mean(dim=1)
            azimuth_hat = azimuth_hat.mean(dim=1)
            elevation_hat = elevation_hat.mean(dim=1)

        judge_prob_ori_src = self.jointRegressor.get_judge_prob(net_output)

        judge_prob_azimuth = judge_prob_ori_src["judge_prob_azimuth"]
        judge_prob_elevation = judge_prob_ori_src["judge_prob_elevation"]

        idx_azimuth_pp_false = torch.where(judge_prob_azimuth < 0.5)[0]
        if len(idx_azimuth_pp_false) > 0:
            azimuth_hat[idx_azimuth_pp_false] = torch.tensor(0.4986)

        idx_elevation_pp_false = torch.where(judge_prob_elevation < 0.5)[0]
        if len(idx_azimuth_pp_false) > 0:
            elevation_hat[idx_elevation_pp_false] = torch.tensor(0.5977)

        # inverse unitary normalization
        Th_hat = unitary_norm_inv(Th_hat, lb=0.005, ub=0.276)
        volume_hat = unitary_norm_inv(volume_hat, lb=1.5051, ub=3.9542)
        dist_src_hat = unitary_norm_inv(dist_src_hat, lb=0.191, ub=28.350)
        azimuth_hat = unitary_norm_inv(azimuth_hat, lb=-1.000, ub=1.000)
        elevation_hat = unitary_norm_inv(elevation_hat, lb=-0.733, ub=0.486)
        azimuth_hat = azimuth_hat * torch.pi  # convert to radian
        elevation_hat = elevation_hat * torch.pi  # convert to radian

        preds = {"Th_hat": Th_hat}
        preds["Tt_hat"] = Tt_hat
        preds["volume_hat"] = volume_hat
        preds["dist_src_hat"] = dist_src_hat
        preds["azimuth_hat"] = azimuth_hat
        preds["elevation_hat"] = elevation_hat

        return preds

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configures optimizers and learning-rate schedulers to be used for training.

        Normally you'd need one, but in the case of GANs or similar you might need multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(self.jointRegressor.parameters())

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

    _ = JointRegressorModule(cfg.model.jointRegressorModule)
