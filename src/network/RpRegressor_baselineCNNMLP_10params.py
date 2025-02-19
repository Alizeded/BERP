from typing import Any, Optional

import torch
from lightning import LightningModule
from torch.nn import L1Loss
from torchmetrics import MeanMetric, MinMetric, PearsonCorrCoef

from src.criterions.nll_loss import NLLCriterion

# ======================== joint regression module ========================


class JointRegressorModuleBaselineCNNMLP(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        optim_cfg: Optional[dict] = None,
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
        self.joint_loss = NLLCriterion()
        self.joint_loss_val = L1Loss()
        self.joint_loss_test = L1Loss()
        self.joint_corrcoef_test = PearsonCorrCoef()

        # for averaging loss across batches
        self.train_loss_sti = MeanMetric()
        self.val_loss_sti = MeanMetric()
        self.test_loss_sti = MeanMetric()

        self.train_loss_alcons = MeanMetric()
        self.val_loss_alcons = MeanMetric()
        self.test_loss_alcons = MeanMetric()

        self.train_loss_t60 = MeanMetric()
        self.val_loss_t60 = MeanMetric()
        self.test_loss_t60 = MeanMetric()

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

        # for tracking evaluation correlation coefficient
        self.test_corrcoef_sti = MeanMetric()
        self.test_corrcoef_alcons = MeanMetric()
        self.test_corrcoef_t60 = MeanMetric()
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
        self.val_loss_best_t60 = MinMetric()
        self.val_loss_best_edt = MinMetric()
        self.val_loss_best_c80 = MinMetric()
        self.val_loss_best_c50 = MinMetric()
        self.val_loss_best_d50 = MinMetric()
        self.val_loss_best_ts = MinMetric()
        self.val_loss_best_volume = MinMetric()
        self.val_loss_best_dist_src = MinMetric()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Perform a forward pass through freezed denoiser and rirRegressor.

        :param x: A tensor of waveform
        :return: A tensor of estimated Th, Tt, volume, distSrc, azimuthSrc, elevationSrc.
        """

        return self.rirRegressor(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss_sti.reset()
        self.val_loss_alcons.reset()
        self.val_loss_t60.reset()
        self.val_loss_edt.reset()
        self.val_loss_c80.reset()
        self.val_loss_c50.reset()
        self.val_loss_d50.reset()
        self.val_loss_ts.reset()
        self.val_loss_volume.reset()
        self.val_loss_dist_src.reset()
        self.val_loss_best_sti.reset()
        self.val_loss_best_alcons.reset()
        self.val_loss_best_t60.reset()
        self.val_loss_best_edt.reset()
        self.val_loss_best_c80.reset()
        self.val_loss_best_c50.reset()
        self.val_loss_best_d50.reset()
        self.val_loss_best_ts.reset()
        self.val_loss_best_volume.reset()
        self.val_loss_best_dist_src.reset()

    def model_step(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a dict) containing the input tensor
            of raw, target Th, target Tt, target volume, target distSrc,
            target azimuthSrc, target elevationSrc.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        raw = batch["raw"]
        sti = batch["sti"]
        alcons = batch["alcons"]
        t60 = batch["t60"]
        edt = batch["edt"]
        c80 = batch["c80"]
        c50 = batch["c50"]
        d50 = batch["d50"]
        ts = batch["ts"]
        volume = batch["volume"]
        dist_src = batch["dist_src"]

        mu_hat, var_hat = self.forward(raw)

        sti_hat = mu_hat[:, 0]
        alcons_hat = mu_hat[:, 1]
        t60_hat = mu_hat[:, 2]
        edt_hat = mu_hat[:, 3]
        c80_hat = mu_hat[:, 4]
        c50_hat = mu_hat[:, 5]
        d50_hat = mu_hat[:, 6]
        ts_hat = mu_hat[:, 7]
        volume_hat = mu_hat[:, 8]
        dist_src_hat = mu_hat[:, 9]

        sti_var = var_hat[:, 0]
        alcons_var = var_hat[:, 1]
        t60_var = var_hat[:, 2]
        edt_var = var_hat[:, 3]
        c80_var = var_hat[:, 4]
        c50_var = var_hat[:, 5]
        d50_var = var_hat[:, 6]
        ts_var = var_hat[:, 7]
        volume_var = var_hat[:, 8]
        dist_src_var = var_hat[:, 9]

        loss_sti = self.joint_loss(sti, sti_hat, sti_var)
        loss_alcons = self.joint_loss(alcons, alcons_hat, alcons_var)
        loss_t60 = self.joint_loss(t60, t60_hat, t60_var)
        loss_edt = self.joint_loss(edt, edt_hat, edt_var)
        loss_c80 = self.joint_loss(c80, c80_hat, c80_var)
        loss_c50 = self.joint_loss(c50, c50_hat, c50_var)
        loss_d50 = self.joint_loss(d50, d50_hat, d50_var)
        loss_ts = self.joint_loss(ts, ts_hat, ts_var)
        loss_volume = self.joint_loss(volume, volume_hat, volume_var)
        loss_dist_src = self.joint_loss(dist_src, dist_src_hat, dist_src_var)

        return (
            loss_sti,
            loss_alcons,
            loss_t60,
            loss_edt,
            loss_c80,
            loss_c50,
            loss_d50,
            loss_ts,
            loss_volume,
            loss_dist_src,
        )

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
        (
            loss_sti,
            loss_alcons,
            loss_t60,
            loss_edt,
            loss_c80,
            loss_c50,
            loss_d50,
            loss_ts,
            loss_volume,
            loss_dist_src,
        ) = self.model_step(batch)

        # update and log metrics
        self.train_loss_sti(loss_sti)
        self.train_loss_alcons(loss_alcons)
        self.train_loss_t60(loss_t60)
        self.train_loss_edt(loss_edt)
        self.train_loss_c80(loss_c80)
        self.train_loss_c50(loss_c50)
        self.train_loss_d50(loss_d50)
        self.train_loss_ts(loss_ts)
        self.train_loss_volume(loss_volume)
        self.train_loss_dist_src(loss_dist_src)

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
            "train/loss/t60",
            self.train_loss_t60,
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

        total_loss = (
            loss_sti
            + loss_alcons
            + loss_t60
            + loss_edt
            + loss_c80
            + loss_c50
            + loss_d50
            + loss_ts
            + loss_volume
            + loss_dist_src
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
        raw = batch["raw"]
        sti = batch["sti"]
        alcons = batch["alcons"]
        t60 = batch["t60"]
        edt = batch["edt"]
        c80 = batch["c80"]
        c50 = batch["c50"]
        d50 = batch["d50"]
        ts = batch["ts"]
        volume = batch["volume"]
        dist_src = batch["dist_src"]

        mu_hat, _ = self.forward(raw)

        sti_hat = mu_hat[:, 0]
        alcons_hat = mu_hat[:, 1]
        t60_hat = mu_hat[:, 2]
        edt_hat = mu_hat[:, 3]
        c80_hat = mu_hat[:, 4]
        c50_hat = mu_hat[:, 5]
        d50_hat = mu_hat[:, 6]
        ts_hat = mu_hat[:, 7]
        volume_hat = mu_hat[:, 8]
        dist_src_hat = mu_hat[:, 9]

        # update and log metrics
        loss_sti = self.joint_loss_val(sti, sti_hat)
        loss_alcons = self.joint_loss_val(alcons, alcons_hat)
        loss_t60 = self.joint_loss_val(t60, t60_hat)
        loss_edt = self.joint_loss_val(edt, edt_hat)
        loss_c80 = self.joint_loss_val(c80, c80_hat)
        loss_c50 = self.joint_loss_val(c50, c50_hat)
        loss_d50 = self.joint_loss_val(d50, d50_hat)
        loss_ts = self.joint_loss_val(ts, ts_hat)
        loss_volume = self.joint_loss_val(volume, volume_hat)
        loss_dist_src = self.joint_loss_val(dist_src, dist_src_hat)

        self.val_loss_sti(loss_sti)
        self.val_loss_alcons(loss_alcons)
        self.val_loss_t60(loss_t60)
        self.val_loss_edt(loss_edt)
        self.val_loss_c80(loss_c80)
        self.val_loss_c50(loss_c50)
        self.val_loss_d50(loss_d50)
        self.val_loss_ts(loss_ts)
        self.val_loss_volume(loss_volume)
        self.val_loss_dist_src(loss_dist_src)

        self.log(
            "val/loss/sti",
            self.val_loss_sti,
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
            "val/loss/t60",
            self.val_loss_t60,
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
            "val/loss/c80",
            self.val_loss_c80,
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
            "val/loss/d50",
            self.val_loss_d50,
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
            "val/loss/volume",
            self.val_loss_volume,
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

        total_loss = (
            loss_sti
            + loss_alcons
            + loss_t60
            + loss_edt
            + loss_c80
            + loss_c50
            + loss_d50
            + loss_ts
            + loss_volume
            + loss_dist_src
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
        metric_sti = self.val_loss_sti.compute()  # get current val metric
        metric_alcons = self.val_loss_alcons.compute()  # get current val metric
        metric_t60 = self.val_loss_t60.compute()  # get current val metric
        metric_edt = self.val_loss_edt.compute()
        metric_c80 = self.val_loss_c80.compute()
        metric_c50 = self.val_loss_c50.compute()
        metric_d50 = self.val_loss_d50.compute()
        metric_ts = self.val_loss_ts.compute()
        metric_volume = self.val_loss_volume.compute()
        metric_dist_src = self.val_loss_dist_src.compute()

        self.val_loss_best_sti(metric_sti)  # update best so far val metric
        self.val_loss_best_alcons(metric_alcons)  # update best so far val metric
        self.val_loss_best_t60(metric_t60)  # update best so far val metric
        self.val_loss_best_edt(metric_edt)
        self.val_loss_best_c80(metric_c80)
        self.val_loss_best_c50(metric_c50)
        self.val_loss_best_d50(metric_d50)
        self.val_loss_best_ts(metric_ts)
        self.val_loss_best_volume(metric_volume)
        self.val_loss_best_dist_src(metric_dist_src)

        total_loss_best = (
            self.val_loss_best_sti
            + self.val_loss_best_alcons
            + self.val_loss_best_t60
            + self.val_loss_best_edt
            + self.val_loss_best_c80
            + self.val_loss_best_c50
            + self.val_loss_best_d50
            + self.val_loss_best_ts
            + self.val_loss_best_volume
            + self.val_loss_best_dist_src
        )

        # log `val_loss_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log(
            "val/loss_best/total",
            total_loss_best.compute(),
            sync_dist=True,
            prog_bar=True,
        )

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
            self.val_loss_best_t60.compute(),
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
        raw = batch["raw"]
        sti = batch["sti"]
        alcons = batch["alcons"]
        t60 = batch["t60"]
        edt = batch["edt"]
        c80 = batch["c80"]
        c50 = batch["c50"]
        d50 = batch["d50"]
        ts = batch["ts"]
        volume = batch["volume"]
        dist_src = batch["dist_src"]

        mu_hat, var_hat = self.forward(raw)

        sti_hat = mu_hat[:, 0]
        alcons_hat = mu_hat[:, 1]
        t60_hat = mu_hat[:, 2]
        edt_hat = mu_hat[:, 3]
        c80_hat = mu_hat[:, 4]
        c50_hat = mu_hat[:, 5]
        d50_hat = mu_hat[:, 6]
        ts_hat = mu_hat[:, 7]
        volume_hat = mu_hat[:, 8]
        dist_src_hat = mu_hat[:, 9]

        loss_sti = self.joint_loss_test(sti, sti_hat)
        loss_alcons = self.joint_loss_test(alcons, alcons_hat)
        loss_t60 = self.joint_loss_test(t60, t60_hat)
        loss_edt = self.joint_loss_test(edt, edt_hat)
        loss_c80 = self.joint_loss_test(c80, c80_hat)
        loss_c50 = self.joint_loss_test(c50, c50_hat)
        loss_d50 = self.joint_loss_test(d50, d50_hat)
        loss_ts = self.joint_loss_test(ts, ts_hat)
        loss_volume = self.joint_loss_test(volume, volume_hat)
        loss_dist_src = self.joint_loss_test(dist_src, dist_src_hat)

        corrcoef_sti = self.joint_corrcoef_test(sti, sti_hat).abs()
        corrcoef_alcons = self.joint_corrcoef_test(alcons, alcons_hat).abs()
        corrcoef_t60 = self.joint_corrcoef_test(t60, t60_hat).abs()
        corrcoef_edt = self.joint_corrcoef_test(edt, edt_hat).abs()
        corrcoef_c80 = self.joint_corrcoef_test(c80, c80_hat).abs()
        corrcoef_c50 = self.joint_corrcoef_test(c50, c50_hat).abs()
        corrcoef_d50 = self.joint_corrcoef_test(d50, d50_hat).abs()
        corrcoef_ts = self.joint_corrcoef_test(ts, ts_hat).abs()
        corrcoef_t60 = self.joint_corrcoef_test(t60, t60_hat).abs()
        corrcoef_volume = self.joint_corrcoef_test(volume, volume_hat).abs()
        corrcoef_dist_src = self.joint_corrcoef_test(dist_src, dist_src_hat).abs()

        # update and log metrics
        self.test_loss_sti(loss_sti)
        self.test_loss_alcons(loss_alcons)
        self.test_loss_t60(loss_t60)
        self.test_loss_edt(loss_edt)
        self.test_loss_c80(loss_c80)
        self.test_loss_c50(loss_c50)
        self.test_loss_d50(loss_d50)
        self.test_loss_ts(loss_ts)
        self.test_loss_volume(loss_volume)
        self.test_loss_dist_src(loss_dist_src)

        self.test_corrcoef_sti(corrcoef_sti)
        self.test_corrcoef_alcons(corrcoef_alcons)
        self.test_corrcoef_t60(corrcoef_t60)
        self.test_corrcoef_edt(corrcoef_edt)
        self.test_corrcoef_c80(corrcoef_c80)
        self.test_corrcoef_c50(corrcoef_c50)
        self.test_corrcoef_d50(corrcoef_d50)
        self.test_corrcoef_ts(corrcoef_ts)
        self.test_corrcoef_volume(corrcoef_volume)
        self.test_corrcoef_dist_src(corrcoef_dist_src)

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
            "test/loss/t60",
            self.test_loss_t60,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "test/corrcoef/t60",
            self.test_corrcoef_t60,
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

    def predict_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> dict[str, torch.Tensor]:
        """Predict a single batch of data from the test set.

        :param batch: A batch of data (a dict) containing the input tensor of raw,
            target Th, target Tt, target volume, target distSrc, target azimuthSrc,
            target elevationSrc.

        :param batch_idx: The index of the current batch.

        :return: A dict containing the model predictions.
        """

        raw = batch["raw"]

        mu_hat, _ = self.forward(raw)

        sti_hat = mu_hat[:, 0]
        alcons_hat = mu_hat[:, 1]
        t60_hat = mu_hat[:, 2]
        edt_hat = mu_hat[:, 3]
        c80_hat = mu_hat[:, 4]
        c50_hat = mu_hat[:, 5]
        d50_hat = mu_hat[:, 6]
        ts_hat = mu_hat[:, 7]
        volume_hat = mu_hat[:, 8]
        dist_src_hat = mu_hat[:, 9]

        preds = {
            "sti_hat": sti_hat,
            "alcons_hat": alcons_hat,
            "t60_hat": t60_hat,
            "edt_hat": edt_hat,
            "c80_hat": c80_hat,
            "c50_hat": c50_hat,
            "d50_hat": d50_hat,
            "ts_hat": ts_hat,
            "volume_hat": volume_hat,
            "dist_src_hat": dist_src_hat,
        }

        return preds

    def configure_optimizers(self) -> dict[str, Any]:
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
    cfg = omegaconf.OmegaConf.load(
        root / "configs" / "model" / "network_baselineCRNN.yaml"
    )
    _ = hydra.utils.instantiate(cfg.model.jointRegressorModule)

    _ = JointRegressorModuleBaselineCNNMLP(cfg.model.jointRegressorModule)
