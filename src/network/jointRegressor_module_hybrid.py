from typing import Any, Optional

import torch
from lightning import LightningModule
from torchmetrics import (
    MaxMetric,
    MeanMetric,
    MinMetric,
)

from criterions.hybrid_joint_eval_metric import JointEstimationEvaluation
from criterions.hybrid_joint_loss import MultiTaskLoss
from src.utils.unitary_linear_norm import unitary_norm_inv

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
        self.criterion_train = MultiTaskLoss(phase="train")
        self.criterion_val = MultiTaskLoss(phase="val")
        self.criterion_test = JointEstimationEvaluation(
            iter_times=self.hparams.optim_cfg.iter_times,
            max_workers=self.hparams.optim_cfg.max_workers,
        )

        self.Th_weight = self.hparams.optim_cfg.Th_weight
        self.Tt_weight = self.hparams.optim_cfg.Tt_weight
        self.volume_weight = self.hparams.optim_cfg.volume_weight
        self.dist_src_weight = self.hparams.optim_cfg.dist_src_weight

        # for averaging loss across batches
        self.train_huber_Th = MeanMetric()
        self.val_l1_Th = MeanMetric()

        self.train_huber_Tt = MeanMetric()
        self.val_l1_Tt = MeanMetric()

        self.test_l1_sti = MeanMetric()
        self.test_l1_alcons = MeanMetric()
        self.test_l1_tr = MeanMetric()
        self.test_l1_edt = MeanMetric()
        self.test_l1_c80 = MeanMetric()
        self.test_l1_c50 = MeanMetric()
        self.test_l1_d50 = MeanMetric()
        self.test_l1_ts = MeanMetric()

        self.train_huber_volume = MeanMetric()
        self.val_l1_volume = MeanMetric()
        self.test_l1_volume = MeanMetric()

        self.train_huber_dist_src = MeanMetric()
        self.val_l1_dist_src = MeanMetric()
        self.test_l1_dist_src = MeanMetric()

        # for tracking validation correlation coefficient of Th, Tt, volume, distSrc
        self.val_corrcoef_Th = MeanMetric()
        self.val_corrcoef_Tt = MeanMetric()
        self.val_corrcoef_volume = MeanMetric()
        self.val_corrcoef_dist_src = MeanMetric()

        # for tracking evaluation correlation coefficient of Th, Tt, volume, distSrc
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
        self.val_l1_best_Th = MinMetric()
        self.val_l1_best_Tt = MinMetric()
        self.val_l1_best_volume = MinMetric()
        self.val_l1_best_dist_src = MinMetric()

        # for tracking best of correlation coefficient
        self.val_corrcoef_best_Th = MaxMetric()
        self.val_corrcoef_best_Tt = MaxMetric()
        self.val_corrcoef_best_volume = MaxMetric()
        self.val_corrcoef_best_dist_src = MaxMetric()

        self.predict_tools = JointEstimationEvaluation()

    def forward(
        self,
        source: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
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

        self.val_corrcoef_Th.reset()
        self.val_corrcoef_Tt.reset()
        self.val_corrcoef_volume.reset()
        self.val_corrcoef_dist_src.reset()

        self.val_l1_best_Th.reset()
        self.val_l1_best_Tt.reset()
        self.val_l1_best_volume.reset()
        self.val_l1_best_dist_src.reset()

        self.val_corrcoef_best_Th.reset()
        self.val_corrcoef_best_Tt.reset()
        self.val_corrcoef_best_volume.reset()
        self.val_corrcoef_best_dist_src.reset()

    def model_step(
        self,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
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

        return (
            loss_Th,
            loss_Tt,
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
        ) = self.model_step(batch)

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

        corr_Th = loss["corr_Th"]
        corr_Tt = loss["corr_Tt"]
        corr_volume = loss["corr_volume"]
        corr_dist_src = loss["corr_dist_src"]

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
        # ------------------- total best so far validation loss -------------------

        total_loss_best = (
            self.Th_weight * self.val_l1_best_Th
            + self.Tt_weight * self.val_l1_best_Tt
            + self.volume_weight * self.val_l1_best_volume
            + self.dist_src_weight * self.val_l1_best_dist_src
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

        # network forward pass
        net_output = self.forward(**batch["net_input"])

        # move to cpu for evaluation
        net_output = {k: v.cpu() for k, v in net_output.items()}

        loss = self.criterion_test(net_output, batch["groundtruth"])

        loss_sti = loss["loss_sti"]
        loss_alcons = loss["loss_alcons"]
        loss_tr = loss["loss_tr"]
        loss_edt = loss["loss_edt"]
        loss_c80 = loss["loss_c80"]
        loss_c50 = loss["loss_c50"]
        loss_d50 = loss["loss_d50"]
        loss_ts = loss["loss_ts"]
        loss_volume = loss["loss_volume"]
        loss_dist_src = loss["loss_dist_src"]
        corrcoef_sti = loss["corr_sti"]
        corrcoef_alcons = loss["corr_alcons"]
        corrcoef_tr = loss["corr_tr"]
        corrcoef_edt = loss["corr_edt"]
        corrcoef_c80 = loss["corr_c80"]
        corrcoef_c50 = loss["corr_c50"]
        corrcoef_d50 = loss["corr_d50"]
        corrcoef_ts = loss["corr_ts"]
        corrcoef_volume = loss["corr_volume"]
        corrcoef_dist_src = loss["corr_dist_src"]

        # update and log metrics
        self.test_l1_sti(loss_sti)
        self.test_l1_alcons(loss_alcons)
        self.test_l1_tr(loss_tr)
        self.test_l1_edt(loss_edt)
        self.test_l1_c80(loss_c80)
        self.test_l1_c50(loss_c50)
        self.test_l1_d50(loss_d50)
        self.test_l1_ts(loss_ts)
        self.test_l1_volume(loss_volume)
        self.test_l1_dist_src(loss_dist_src)

        self.test_corrcoef_sti(corrcoef_sti)
        self.test_corrcoef_alcons(corrcoef_alcons)
        self.test_corrcoef_tr(corrcoef_tr)
        self.test_corrcoef_edt(corrcoef_edt)
        self.test_corrcoef_c80(corrcoef_c80)
        self.test_corrcoef_c50(corrcoef_c50)
        self.test_corrcoef_d50(corrcoef_d50)
        self.test_corrcoef_ts(corrcoef_ts)
        self.test_corrcoef_volume(corrcoef_volume)
        self.test_corrcoef_dist_src(corrcoef_dist_src)

        self.log(
            "test/loss/sti",
            self.test_l1_sti,
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
            self.test_l1_alcons,
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
            self.test_l1_tr,
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
            self.test_l1_edt,
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
            self.test_l1_c80,
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
            self.test_l1_c50,
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
            self.test_l1_d50,
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
            self.test_l1_ts,
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

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        # Th_pred = self.model_step(batch)
        pass

    def predict_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        norm_span: dict[str, tuple[float, float]] = None,
    ) -> dict[str, torch.Tensor]:
        """Perform a single prediction step on a batch of data from the test set.

        :param batch: A batch of data (a dict) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.

        :return: A dict containing the model predictions.
        """
        net_output = self.forward(**batch["net_input"])

        if net_output["Th_hat"].dim() == 3:  # MLE output
            Th_hat = net_output["Th_hat"][:, 0]
            Tt_hat = net_output["Tt_hat"][:, 0]
            volume_hat = net_output["volume_hat"][:, 0]
            dist_src_hat = net_output["dist_src_hat"][:, 0]

        else:
            Th_hat = net_output["Th_hat"]
            Tt_hat = net_output["Tt_hat"]
            volume_hat = net_output["volume_hat"]
            dist_src_hat = net_output["dist_src_hat"]
        padding_mask = net_output["padding_mask"]

        assert Th_hat.shape == Tt_hat.shape == volume_hat.shape == dist_src_hat.shape

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

        else:
            Th_hat = Th_hat.mean(dim=1)
            Tt_hat = Tt_hat.mean(dim=1)
            volume_hat = volume_hat.mean(dim=1)
            dist_src_hat = dist_src_hat.mean(dim=1)

        preds = {"Th_hat": Th_hat}
        preds["Tt_hat"] = Tt_hat
        preds["volume_hat"] = volume_hat
        preds["dist_src_hat"] = dist_src_hat

        return preds

    def configure_optimizers(self) -> dict[str, Any]:
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
