from typing import Any, Optional

import torch
from lightning import LightningModule
from torchmetrics import (
    MaxMetric,
    MeanMetric,
    MinMetric,
)

from criterions.e2e_joint_eval_metric_4params import RPJointEstimationEvaluation
from criterions.e2e_joint_loss_4params import RPMultiTaskLoss

# ======================== joint regression module ========================


class JointRegressorModuleE2E(LightningModule):
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
        self.criterion_train = RPMultiTaskLoss(phase="train")
        self.criterion_val = RPMultiTaskLoss(phase="val")
        self.criterion_test = RPJointEstimationEvaluation()

        self.sti_weight = self.hparams.optim_cfg.sti_weight
        self.tr_weight = self.hparams.optim_cfg.tr_weight
        self.c80_weight = self.hparams.optim_cfg.c80_weight
        self.c50_weight = self.hparams.optim_cfg.c50_weight

        # for averaging loss across batches
        self.train_huber_sti = MeanMetric()
        self.val_l1_sti = MeanMetric()
        self.test_l1_sti = MeanMetric()

        self.train_huber_tr = MeanMetric()
        self.val_l1_tr = MeanMetric()
        self.test_l1_tr = MeanMetric()

        self.train_huber_c80 = MeanMetric()
        self.val_l1_c80 = MeanMetric()
        self.test_l1_c80 = MeanMetric()

        self.train_huber_c50 = MeanMetric()
        self.val_l1_c50 = MeanMetric()
        self.test_l1_c50 = MeanMetric()

        # for tracking validation correlation coefficient of Th, Tt, volume, distSrc
        self.val_corrcoef_sti = MeanMetric()
        self.val_corrcoef_tr = MeanMetric()
        self.val_corrcoef_c80 = MeanMetric()
        self.val_corrcoef_c50 = MeanMetric()

        # for tracking evaluation correlation coefficient of Th, Tt, volume, distSrc
        self.test_corrcoef_sti = MeanMetric()
        self.test_corrcoef_tr = MeanMetric()
        self.test_corrcoef_c80 = MeanMetric()
        self.test_corrcoef_c50 = MeanMetric()

        # for tracking best so far validation best metrics
        self.val_l1_best_sti = MinMetric()
        self.val_l1_best_tr = MinMetric()
        self.val_l1_best_c80 = MinMetric()
        self.val_l1_best_c50 = MinMetric()

        # for tracking best of correlation coefficient
        self.val_corrcoef_best_sti = MaxMetric()
        self.val_corrcoef_best_tr = MaxMetric()
        self.val_corrcoef_best_c80 = MaxMetric()
        self.val_corrcoef_best_c50 = MaxMetric()

        self.predict_tools = RPJointEstimationEvaluation()

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
        self.val_l1_sti.reset()
        self.val_l1_tr.reset()
        self.val_l1_c80.reset()
        self.val_l1_c50.reset()

        self.val_corrcoef_sti.reset()
        self.val_corrcoef_tr.reset()
        self.val_corrcoef_c80.reset()
        self.val_corrcoef_c50.reset()

        self.val_l1_best_sti.reset()
        self.val_l1_best_tr.reset()
        self.val_l1_best_c80.reset()
        self.val_l1_best_c50.reset()

        self.val_corrcoef_best_sti.reset()
        self.val_corrcoef_best_tr.reset()
        self.val_corrcoef_best_c80.reset()
        self.val_corrcoef_best_c50.reset()

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

        loss_sti = loss["loss_sti"]
        loss_tr = loss["loss_tr"]
        loss_c80 = loss["loss_c80"]
        loss_c50 = loss["loss_c50"]

        return (
            loss_sti,
            loss_tr,
            loss_c80,
            loss_c50,
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
            loss_sti,
            loss_tr,
            loss_c80,
            loss_c50,
        ) = self.model_step(batch)

        # ------------------- other parameters -------------------

        # update huber loss of Th, Tt, volume, distSrc
        self.train_huber_sti(loss_sti)
        self.train_huber_tr(loss_tr)
        self.train_huber_c80(loss_c80)
        self.train_huber_c50(loss_c50)

        # ------------------- total loss -------------------
        total_loss = (
            self.sti_weight * loss_sti
            + self.tr_weight * loss_tr
            + self.c80_weight * loss_c80
            + self.c50_weight * loss_c50
        )

        # ------------------- log metrics -------------------
        self.log(
            "train/loss/sti",
            self.train_huber_sti,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "train/loss/tr",
            self.train_huber_tr,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "train/loss/c80",
            self.train_huber_c80,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "train/loss/c50",
            self.train_huber_c50,
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

        loss_sti = loss["loss_sti"]
        loss_tr = loss["loss_tr"]
        loss_c80 = loss["loss_c80"]
        loss_c50 = loss["loss_c50"]

        corr_sti = loss["corr_sti"]
        corr_tr = loss["corr_tr"]
        corr_c80 = loss["corr_c80"]
        corr_c50 = loss["corr_c50"]

        # ------------------- other parameters -------------------

        self.val_l1_sti(loss_sti)
        self.val_l1_tr(loss_tr)
        self.val_l1_c80(loss_c80)
        self.val_l1_c50(loss_c50)

        self.val_corrcoef_sti(corr_sti)
        self.val_corrcoef_tr(corr_tr)
        self.val_corrcoef_c80(corr_c80)
        self.val_corrcoef_c50(corr_c50)

        # ------------------- total loss -------------------

        total_loss = (
            self.sti_weight * loss_sti
            + self.tr_weight * loss_tr
            + self.c80_weight * loss_c80
            + self.c50_weight * loss_c50
        )

        # ------------------- log metrics -------------------
        self.log(
            "val/loss/sti",
            self.val_l1_sti,
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
            "val/loss/tr",
            self.val_l1_tr,
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
            "val/loss/c80",
            self.val_l1_c80,
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
            self.val_l1_c50,
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
        metric_l1_sti = self.val_l1_sti.compute()  # get current val metric
        metric_l1_tr = self.val_l1_tr.compute()  # get current val metric
        metric_l1_c80 = self.val_l1_c80.compute()  # get current val metric
        metric_l1_c50 = self.val_l1_c50.compute()  # get current val metric

        # ------------- best so far validation correlation coefficient ------------
        metric_sti_corrcoef = self.val_corrcoef_sti.compute()
        metric_tr_corrcoef = self.val_corrcoef_tr.compute()
        metric_c80_corrcoef = self.val_corrcoef_c80.compute()
        metric_c50_corrcoef = self.val_corrcoef_c50.compute()

        # ------- update best so far val metric of Th, Tt, volume, distSrc -------
        # update best so far val metric of loss
        self.val_l1_best_sti(metric_l1_sti)  # update best so far val metric
        self.val_l1_best_tr(metric_l1_tr)
        self.val_l1_best_c80(metric_l1_c80)
        self.val_l1_best_c50(metric_l1_c50)

        # update best so far val metric of correlation coefficient
        self.val_corrcoef_best_sti(metric_sti_corrcoef)
        self.val_corrcoef_best_tr(metric_tr_corrcoef)
        self.val_corrcoef_best_c80(metric_c80_corrcoef)
        self.val_corrcoef_best_c50(metric_c50_corrcoef)
        # ------------------- total best so far validation loss -------------------

        total_loss_best = (
            self.sti_weight * self.val_l1_best_sti
            + self.tr_weight * self.val_l1_best_tr
            + self.c80_weight * self.val_l1_best_c80
            + self.c50_weight * self.val_l1_best_c50
        )

        # log `val_loss_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch

        # ------------------- log best so far validation loss -------------------

        self.log(
            "val/loss_best/sti",
            self.val_l1_best_sti.compute(),
            sync_dist=True,
            prog_bar=True,
        )

        self.log(
            "val/loss_best/tr",
            self.val_l1_best_tr.compute(),
            sync_dist=True,
            prog_bar=True,
        )

        self.log(
            "val/loss_best/c80",
            self.val_l1_best_c80.compute(),
            sync_dist=True,
            prog_bar=True,
        )

        self.log(
            "val/loss_best/c50",
            self.val_l1_best_c50.compute(),
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
            "val/corrcoef_best/tr",
            self.val_corrcoef_best_tr.compute(),
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

        loss = self.criterion_test(net_output, batch["groundtruth"])

        loss_sti = loss["loss_sti"]
        loss_tr = loss["loss_tr"]
        loss_c80 = loss["loss_c80"]
        loss_c50 = loss["loss_c50"]
        corrcoef_sti = loss["corr_sti"]
        corrcoef_tr = loss["corr_tr"]
        corrcoef_c80 = loss["corr_c80"]
        corrcoef_c50 = loss["corr_c50"]

        # update and log metrics
        self.test_l1_sti(loss_sti)
        self.test_l1_tr(loss_tr)
        self.test_l1_c80(loss_c80)
        self.test_l1_c50(loss_c50)

        self.test_corrcoef_sti(corrcoef_sti)
        self.test_corrcoef_tr(corrcoef_tr)
        self.test_corrcoef_c80(corrcoef_c80)
        self.test_corrcoef_c50(corrcoef_c50)

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

        sti_hat = net_output["sti_hat"]
        tr_hat = net_output["tr_hat"]
        c80_hat = net_output["c80_hat"]
        c50_hat = net_output["c50_hat"]
        padding_mask = net_output["padding_mask"]

        assert sti_hat.shape == tr_hat.shape == c80_hat.shape == c50_hat.shape

        if padding_mask is not None and padding_mask.any():
            # B x T
            input_lengths = (1 - padding_mask.long()).sum(-1)
            # apply conv formula to get real output_lengths
            output_lengths = self.predict_tools._get_param_pred_output_lengths(
                input_lengths=input_lengths
            )

            padding_mask = torch.zeros(
                tr_hat.shape[:2], dtype=tr_hat.dtype, device=tr_hat.device
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
        if sti_hat.dim() == 2:
            if padding_mask is not None and padding_mask.any():
                # ---------------------- padding mask handling ----------------------
                sti_hat = (sti_hat * reverse_padding_mask).sum(
                    dim=1
                ) / reverse_padding_mask.sum(dim=1)

                tr_hat = (tr_hat * reverse_padding_mask).sum(
                    dim=1
                ) / reverse_padding_mask.sum(dim=1)

                c80_hat = (c80_hat * reverse_padding_mask).sum(
                    dim=1
                ) / reverse_padding_mask.sum(dim=1)

                c50_hat = (c50_hat * reverse_padding_mask).sum(
                    dim=1
                ) / reverse_padding_mask.sum(dim=1)

            else:
                sti_hat = sti_hat.mean(dim=1)
                tr_hat = tr_hat.mean(dim=1)
                c80_hat = c80_hat.mean(dim=1)
                c50_hat = c50_hat.mean(dim=1)

        elif sti_hat.dim() == 1:
            sti_hat = sti_hat.mean()
            tr_hat = tr_hat.mean()
            c80_hat = c80_hat.mean()
            c50_hat = c50_hat.mean()

        preds = {"sti_hat": sti_hat}
        preds["tr_hat"] = tr_hat
        preds["c80_hat"] = c80_hat
        preds["c50_hat"] = c50_hat

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
    _ = hydra.utils.instantiate(cfg.model.jointRegressorModuleRAP)

    _ = JointRegressorModuleE2E(cfg.model.jointRegressorModuleRAP)
