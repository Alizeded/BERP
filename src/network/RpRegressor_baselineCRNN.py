from typing import Any, Optional

import torch
from lightning import LightningModule
from torch.nn import L1Loss, MSELoss
from torchmetrics import MeanMetric, MinMetric, PearsonCorrCoef

# ======================== joint regression module ========================


class JointRegressorModuleBaselineCRNN(LightningModule):
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
        self.joint_loss = MSELoss()
        self.joint_loss_val = L1Loss()
        self.joint_loss_test = L1Loss()
        self.joint_corrcoef_test = PearsonCorrCoef()

        # for averaging loss across batches
        self.train_loss_sti = MeanMetric()
        self.val_loss_sti = MeanMetric()
        self.test_loss_sti = MeanMetric()

        self.train_loss_t60 = MeanMetric()
        self.val_loss_t60 = MeanMetric()
        self.test_loss_t60 = MeanMetric()

        self.train_loss_c50 = MeanMetric()
        self.val_loss_c50 = MeanMetric()
        self.test_loss_c50 = MeanMetric()

        self.train_loss_c80 = MeanMetric()
        self.val_loss_c80 = MeanMetric()
        self.test_loss_c80 = MeanMetric()

        # for tracking evaluation correlation coefficient
        self.test_corrcoef_sti = MeanMetric()
        self.test_corrcoef_t60 = MeanMetric()
        self.test_corrcoef_c50 = MeanMetric()
        self.test_corrcoef_c80 = MeanMetric()

        # for tracking best so far validation best metrics
        self.val_loss_best_sti = MinMetric()
        self.val_loss_best_t60 = MinMetric()
        self.val_loss_best_c50 = MinMetric()
        self.val_loss_best_c80 = MinMetric()

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
        self.val_loss_t60.reset()
        self.val_loss_c50.reset()
        self.val_loss_c80.reset()
        self.val_loss_best_sti.reset()
        self.val_loss_best_t60.reset()
        self.val_loss_best_c50.reset()
        self.val_loss_best_c80.reset()

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
        t60 = batch["t60"]
        c50 = batch["c50"]
        c80 = batch["c80"]

        total_hat = self.forward(raw)

        sti_hat = total_hat[:, 0]
        t60_hat = total_hat[:, 1]
        c50_hat = total_hat[:, 2]
        c80_hat = total_hat[:, 3]

        loss_sti = self.joint_loss(sti_hat, sti)
        loss_t60 = self.joint_loss(t60_hat, t60)
        loss_c50 = self.joint_loss(c50_hat, c50)
        loss_c80 = self.joint_loss(c80_hat, c80)

        return (
            loss_sti,
            loss_t60,
            loss_c50,
            loss_c80,
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
            loss_t60,
            loss_c50,
            loss_c80,
        ) = self.model_step(batch)

        # update and log metrics
        self.train_loss_sti(loss_sti)
        self.train_loss_t60(loss_t60)
        self.train_loss_c50(loss_c50)
        self.train_loss_c80(loss_c80)

        self.log(
            "train/loss/sti",
            self.train_loss_sti,
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
            "train/loss/c50",
            self.train_loss_c50,
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

        total_loss = loss_sti + loss_t60 + loss_c50 + loss_c80

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
        t60 = batch["t60"]
        c50 = batch["c50"]
        c80 = batch["c80"]

        total_hat = self.forward(raw)

        sti_hat = total_hat[:, 0]
        t60_hat = total_hat[:, 1]
        c50_hat = total_hat[:, 2]
        c80_hat = total_hat[:, 3]

        # update and log metrics
        loss_sti = self.joint_loss_val(sti, sti_hat)
        loss_t60 = self.joint_loss_val(t60, t60_hat)
        loss_c50 = self.joint_loss_val(c50, c50_hat)
        loss_c80 = self.joint_loss_val(c80, c80_hat)

        self.val_loss_sti(loss_sti)
        self.val_loss_t60(loss_t60)
        self.val_loss_c50(loss_c50)
        self.val_loss_c80(loss_c80)

        self.log(
            "val/loss/sti",
            self.val_loss_sti,
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
            "val/loss/c50",
            self.val_loss_c50,
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

        total_loss = loss_sti + loss_t60 + loss_c50 + loss_c80

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
        metric_t60 = self.val_loss_t60.compute()  # get current val metric
        metric_c50 = self.val_loss_c50.compute()  # get current val metric
        metric_c80 = self.val_loss_c80.compute()  # get current val metric

        self.val_loss_best_sti(metric_sti)  # update best so far val metric
        self.val_loss_best_t60(metric_t60)  # update best so far val metric
        self.val_loss_best_c50(metric_c50)  # update best so far val metric
        self.val_loss_best_c80(metric_c80)  # update best so far val metric

        total_loss_best = (
            self.val_loss_best_sti
            + self.val_loss_best_t60
            + self.val_loss_best_c50
            + self.val_loss_best_c80
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
            "val/loss_best/t60",
            self.val_loss_best_t60.compute(),
            sync_dist=True,
            prog_bar=True,
        )

        self.log(
            "val/loss_best/C50",
            self.val_loss_best_c50.compute(),
            sync_dist=True,
            prog_bar=True,
        )

        self.log(
            "val/loss_best/C80",
            self.val_loss_best_c80.compute(),
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
        t60 = batch["t60"]
        c50 = batch["c50"]
        c80 = batch["c80"]

        total_hat = self.forward(raw)

        sti_hat = total_hat[:, 0]
        t60_hat = total_hat[:, 1]
        c50_hat = total_hat[:, 2]
        c80_hat = total_hat[:, 3]

        loss_sti = self.joint_loss_test(sti, sti_hat)
        loss_t60 = self.joint_loss_test(t60, t60_hat)
        loss_c50 = self.joint_loss_test(c50, c50_hat)
        loss_c80 = self.joint_loss_test(c80, c80_hat)

        corrcoef_sti = self.joint_corrcoef_test(sti, sti_hat).abs()
        corrcoef_t60 = self.joint_corrcoef_test(t60, t60_hat).abs()
        corrcoef_c50 = self.joint_corrcoef_test(c50, c50_hat).abs()
        corrcoef_c80 = self.joint_corrcoef_test(c80, c80_hat).abs()

        # update and log metrics
        self.test_loss_sti(loss_sti)
        self.test_loss_t60(loss_t60)
        self.test_loss_c50(loss_c50)
        self.test_loss_c80(loss_c80)

        self.test_corrcoef_sti(corrcoef_sti)
        self.test_corrcoef_t60(corrcoef_t60)
        self.test_corrcoef_c50(corrcoef_c50)
        self.test_corrcoef_c80(corrcoef_c80)

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

        total_loss = loss_sti + loss_t60 + loss_c50 + loss_c80

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
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> dict[str, torch.Tensor]:
        """Predict a single batch of data from the test set.

        :param batch: A batch of data (a dict) containing the input tensor of raw,
            target Th, target Tt, target volume, target distSrc

        :param batch_idx: The index of the current batch.

        :return: A dict containing the model predictions.
        """

        raw = batch["raw"]

        total_hat = self.forward(raw)

        sti_hat = total_hat[:, 0]
        t60_hat = total_hat[:, 1]
        c50_hat = total_hat[:, 2]
        c80_hat = total_hat[:, 3]

        preds = {
            "sti_hat": sti_hat,
            "t60_hat": t60_hat,
            "c50_hat": c50_hat,
            "c80_hat": c80_hat,
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

    _ = JointRegressorModuleBaselineCRNN(cfg.model.jointRegressorModule)
