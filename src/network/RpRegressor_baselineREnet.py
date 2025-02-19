from typing import Any, Optional

import torch
from einops import repeat
from lightning import LightningModule
from torch.nn import L1Loss, MSELoss
from torchmetrics import MeanMetric, MinMetric, PearsonCorrCoef

# ======================== joint regression module ========================


class JointRegressorModuleBaselineREnet(LightningModule):
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
        self.loss = MSELoss()
        self.loss_test = L1Loss()
        self.corrcoef_test = PearsonCorrCoef()

        # for averaging loss across batches
        self.train_loss_t60 = MeanMetric()
        self.val_loss_t60 = MeanMetric()
        self.test_loss_t60 = MeanMetric()

        # for tracking evaluation correlation coefficient
        self.test_corrcoef_t60 = MeanMetric()

        # for tracking best so far validation best metrics
        self.val_loss_best_t60 = MinMetric()

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
        self.val_loss_t60.reset()
        self.val_loss_best_t60.reset()

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
        t60 = batch["t60"]

        t60_hat = self.forward(raw)

        t60 = repeat(t60, "b -> b t", t=t60_hat.shape[1])

        loss_Tt = self.loss(t60_hat, t60)

        return loss_Tt

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
        loss_t60 = self.model_step(batch)

        # update and log metrics
        self.train_loss_t60(loss_t60)

        self.log(
            "train/loss/total",
            self.train_loss_t60,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        # return loss or backpropagation will fail
        return loss_t60

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
        loss_t60 = self.model_step(batch)

        # update and log metrics
        self.val_loss_t60(loss_t60)

        self.log(
            "val/loss/total",
            self.val_loss_t60,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        metric_t60 = self.val_loss_t60.compute()  # get current val metric

        self.val_loss_best_t60(metric_t60)  # update best so far val metric

        # log `val_loss_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch

        self.log(
            "val/loss_best/t60",
            self.val_loss_best_t60.compute(),
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
        t60 = batch["t60"]

        t60_hat = self.forward(raw)

        t60_hat = t60_hat[:, -1]  # take the last time step

        loss_t60 = self.loss_test(t60_hat, t60)

        corrcoef_t60 = self.corrcoef_test(t60_hat, t60)

        # update and log metrics
        self.test_loss_t60(loss_t60)

        self.test_corrcoef_t60(corrcoef_t60).abs()

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

        t60_hat = self.forward(raw)

        t60_hat = t60_hat[:, -1]  # take the last time step

        preds = {
            "t60_hat": t60_hat,
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
            min_lr = 1e-8

            scheduler = self.hparams.scheduler(
                optimizer=optimizer,
                min_lr=min_lr,
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss/total",
                    "interval": "epoch",
                    "frequency": 3,
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

    _ = JointRegressorModuleBaselineREnet(cfg.model.jointRegressorModule)
