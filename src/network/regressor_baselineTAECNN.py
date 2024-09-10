import math
from typing import Any, Optional

import torch
from lightning import LightningModule
from torch.nn import L1Loss, MSELoss
from torchmetrics import MeanMetric, MinMetric, PearsonCorrCoef


# ======================== RIR regression module ========================
class RMSELoss(torch.nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.mse = MSELoss()
        self.eps = eps

    def forward(self, y_hat, y):
        return torch.sqrt(self.mse(y_hat, y) + self.eps)


class rirRegressorModuleBaselineTAECNN(LightningModule):
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

        # this line allows to access init params with 'self.hparams' aTRribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore="net")

        self.rirRegressor = net

        self.Th_weight = self.hparams.optim_cfg.Th_weight
        self.Tt_weight = self.hparams.optim_cfg.Tt_weight

        # loss function
        self.loss_fn_Th = RMSELoss()
        self.loss_fn_Tt = RMSELoss()
        self.loss_fn_Th_test = L1Loss()
        self.loss_fn_Tt_test = L1Loss()
        self.corrcoef_Th = PearsonCorrCoef()
        self.corrcoef_Tt = PearsonCorrCoef()

        # for averaging loss across batches
        self.train_loss_Th = MeanMetric()
        self.val_loss_Th = MeanMetric()
        self.test_loss_Th = MeanMetric()

        self.train_loss_Tt = MeanMetric()
        self.val_loss_Tt = MeanMetric()
        self.test_loss_Tt = MeanMetric()

        # for tracking best so far validation best metrics
        self.val_loss_best_Th = MinMetric()
        self.val_loss_best_Tt = MinMetric()

        # test of Pearson correlation coefficient
        self.test_corrcoef_Th = MeanMetric()
        self.test_corrcoef_Tt = MeanMetric()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Perform a forward pass.

        :param x: A tensor of waveform
        :return: A tensor of estimated Th, TR.
        """
        return self.rirRegressor(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss_Th.reset()
        self.val_loss_Tt.reset()
        self.val_loss_best_Th.reset()
        self.val_loss_best_Tt.reset()

    def model_step(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor
            of raw, target Th, target TR.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        raw = batch["raw"]
        Th = batch["Th"]
        Tt = batch["Tt"]

        ThTt_hat = self.forward(raw)
        Th_hat = ThTt_hat[:, 0]
        Tt_hat = ThTt_hat[:, 1]

        loss_Th = math.sqrt(self.Th_weight) * self.loss_fn_Th(Th_hat, Th)
        loss_Tt = math.sqrt(self.Tt_weight) * self.loss_fn_Tt(Tt_hat, Tt)

        return loss_Th, loss_Tt

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of raw,
            target Th, target TR.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss_Th, loss_Tt = self.model_step(batch)

        # update and log metrics
        self.train_loss_Th(loss_Th)
        self.train_loss_Tt(loss_Tt)

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

        total_loss = loss_Th + loss_Tt

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

        :param batch: A batch of data (a tuple) containing the input tensor of raw,
            target Th, target TR.
        :param batch_idx: The index of the current batch.
        """
        loss_Th, loss_Tt = self.model_step(batch)

        # update and log metrics
        self.val_loss_Th(loss_Th)
        self.val_loss_Tt(loss_Tt)

        self.log(
            "val/loss/Th",
            self.val_loss_Th,
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

        total_loss = loss_Th + loss_Tt

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
        metric_Th = self.val_loss_Th.compute()  # get current val metric
        metric_Tt = self.val_loss_Tt.compute()  # get current val metric

        self.val_loss_best_Th(metric_Th)  # update best so far val metric
        self.val_loss_best_Tt(metric_Tt)  # update best so far val metric

        total_loss_best = (
            self.Th_weight * self.val_loss_best_Th
            + self.Tt_weight * self.val_loss_best_Tt
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

    def test_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        raw = batch["raw"]
        Th = batch["Th"]
        Tt = batch["Tt"]

        ThTt_hat = self.forward(raw)
        Th_hat = ThTt_hat[:, 0]
        Tt_hat = ThTt_hat[:, 1]

        loss_Th = self.loss_fn_Th_test(Th_hat, Th)
        loss_Tt = self.loss_fn_Tt_test(Tt_hat, Tt)
        corrcoef_Th = self.corrcoef_Th(Th_hat, Th)
        corrcoef_Tt = self.corrcoef_Tt(Tt_hat, Tt)

        # update and log metrics
        self.test_loss_Th(loss_Th)
        self.test_loss_Tt(loss_Tt)
        self.test_corrcoef_Th(corrcoef_Th)
        self.test_corrcoef_Tt(corrcoef_Tt)

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

        total_loss = self.Th_weight * loss_Th + self.Tt_weight * loss_Tt

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
        """Perform a single prediction step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        raw = batch["raw"]

        ThTt_hat = self.forward(raw)
        Th_hat = ThTt_hat[:, 0]
        Tt_hat = ThTt_hat[:, 1]

        preds = {"Th_hat": Th_hat}
        preds["Tt_hat"] = Tt_hat

        return preds

    def configure_optimizers(self) -> dict[str, Any]:
        """Configures optimizers and learning-rate schedulers to be used for training.

        Normally you'd need one, but in the case of GANs or similar you might need multiple.

        Examples:
            hTRps://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

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
        root / "configs" / "model" / "network_baselineTAECNN.yaml"
    )
    _ = hydra.utils.instantiate(cfg.model.rirRegressorModule)

    _ = rirRegressorModuleBaselineTAECNN(cfg.model.rirRegressorModule)
