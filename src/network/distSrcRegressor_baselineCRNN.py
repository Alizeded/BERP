from typing import Any

import torch
from lightning import LightningModule
from torch.nn import L1Loss, MSELoss
from torchmetrics import MeanMetric, MinMetric, PearsonCorrCoef

# ======================== dist regression module ========================


class DistSrcRegressorModuleBaselineCRNN(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
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

        self.dist_srcRegressor = net

        # loss function
        self.loss_fn_dist_src = MSELoss()
        self.loss_test_dist_src = L1Loss()
        self.corrcoef_dist_src = PearsonCorrCoef()

        # for averaging loss across batches
        self.train_loss_dist_src = MeanMetric()
        self.val_loss_dist_src = MeanMetric()
        self.test_loss_dist_src = MeanMetric()

        # for tracking best so far validation best metrics
        self.val_loss_best_dist_src = MinMetric()

        # test of Pearson correlation coefficient
        self.test_corrcoef_dist_src = MeanMetric()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Perform a forward pass.

        :param x: A tensor of waveform
        :return: A tensor of estimated dist_src.
        """
        return self.dist_srcRegressor(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss_dist_src.reset()
        self.val_loss_best_dist_src.reset()

    def model_step(
        self,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a dict) containing the input tensor
            of raw, target clean, target dist_src.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        raw = batch["raw"]
        dist_src = batch["dist_src"]

        dist_src_hat = self.forward(raw)

        loss_dist_src = self.loss_fn_dist_src(dist_src_hat, dist_src)
        return loss_dist_src

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a dict) containing the input tensor of raw,
            target dist_src.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss_dist_src = self.model_step(batch)

        # update and log metrics
        self.train_loss_dist_src(loss_dist_src)

        self.log(
            "train/loss/dist_src",
            self.train_loss_dist_src,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        # return loss or backpropagation will fail
        return loss_dist_src

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
            target dist_src.
        :param batch_idx: The index of the current batch.
        """
        loss_dist_src = self.model_step(batch)

        # update and log metrics
        self.val_loss_dist_src(loss_dist_src)

        self.log(
            "val/loss/dist_src",
            self.val_loss_dist_src,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        metric_dist_src = self.val_loss_dist_src.compute()  # get current val metric

        self.val_loss_best_dist_src(metric_dist_src)  # update best so far val metric

        # log `val_loss_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch

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

        :param batch: A batch of data (a dict) containing the input tensor and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        raw = batch["raw"]
        dist_src = batch["dist_src"]

        dist_src_hat = self.forward(raw)

        loss_dist_src = self.loss_test_dist_src(dist_src_hat, dist_src)
        corrcoef_dist_src = self.corrcoef_dist_src(dist_src_hat, dist_src)

        # update and log metrics
        self.test_loss_dist_src(loss_dist_src)
        self.test_corrcoef_dist_src(corrcoef_dist_src)

        self.log(
            "test/loss/dist_src",
            self.test_loss_dist_src,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.log(
            "test/corrcoef/dist_src",
            self.test_corrcoef_dist_src,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def predict_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> dict[str, torch.Tensor]:
        """Perform a single prediction step on a batch of data from the test set.

        :param batch: A batch of data (a dict) containing the input tensor and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A dict containing the model predictions.
        """
        raw = batch["raw"]

        dist_src_hat = self.forward(raw)

        preds = {"dist_src_hat": dist_src_hat}

        return preds

    def configure_optimizers(self) -> dict[str, Any]:
        """Configures optimizers and learning-rate schedulers to be used for training.

        Normally you'd need one, but in the case of GANs or similar you might need multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(self.dist_srcRegressor.parameters())

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
                    "monitor": "val/loss/dist_src",
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
    _ = hydra.utils.instantiate(cfg.model.distSrcRegressorModule)

    _ = DistSrcRegressorModuleBaselineCRNN(cfg.model.distSrcRegressorModule)
