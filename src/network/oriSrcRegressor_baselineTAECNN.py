from typing import Any

import torch
from lightning import LightningModule
from torch.nn import L1Loss, MSELoss
from torchmetrics import MeanMetric, MinMetric, PearsonCorrCoef


# ======================== OriSrc regression module ========================
class RMSELoss(torch.nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.mse = MSELoss()
        self.eps = eps

    def forward(self, y_hat, y):
        loss = torch.sqrt(self.mse(y_hat, y) + self.eps)
        return loss


class OriSrcRegressorModuleBaselineTAECNN(LightningModule):
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

        self.ori_srcRegressor = net

        # loss function
        self.loss_fn_ori_src = RMSELoss()

        self.loss_test_ori_src = L1Loss()
        self.corrcoef_test_ori_src = PearsonCorrCoef()

        # for averaging loss across batches

        self.train_loss_ori_azimuth = MeanMetric()
        self.val_loss_ori_azimuth = MinMetric()
        self.test_loss_ori_azimuth = MeanMetric()

        self.train_loss_ori_elevation = MeanMetric()
        self.val_loss_ori_elevation = MinMetric()
        self.test_loss_ori_elevation = MeanMetric()

        # for tracking best so far validation best metrics
        self.val_loss_best_ori_azimuth = MinMetric()
        self.val_loss_best_ori_elevation = MinMetric()

        # test of Pearson correlation coefficient
        self.test_corrcoef_ori_azimuth = MeanMetric()
        self.test_corrcoef_ori_elevation = MeanMetric()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Perform a forward pass.

        :param x: A tensor of waveform
        :return: A tensor of estimated ori_src.
        """

        return self.ori_srcRegressor(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss_ori_azimuth.reset()
        self.val_loss_ori_elevation.reset()
        self.val_loss_best_ori_azimuth.reset()
        self.val_loss_best_ori_elevation.reset()

    def model_step(
        self,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor
            of raw, target clean, target ori_src.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        raw = batch["raw"]
        ori_azimuth = batch["azimuth_src"]
        ori_elevation = batch["elevation_src"]

        ori_src_hat = self.forward(raw)

        ori_azimuth_hat = ori_src_hat[:, 0]
        ori_elevation_hat = ori_src_hat[:, 1]

        loss_ori_azimuth = self.loss_fn_ori_src(ori_azimuth_hat, ori_azimuth)
        loss_ori_elevation = self.loss_fn_ori_src(ori_elevation_hat, ori_elevation)

        return loss_ori_azimuth, loss_ori_elevation

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of raw,
            target ori_src.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss_ori_azimuth, loss_ori_elevation = self.model_step(batch)

        # update and log metrics
        self.train_loss_ori_azimuth(loss_ori_azimuth)
        self.train_loss_ori_elevation(loss_ori_elevation)

        self.log(
            "train/loss/ori_azimuth",
            self.train_loss_ori_azimuth,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "train/loss/ori_elevation",
            self.train_loss_ori_elevation,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        loss_ori_src = loss_ori_azimuth + loss_ori_elevation
        self.log(
            "train/loss/total",
            loss_ori_src,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        # return loss or backpropagation will fail
        return loss_ori_src

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
            target ori_src.
        :param batch_idx: The index of the current batch.
        """
        loss_ori_azimuth, loss_ori_elevation = self.model_step(batch)

        # update and log metrics
        self.val_loss_ori_azimuth(loss_ori_azimuth)
        self.val_loss_ori_elevation(loss_ori_elevation)

        self.log(
            "val/loss/ori_azimuth",
            self.val_loss_ori_azimuth,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "val/loss/ori_elevation",
            self.val_loss_ori_elevation,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        loss_ori_src = loss_ori_azimuth + loss_ori_elevation
        self.log(
            "val/loss/ori_src",
            loss_ori_src,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        metric_ori_azimuth = (
            self.val_loss_ori_azimuth.compute()
        )  # get current val metric
        metric_ori_elevation = (
            self.val_loss_ori_elevation.compute()
        )  # get current val metric

        self.val_loss_best_ori_azimuth(
            metric_ori_azimuth
        )  # update best so far val metric
        self.val_loss_best_ori_elevation(
            metric_ori_elevation
        )  # update best so far val metric

        metric_ori_src = (
            self.val_loss_best_ori_azimuth + self.val_loss_best_ori_elevation
        )  # combine metrics

        # log `val_loss_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch

        self.log(
            "val/loss_best/ori_azimuth",
            self.val_loss_best_ori_azimuth.compute(),
            sync_dist=True,
            prog_bar=True,
        )

        self.log(
            "val/loss_best/ori_elevation",
            self.val_loss_best_ori_elevation.compute(),
            sync_dist=True,
            prog_bar=True,
        )

        self.log(
            "val/loss_best/ori_src",
            metric_ori_src.compute(),
            sync_dist=True,
            prog_bar=True,
        )

    def test_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a dict) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        raw = batch["raw"]
        ori_azimuth = batch["azimuth_src"]
        ori_elevation = batch["elevation_src"]

        ori_src_hat = self.forward(raw)
        ori_azimuth_hat = ori_src_hat[:, 0]
        ori_elevation_hat = ori_src_hat[:, 1]

        loss_ori_azimuth = self.loss_test_ori_src(ori_azimuth_hat, ori_azimuth)
        loss_ori_elevation = self.loss_test_ori_src(ori_elevation_hat, ori_elevation)

        corrcoef_ori_azimuth = self.corrcoef_test_ori_src(ori_azimuth_hat, ori_azimuth)
        corrcoef_ori_elevation = self.corrcoef_test_ori_src(
            ori_elevation_hat, ori_elevation
        )

        # update and log metrics
        self.test_loss_ori_azimuth(loss_ori_azimuth)
        self.test_loss_ori_elevation(loss_ori_elevation)

        self.test_corrcoef_ori_azimuth(corrcoef_ori_azimuth)
        self.test_corrcoef_ori_elevation(corrcoef_ori_elevation)

        self.log(
            "test/loss/ori_azimuth",
            self.test_loss_ori_azimuth,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.log(
            "test/corrcoef/ori_azimuth",
            self.test_corrcoef_ori_azimuth,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.log(
            "test/loss/ori_elevation",
            self.test_loss_ori_elevation,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.log(
            "test/corrcoef/ori_elevation",
            self.test_corrcoef_ori_elevation,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        loss_ori_src = loss_ori_azimuth + loss_ori_elevation
        self.log(
            "test/loss/ori_src",
            loss_ori_src,
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
        """
        raw = batch["raw"]

        ori_src_hat = self.forward(raw)
        ori_azimuth_hat = ori_src_hat[:, 0]
        ori_elevation_hat = ori_src_hat[:, 1]

        preds = {"ori_azimuth_hat": ori_azimuth_hat}
        preds["ori_elevation_hat"] = ori_elevation_hat

        return preds

    def configure_optimizers(self) -> dict[str, Any]:
        """Configures optimizers and learning-rate schedulers to be used for training.

        Normally you'd need one, but in the case of GANs or similar you might need multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(self.ori_srcRegressor.parameters())

        if self.hparams.scheduler is not None:
            T_max = self.hparams.optim_cfg.T_max
            eta_min = 1e-7

            scheduler = self.hparams.scheduler(
                optimizer=optimizer,
                T_max=T_max,
                eta_min=eta_min,
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss/ori_src",
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
    _ = hydra.utils.instantiate(cfg.model.oriSrcRegressorModule)

    _ = OriSrcRegressorModuleBaselineTAECNN(cfg.model.oriSrcRegressorModule)
