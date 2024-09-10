import math
from typing import Any, Optional

import torch
from lightning import LightningModule
from torch.nn import L1Loss, MSELoss
from torchmetrics import MeanMetric, MinMetric, PearsonCorrCoef


# ======================== joint regression module ========================
class RMSELoss(torch.nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.mse = MSELoss()
        self.eps = eps

    def forward(self, y_hat, y):
        loss = torch.sqrt(self.mse(y_hat, y) + self.eps)
        return loss


class JointRegressorModuleBaselineTAECNN(LightningModule):
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
        self.joint_loss = RMSELoss()
        self.joint_loss_test = L1Loss()
        self.joint_corrcoef_test = PearsonCorrCoef()

        self.Th_weight = self.hparams.optim_cfg.Th_weight
        self.Tt_weight = self.hparams.optim_cfg.Tt_weight
        self.volume_weight = self.hparams.optim_cfg.volume_weight
        self.distSrc_weight = self.hparams.optim_cfg.distSrc_weight
        self.azimuthSrc_weight = self.hparams.optim_cfg.azimuthSrc_weight
        self.elevationSrc_weight = self.hparams.optim_cfg.elevationSrc_weight

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

        self.train_loss_distSrc = MeanMetric()
        self.val_loss_distSrc = MeanMetric()
        self.test_loss_distSrc = MeanMetric()

        self.train_loss_azimuthSrc = MeanMetric()
        self.val_loss_azimuthSrc = MeanMetric()
        self.test_loss_azimuthSrc = MeanMetric()

        self.train_loss_elevationSrc = MeanMetric()
        self.val_loss_elevationSrc = MeanMetric()
        self.test_loss_elevationSrc = MeanMetric()

        # for tracking evaluation correlation coefficient
        self.test_corrcoef_Th = MeanMetric()
        self.test_corrcoef_Tt = MeanMetric()
        self.test_corrcoef_volume = MeanMetric()
        self.test_corrcoef_distSrc = MeanMetric()
        self.test_corrcoef_azimuthSrc = MeanMetric()
        self.test_corrcoef_elevationSrc = MeanMetric()

        # for tracking best so far validation best metrics
        self.val_loss_best_Th = MinMetric()
        self.val_loss_best_Tt = MinMetric()
        self.val_loss_best_volume = MinMetric()
        self.val_loss_best_distSrc = MinMetric()
        self.val_loss_best_azimuthSrc = MinMetric()
        self.val_loss_best_elevationSrc = MinMetric()

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
        self.val_loss_Th.reset()
        self.val_loss_Tt.reset()
        self.val_loss_volume.reset()
        self.val_loss_distSrc.reset()
        self.val_loss_azimuthSrc.reset()
        self.val_loss_elevationSrc.reset()
        self.val_loss_best_Th.reset()
        self.val_loss_best_Tt.reset()
        self.val_loss_best_volume.reset()
        self.val_loss_best_distSrc.reset()
        self.val_loss_best_azimuthSrc.reset()
        self.val_loss_best_elevationSrc.reset()

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
        Th = batch["Th"]
        Tt = batch["Tt"]
        volume = batch["volume"]
        dist_src = batch["dist_src"]
        azimuth_src = batch["azimuth_src"]
        elevation_src = batch["elevation_src"]

        total_hat = self.forward(raw)

        Th_hat = total_hat[:, 0]
        Tt_hat = total_hat[:, 1]
        volume_hat = total_hat[:, 2]
        dist_src_hat = total_hat[:, 3]
        azimuth_src_hat = total_hat[:, 4]
        elevation_src_hat = total_hat[:, 5]

        loss_Th = math.sqrt(self.Th_weight) * self.joint_loss(Th_hat, Th)
        loss_Tt = math.sqrt(self.Tt_weight) * self.joint_loss(Tt_hat, Tt)
        loss_volume = math.sqrt(self.volume_weight) * self.joint_loss(
            volume_hat, volume
        )
        loss_dist_src = math.sqrt(self.distSrc_weight) * self.joint_loss(
            dist_src_hat, dist_src
        )
        loss_azimuth_src = math.sqrt(self.azimuthSrc_weight) * self.joint_loss(
            azimuth_src_hat, azimuth_src
        )
        loss_elevation_src = math.sqrt(self.elevationSrc_weight) * self.joint_loss(
            elevation_src_hat, elevation_src
        )

        return (
            loss_Th,
            loss_Tt,
            loss_volume,
            loss_dist_src,
            loss_azimuth_src,
            loss_elevation_src,
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
            loss_Th,
            loss_Tt,
            loss_volume,
            loss_dist_src,
            loss_azimuth_src,
            loss_elevation_src,
        ) = self.model_step(batch)

        # update and log metrics
        self.train_loss_Th(loss_Th)
        self.train_loss_Tt(loss_Tt)
        self.train_loss_volume(loss_volume)
        self.train_loss_distSrc(loss_dist_src)
        self.train_loss_azimuthSrc(loss_azimuth_src)
        self.train_loss_elevationSrc(loss_elevation_src)

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
            self.train_loss_distSrc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "train/loss/azimuth_src",
            self.train_loss_azimuthSrc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "train/loss/elevation_src",
            self.train_loss_elevationSrc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        total_loss = (
            loss_Th
            + loss_Tt
            + loss_volume
            + loss_dist_src
            + loss_azimuth_src
            + loss_elevation_src
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
        (
            loss_Th,
            loss_Tt,
            loss_volume,
            loss_dist_src,
            loss_azimuth_src,
            loss_elevation_src,
        ) = self.model_step(batch)

        # update and log metrics
        self.val_loss_Th(loss_Th)
        self.val_loss_Tt(loss_Tt)
        self.val_loss_volume(loss_volume)
        self.val_loss_distSrc(loss_dist_src)
        self.val_loss_azimuthSrc(loss_azimuth_src)
        self.val_loss_elevationSrc(loss_elevation_src)

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
            self.val_loss_distSrc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "val/loss/azimuth_src",
            self.val_loss_azimuthSrc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "val/loss/elevation_src",
            self.val_loss_elevationSrc,
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
            + loss_azimuth_src
            + loss_elevation_src
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
        metric_Th = self.val_loss_Th.compute()  # get current val metric
        metric_Tt = self.val_loss_Tt.compute()  # get current val metric
        metric_volume = self.val_loss_volume.compute()  # get current val metric
        metric_distSrc = self.val_loss_distSrc.compute()  # get current val metric
        metric_azimuthSrc = self.val_loss_azimuthSrc.compute()  # get current val metric
        metric_elevationSrc = (
            self.val_loss_elevationSrc.compute()
        )  # get current val metric

        self.val_loss_best_Th(metric_Th)  # update best so far val metric
        self.val_loss_best_Tt(metric_Tt)  # update best so far val metric
        self.val_loss_best_volume(metric_volume)  # update best so far val metric
        self.val_loss_best_distSrc(metric_distSrc)  # update best so far val metric
        self.val_loss_best_azimuthSrc(
            metric_azimuthSrc
        )  # update best so far val metric
        self.val_loss_best_elevationSrc(metric_elevationSrc)

        total_loss_best = (
            self.Th_weight * self.val_loss_best_Th
            + self.Tt_weight * self.val_loss_best_Tt
            + self.volume_weight * self.val_loss_best_volume
            + self.distSrc_weight * self.val_loss_best_distSrc
            + self.azimuthSrc_weight * self.val_loss_best_azimuthSrc
            + self.elevationSrc_weight * self.val_loss_best_elevationSrc
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

        self.log(
            "val/loss_best/volume",
            self.val_loss_best_volume.compute(),
            sync_dist=True,
            prog_bar=True,
        )

        self.log(
            "val/loss_best/dist_src",
            self.val_loss_best_distSrc.compute(),
            sync_dist=True,
            prog_bar=True,
        )

        self.log(
            "val/loss_best/azimuth_src",
            self.val_loss_best_azimuthSrc.compute(),
            sync_dist=True,
            prog_bar=True,
        )

        self.log(
            "val/loss_best/elevation_src",
            self.val_loss_best_elevationSrc.compute(),
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
        Th = batch["Th"]
        Tt = batch["Tt"]
        volume = batch["volume"]
        dist_src = batch["dist_src"]
        azimuth_src = batch["azimuth_src"]
        elevation_src = batch["elevation_src"]

        total_hat = self.forward(raw)

        Th_hat = total_hat[:, 0]
        Tt_hat = total_hat[:, 1]
        volume_hat = total_hat[:, 2]
        dist_src_hat = total_hat[:, 3]
        azimuth_src_hat = total_hat[:, 4]
        elevation_src_hat = total_hat[:, 5]

        loss_Th = self.joint_loss_test(Th_hat, Th)
        loss_Tt = self.joint_loss_test(Tt_hat, Tt)
        loss_volume = self.joint_loss_test(volume_hat, volume)
        loss_dist_src = self.joint_loss_test(dist_src_hat, dist_src)
        loss_azimuth_src = self.joint_loss_test(azimuth_src_hat, azimuth_src)
        loss_elevation_src = self.joint_loss_test(elevation_src_hat, elevation_src)

        corrcoef_Th = self.joint_corrcoef_test(Th_hat, Th)
        corrcoef_Tt = self.joint_corrcoef_test(Tt_hat, Tt)
        corrcoef_volume = self.joint_corrcoef_test(volume_hat, volume)
        corrcoef_dist_src = self.joint_corrcoef_test(dist_src_hat, dist_src)
        corrcoef_azimuth_src = self.joint_corrcoef_test(azimuth_src_hat, azimuth_src)
        corrcoef_elevation_src = self.joint_corrcoef_test(
            elevation_src_hat, elevation_src
        )

        # update and log metrics
        self.test_loss_Th(loss_Th)
        self.test_loss_Tt(loss_Tt)
        self.test_loss_volume(loss_volume)
        self.test_loss_distSrc(loss_dist_src)
        self.test_loss_azimuthSrc(loss_azimuth_src)
        self.test_loss_elevationSrc(loss_elevation_src)

        self.test_corrcoef_Th(corrcoef_Th)
        self.test_corrcoef_Tt(corrcoef_Tt)
        self.test_corrcoef_volume(corrcoef_volume)
        self.test_corrcoef_distSrc(corrcoef_dist_src)
        self.test_corrcoef_azimuthSrc(corrcoef_azimuth_src)
        self.test_corrcoef_elevationSrc(corrcoef_elevation_src)

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
            self.test_loss_distSrc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "test/corrcoef/dist_src",
            self.test_corrcoef_distSrc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "test/loss/azimuth_src",
            self.test_loss_azimuthSrc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "test/corrcoef/azimuth_src",
            self.test_corrcoef_azimuthSrc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "test/loss/elevation_src",
            self.test_loss_elevationSrc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "test/corrcoef/elevation_src",
            self.test_corrcoef_elevationSrc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        total_loss = (
            self.Th_weight * loss_Th
            + self.Tt_weight * loss_Tt
            + self.volume_weight * loss_volume
            + self.distSrc_weight * loss_dist_src
            + self.azimuthSrc_weight * loss_azimuth_src
            + self.elevationSrc_weight * loss_elevation_src
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

        total_hat = self.forward(raw)

        Th_hat = total_hat[:, 0]
        Tt_hat = total_hat[:, 1]
        volume_hat = total_hat[:, 2]
        dist_src_hat = total_hat[:, 3]
        azimuth_src_hat = total_hat[:, 4]
        elevation_src_hat = total_hat[:, 5]

        preds = {
            "Th_hat": Th_hat,
            "Tt_hat": Tt_hat,
            "volume_hat": volume_hat,
            "dist_src_hat": dist_src_hat,
            "azimuth_src_hat": azimuth_src_hat,
            "elevation_src_hat": elevation_src_hat,
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
        root / "configs" / "model" / "network_baselineTAECNN.yaml"
    )
    _ = hydra.utils.instantiate(cfg.model.jointRegressorModule)

    _ = JointRegressorModuleBaselineTAECNN(cfg.model.jointRegressorModule)
