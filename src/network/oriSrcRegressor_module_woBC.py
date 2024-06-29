from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningModule
from torchmetrics import (
    MaxMetric,
    MinMetric,
    MeanMetric,
)

from src.criterions.huber_loss import HuberLoss
from src.utils.unitary_linear_norm import unitary_norm_inv

# ======================== OriSrc regression module ========================


class OriSrcRegressorModuleWOBC(LightningModule):
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

        self.ori_srcRegressor = net

        # loss function
        self.criterion_train = HuberLoss(phase="train", module="orientation")
        self.criterion_val = HuberLoss(phase="val", module="orientation")
        self.criterion_test = HuberLoss(phase="test", module="orientation")

        # for averaging loss across batches
        self.train_huber_azimuth = MeanMetric()
        self.train_huber_elevation = MeanMetric()

        self.val_l1_azimuth = MeanMetric()
        self.val_l1_elevation = MeanMetric()

        self.val_corrcoef_azimuth = MeanMetric()
        self.val_corrcoef_elevation = MeanMetric()

        self.test_l1_azimuth = MeanMetric()
        self.test_l1_elevation = MeanMetric()

        self.test_corrcoef_azimuth = MeanMetric()
        self.test_corrcoef_elevation = MeanMetric()

        # for tracking best so far validation best metrics
        self.val_l1_best_azimuth = MinMetric()
        self.val_l1_best_elevation = MinMetric()

        self.val_corrcoef_best_azimuth = MaxMetric()
        self.val_corrcoef_best_elevation = MaxMetric()

    def forward(
        self,
        source: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Perform a forward pass.

        :param x: A tensor of waveform
        :return: A tensor of estimated azimuth and elevation.
        """
        return self.ori_srcRegressor(source, padding_mask)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_l1_elevation.reset()
        self.val_l1_azimuth.reset()
        self.val_l1_best_azimuth.reset()
        self.val_l1_best_elevation.reset()
        self.val_corrcoef_azimuth.reset()
        self.val_corrcoef_elevation.reset()

    def model_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor
            of raw, target clean, target ori_src.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        # network forward pass
        net_output = self.forward(**batch["net_input"])

        # compute loss
        loss = self.criterion_train(net_output, batch["groundtruth"])

        loss_azimuth = loss["loss_azimuth"]
        loss_elevation = loss["loss_elevation"]

        return loss_azimuth, loss_elevation

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Perfor
        m a single training step on a batch of data from the training set.

        :param batch: A batch of data (a dict) containing the input tensor of raw,
             target ori_src.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss_azimuth, loss_elevation = self.model_step(batch)

        # update and log metrics
        self.train_huber_azimuth(loss_azimuth)
        self.train_huber_elevation(loss_elevation)

        loss = loss_azimuth + loss_elevation

        # log metrics
        self.log(
            "train/loss/ori_elevation",
            self.train_huber_elevation,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "train/loss/ori_azimuth",
            self.train_huber_azimuth,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "train/loss/total",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        # return loss or backpropagation will fail
        return loss

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
            target ori_src.
        :param batch_idx: The index of the current batch.
        """
        # total_loss, pred_labels = self.model_step(batch)

        # BC_loss_azimuth = total_loss["BC_loss_azimuth"][0]
        # BC_loss_elevation = total_loss["BC_loss_elevation"][0]

        net_output = self.forward(**batch["net_input"])

        loss = self.criterion_val(net_output, batch["groundtruth"])

        loss_azimuth = loss["loss_azimuth"]
        loss_elevation = loss["loss_elevation"]
        corr_coef_azimuth = loss["corr_coef_azimuth"]
        corr_coef_elevation = loss["corr_coef_elevation"]

        # update and log metrics
        self.val_l1_azimuth(loss_azimuth)
        self.val_l1_elevation(loss_elevation)

        self.val_corrcoef_azimuth(corr_coef_azimuth)
        self.val_corrcoef_elevation(corr_coef_elevation)

        total_loss = loss_azimuth + loss_elevation

        # log metrics

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
            "val/loss/ori_elevation",
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
            "val/loss/ori_src",
            total_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."

        metric_l1_azimuth = (
            self.val_l1_azimuth.compute()
        )  # get current val metric of loss for azimuth
        metric_l1_elevation = (
            self.val_l1_elevation.compute()
        )  # get current val metric of loss for elevation

        self.val_l1_best_azimuth(
            metric_l1_azimuth
        )  # update best so far val metric for azimuth loss
        self.val_l1_best_elevation(
            metric_l1_elevation
        )  # update best so far val metric for elevation loss

        metric_oriSrc_best = (
            self.val_l1_best_azimuth + self.val_l1_best_elevation
        )  # sum of best so far val metrics for azimuth and elevation loss

        # log `val_loss_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch

        self.log(
            "val/loss_best/azimuth",
            self.val_l1_best_azimuth.compute(),
            sync_dist=True,
            prog_bar=True,
        )

        self.log(
            "val/corrcoef_best/azimuth",
            self.val_corrcoef_azimuth.compute(),
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
            "val/corrcoef_best/elevation",
            self.val_corrcoef_elevation.compute(),
            sync_dist=True,
            prog_bar=True,
        )

        self.log(
            "val/loss_best/ori_src",
            metric_oriSrc_best.compute(),
            sync_dist=True,
            prog_bar=True,
        )

    def test_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a dict) containing the input tensor and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        net_output = self.forward(**batch["net_input"])

        loss = self.criterion_test(net_output, batch["groundtruth"])

        loss_azimuth = loss["loss_azimuth"]
        loss_elevation = loss["loss_elevation"]
        corr_coef_azimuth = loss["corr_coef_azimuth"]
        corr_coef_elevation = loss["corr_coef_elevation"]

        # update and log metrics
        self.test_l1_azimuth(loss_azimuth)
        self.test_l1_elevation(loss_elevation)

        self.test_corrcoef_azimuth(corr_coef_azimuth)
        self.test_corrcoef_elevation(corr_coef_elevation)

        loss_ori_src = loss_azimuth + loss_elevation

        self.log(
            "test/loss/azimuth",
            self.test_l1_azimuth,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.log(
            "test/loss/elevation",
            self.test_l1_elevation,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.log(
            "test/corrcoef/azimuth",
            self.test_corrcoef_azimuth,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.log(
            "test/corrcoef/elevation",
            self.test_corrcoef_elevation,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

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
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        norm_span: Dict[str, Tuple[float, float]],
    ) -> Dict[str, torch.Tensor]:
        # sourcery skip: inline-immediately-returned-variable, merge-dict-assign
        """Perform a single prediction step on a batch of data from the test set.

        :param batch: A batch of data (a dict) containing the input tensor and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        net_output = self.forward(**batch["net_input"])

        azimuth_hat = net_output["azimuth_hat"]
        elevation_hat = net_output["elevation_hat"]
        padding_mask = net_output["padding_mask"]

        if padding_mask is not None and padding_mask.any():
            # B x T
            # B x T
            input_lengths = (1 - padding_mask.long()).sum(-1)

            # apply conv formula to get real output_lengths
            output_lengths = HuberLoss._get_param_pred_output_lengths(input_lengths)

            padding_mask = torch.zeros(
                azimuth_hat.shape[:2],
                dtype=azimuth_hat.dtype,
                device=azimuth_hat.device,
            )

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
            ).bool()  # for padded values, set to True, else False

            reverse_padding_mask = padding_mask.logical_not()

        if padding_mask is not None and padding_mask.any():

            azimuth_hat = (azimuth_hat * reverse_padding_mask).sum(
                dim=1
            ) / reverse_padding_mask.sum(dim=1)

            elevation_hat = (elevation_hat * reverse_padding_mask).sum(
                dim=1
            ) / reverse_padding_mask.sum(dim=1)

        else:
            azimuth_hat = azimuth_hat.mean(dim=1)
            elevation_hat = elevation_hat.mean(dim=1)

        # inverse unitary normalization
        ori_azimuth_hat = unitary_norm_inv(azimuth_hat, lb=-1.000, ub=1.000)
        ori_azimuth_hat = ori_azimuth_hat * torch.pi

        elevation_hat = unitary_norm_inv(elevation_hat, lb=-0.733, ub=0.486)
        ori_azimuth_hat = ori_azimuth_hat * torch.pi

        preds = {"azimuth_hat": azimuth_hat}
        preds["elevation_hat"] = elevation_hat

        return preds

    def configure_optimizers(self) -> Dict[str, Any]:
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
        root / "configs" / "model" / "network_oriSrc_woBC.yaml"
    )
    _ = hydra.utils.instantiate(cfg.model.oriSrcRegressorModule)

    _ = OriSrcRegressorModuleWOBC(cfg.model.oriSrcRegressorModule)
