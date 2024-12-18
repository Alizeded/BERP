from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric, MaxMetric

from src.criterions.huber_loss import HuberLoss
from src.utils.unitary_linear_norm import unitary_norm_inv

# ======================== volume regression module ========================


class VolumeRegressorModule(LightningModule):
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

        self.volumeRegressor = net

        # loss function
        self.criterion_train = HuberLoss(phase="train", module="volume")
        self.criterion_val = HuberLoss(phase="val", module="volume")
        self.criterion_test = HuberLoss(phase="test", module="volume")

        # for averaging loss across batches

        self.train_huber_volume = MeanMetric()
        self.val_l1_volume = MeanMetric()
        self.test_l1_volume = MeanMetric()

        self.val_corrcoef_volume = MeanMetric()
        self.test_corrcoef_volume = MeanMetric()

        # for tracking best so far validation best metrics
        self.val_l1_best_volume = MinMetric()
        self.val_corrcoef_best_volume = MaxMetric()

        self.predict_tools = HuberLoss(phase="infer", module="volume")

    def forward(
        self,
        source: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Perform a forward pass.

        :param x: A tensor of waveform
        :return: A tensor of estimated volume.
        """

        return self.volumeRegressor(source, padding_mask)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_l1_volume.reset()
        self.val_l1_best_volume.reset()

        self.val_corrcoef_volume.reset()
        self.val_corrcoef_best_volume.reset()

    def model_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:  # sourcery skip: inline-immediately-returned-variable
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a dict) containing the input tensor
            of raw, target volume.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        # network forward pass
        net_output = self.forward(**batch["net_input"])

        # compute loss
        loss = self.criterion_train(net_output, batch["groundtruth"])

        loss_volume = loss["loss_volume"]

        return loss_volume

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a dict) containing the input tensor of raw,
            target volume.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss_volume = self.model_step(batch)

        # update and log metrics
        self.train_huber_volume(loss_volume)

        self.log(
            "train/loss/volume",
            self.train_huber_volume,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        # return loss or backpropagation will fail
        return loss_volume

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
            target volume.
        :param batch_idx: The index of the current batch.
        """
        net_output = self.forward(**batch["net_input"])

        loss = self.criterion_val(net_output, batch["groundtruth"])

        loss_volume = loss["loss_volume"]
        corr_coef_volume = loss["corr_coef_volume"]

        # update and log metrics
        self.val_l1_volume(loss_volume)
        self.val_corrcoef_volume(corr_coef_volume)

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

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        metric_l1_volume = self.val_l1_volume.compute()  # get current val metric
        metric_corrcoef_volume = self.val_corrcoef_volume.compute()

        self.val_l1_best_volume(metric_l1_volume)  # update best so far val metric
        self.val_corrcoef_best_volume(
            metric_corrcoef_volume
        )  # update best so far val metric

        # log `val_loss_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch

        self.log(
            "val/loss_best/volume",
            self.val_l1_best_volume.compute(),
            sync_dist=True,
            prog_bar=True,
        )

        self.log(
            "val/corrcoef_best/volume",
            self.val_corrcoef_best_volume.compute(),
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

        loss_volume = loss["loss_volume"]
        corr_coef_volume = loss["corr_coef_volume"]

        # update and log metrics
        self.test_l1_volume(loss_volume)
        self.test_corrcoef_volume(corr_coef_volume)

        self.log(
            "test/loss/volume",
            self.test_l1_volume,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.log(
            "test/corrcoef/volume",
            self.test_corrcoef_volume,
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
        norm_span: Dict[str, Tuple[float, float]] = None,
    ) -> Dict[str, torch.Tensor]:
        # sourcery skip: inline-immediately-returned-variable
        """Perform a single prediction step on a batch of data from the test set.

        :param batch: A batch of data (a dict) containing the input tensor and target
            labels.
        :param batch_idx: The index of the current batch.
        :param dataloader_idx: The index of the current dataloader.
        """
        # network forward pass
        net_output = self.forward(**batch["net_input"])

        volume_hat = net_output["volume_hat"]
        padding_mask = net_output["padding_mask"]

        if padding_mask is not None and padding_mask.any():
            # B x T
            input_lengths = (1 - padding_mask.long()).sum(-1)

            # apply conv formula to get real output_lengths
            output_lengths = self.predict_tools._get_param_pred_output_lengths(
                input_lengths=input_lengths
            )

            # these two operations makes sure that all values
            # before the output lengths indices are attended to
            padding_mask = torch.zeros(
                volume_hat.shape[:2], dtype=volume_hat.dtype, device=volume_hat.device
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

        if padding_mask is not None and not padding_mask.any():
            # ----------------- padding mask handling -----------------
            volume_hat = (volume_hat * reverse_padding_mask).sum(
                dim=1
            ) / reverse_padding_mask.sum(dim=1)

        else:
            # ----------------- no padding mask handling -----------------
            volume_hat = volume_hat.mean(dim=1)

        # inverse unitary normalization
        if norm_span is not None:
            lb, ub = norm_span["volume"]
            volume_hat = unitary_norm_inv(volume_hat, lb=lb, ub=ub)

        else:  # default normalization span
            volume_hat = unitary_norm_inv(volume_hat, lb=1.5051, ub=3.9542)

        preds = {"volume_hat": volume_hat}

        return preds

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configures optimizers and learning-rate schedulers to be used for training.

        Normally you'd need one, but in the case of GANs or similar you might need multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(self.volumeRegressor.parameters())

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
                    "monitor": "val/loss/volume",
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
    _ = hydra.utils.instantiate(cfg.model.volumeRegressorModule)

    _ = VolumeRegressorModule(cfg.model.volumeRegressorModule)
