from typing import Any, Optional

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, MinMetric

from src.criterions.e2e_huber_loss import HuberLoss
from src.utils.unitary_linear_norm import unitary_norm_inv

# ======================== volume regression module ========================


class RPRegressorModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        optim_cfg: Optional[dict],
        criterion_config: Optional[dict],
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
        self.criterion_train = HuberLoss(
            phase="train", module=self.hparams.criterion_config.module
        )
        self.criterion_val = HuberLoss(
            phase="val", module=self.hparams.criterion_config.module
        )
        self.criterion_test = HuberLoss(
            phase="test", module=self.hparams.criterion_config.module
        )

        # for averaging loss across batches

        self.train_huber_rap = MeanMetric()
        self.val_l1_rap = MeanMetric()
        self.test_l1_rap = MeanMetric()

        self.val_corrcoef_rap = MeanMetric()
        self.test_corrcoef_rap = MeanMetric()

        # for tracking best so far validation best metrics
        self.val_l1_best_rap = MinMetric()
        self.val_corrcoef_best_rap = MaxMetric()

        self.predict_tools = HuberLoss(
            phase="infer", module=self.hparams.criterion_config.module
        )

    def forward(
        self,
        source: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Perform a forward pass.

        :param x: A tensor of waveform
        :return: A tensor of estimated volume.
        """

        net_output = self.volumeRegressor(source, padding_mask)

        return net_output

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_l1_rap.reset()
        self.val_l1_best_rap.reset()

        self.val_corrcoef_rap.reset()
        self.val_corrcoef_best_rap.reset()

    def model_step(
        self,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
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
        param_hat = net_output[f"{self.hparams.criterion_config.module}_hat"]
        param = batch["groundtruth"][f"{self.hparams.criterion_config.module}"]
        padding_mask = net_output["padding_mask"]
        loss = self.criterion_train(param_hat, param, padding_mask)

        loss = loss["loss"]

        return loss

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a dict) containing the input tensor of raw,
            target volume.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss = self.model_step(batch)

        # update and log metrics
        self.train_huber_rap(loss)

        self.log(
            f"train/loss/{self.hparams.criterion_config.module}",
            self.train_huber_rap,
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
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a dict) containing the input tensor of raw,
            target volume.
        :param batch_idx: The index of the current batch.
        """
        net_output = self.forward(**batch["net_input"])

        param_hat = net_output[f"{self.hparams.criterion_config.module}_hat"]
        param = batch["groundtruth"][f"{self.hparams.criterion_config.module}"]
        padding_mask = net_output["padding_mask"]
        loss = self.criterion_val(param_hat, param, padding_mask)

        loss_volume = loss["loss"]
        corr_coef_volume = loss["corr_coef"]

        # update and log metrics
        self.val_l1_rap(loss_volume)
        self.val_corrcoef_rap(corr_coef_volume)

        self.log(
            f"val/loss/{self.hparams.criterion_config.module}",
            self.val_l1_rap,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            f"val/corrcoef/{self.hparams.criterion_config.module}",
            self.val_corrcoef_rap,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        metric_l1_rap = self.val_l1_rap.compute()  # get current val metric
        metric_corrcoef_rap = self.val_corrcoef_rap.compute()

        self.val_l1_best_rap(metric_l1_rap)  # update best so far val metric
        self.val_corrcoef_best_rap(metric_corrcoef_rap)  # update best so far val metric

        # log `val_loss_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch

        self.log(
            f"val/loss_best/{self.hparams.criterion_config.module}",
            self.val_l1_best_rap.compute(),
            sync_dist=True,
            prog_bar=True,
        )

        self.log(
            f"val/corrcoef_best/{self.hparams.criterion_config.module}",
            self.val_corrcoef_best_rap.compute(),
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
        net_output = self.forward(**batch["net_input"])

        param_hat = net_output[f"{self.hparams.criterion_config.module}_hat"]
        param = batch["groundtruth"][f"{self.hparams.criterion_config.module}"]
        padding_mask = net_output["padding_mask"]

        loss = self.criterion_test(param_hat, param, padding_mask)

        loss_test = loss["loss"]
        corr_coef_test = loss["corr_coef"]

        # update and log metrics
        self.test_l1_rap(loss_test)
        self.test_corrcoef_rap(corr_coef_test)

        self.log(
            f"test/loss/{self.hparams.criterion_config.module}",
            self.test_l1_rap,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.log(
            f"test/corrcoef/{self.hparams.criterion_config.module}",
            self.test_corrcoef_rap,
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
        :param dataloader_idx: The index of the current dataloader.
        """
        # network forward pass
        net_output = self.forward(**batch["net_input"])

        param_hat = net_output[f"{self.hparams.criterion_config.module}_hat"]
        padding_mask = net_output["padding_mask"]

        if padding_mask is not None and padding_mask.any():
            # B x T
            input_lengths = (1 - padding_mask.long()).sum(-1)

            # apply conv formula to get real output_lengths
            output_lengths = self.predict_tools._get_feat_extract_output_lengths(
                input_lengths=input_lengths
            )

            # these two operations makes sure that all values
            # before the output lengths indices are attended to
            padding_mask = torch.zeros(
                param_hat.shape[:2], dtype=param_hat.dtype, device=param_hat.device
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
            param_hat = (param_hat * reverse_padding_mask).sum(
                dim=1
            ) / reverse_padding_mask.sum(dim=1)

        else:
            # ----------------- no padding mask handling -----------------
            param_hat = param_hat.mean(dim=1)

        # inverse unitary normalization
        if self.hparams.criterion_config.module == "ts":
            param_hat = unitary_norm_inv(param_hat, lb=0.0034, ub=0.4452)

        preds = {f"{self.hparams.criterion_config.module}_hat": param_hat}

        return preds

    def configure_optimizers(self) -> dict[str, Any]:
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
    _ = hydra.utils.instantiate(cfg.model.RAPRegressorModule)

    _ = RPRegressorModule(cfg.model.RAPRegressorModule)
