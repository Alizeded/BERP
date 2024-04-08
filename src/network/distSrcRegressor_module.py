from typing import Any, Dict, Optional

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MinMetric, MeanMetric

from src.criterions.huber_loss import HuberLoss

# ======================== dist regression module ========================


class DistSrcRegressorModule(LightningModule):
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

        self.dist_srcRegressor = net

        # loss function
        self.criterion_train = HuberLoss(phase="train", module="distance")
        self.criterion_val = HuberLoss(phase="val", module="distance")
        self.criterion_test = HuberLoss(phase="test", module="distance")

        # for averaging loss across batches

        self.train_huber_dist_src = MeanMetric()
        self.val_l1_dist_src = MeanMetric()
        self.test_l1_dist_src = MeanMetric()

        self.val_corrcoef_dist_src = MeanMetric()
        self.test_corrcoef_dist_src = MeanMetric()

        # for tracking best so far validation best metrics
        self.val_l1_best_dist_src = MinMetric()
        self.val_corrcoef_best_dist_src = MaxMetric()

        self.predict_tools = HuberLoss(phase="infer", module="distance")

    def forward(
        self,
        source: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Perform a forward pass.

        :param x: A tensor of waveform
        :return: A tensor of estimated dist_src.
        """
        net_output = self.dist_srcRegressor(source, padding_mask)

        return net_output

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_l1_dist_src.reset()
        self.val_corrcoef_dist_src.reset()

        self.val_l1_best_dist_src.reset()
        self.val_corrcoef_best_dist_src.reset()

    def model_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a dict) containing the input tensor
            of raw, target dist_src.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        # network forward pass
        net_output = self.forward(**batch["net_input"])

        # compute loss
        loss = self.criterion_train(net_output, batch["groundtruth"])

        loss_dist_src = loss["loss_dist_src"]

        return loss_dist_src

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
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
        self.train_huber_dist_src(loss_dist_src)

        self.log(
            "train/loss/dist_src",
            self.train_huber_dist_src,
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
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a dict) containing the input tensor of raw,
            target dist_src.
        :param batch_idx: The index of the current batch.
        """
        # network forward pass
        net_output = self.forward(**batch["net_input"])

        # compute loss
        loss = self.criterion_val(net_output, batch["groundtruth"])

        loss_dist_src = loss["loss_dist_src"]
        corr_coef_dist_src = loss["corr_coef_dist_src"]

        # update and log metrics
        self.val_l1_dist_src(loss_dist_src)
        self.val_corrcoef_dist_src(corr_coef_dist_src)

        self.log(
            "val/loss/dist_src",
            self.val_l1_dist_src,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "val/corrcoef/dist_src",
            self.val_corrcoef_dist_src,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        metric_l1_dist_src = self.val_l1_dist_src.compute()  # get current val metric
        metric_corrcoef_dist_src = (
            self.val_corrcoef_dist_src.compute()
        )  # get current val metric

        self.val_l1_best_dist_src(metric_l1_dist_src)  # update best so far val metric
        self.val_corrcoef_best_dist_src(
            metric_corrcoef_dist_src
        )  # update best so far val metric

        # log `val_loss_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch

        self.log(
            "val/loss_best/dist_src",
            self.val_l1_best_dist_src.compute(),
            sync_dist=True,
            prog_bar=True,
        )

        self.log(
            "val/corrcoef_best/dist_src",
            self.val_corrcoef_best_dist_src.compute(),
            sync_dist=True,
            prog_bar=True,
        )

    def test_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        # network forward pass
        net_output = self.forward(**batch["net_input"])

        # compute loss
        loss = self.criterion_test(net_output, batch["groundtruth"])

        loss_dist_src = loss["loss_dist_src"]
        corr_coef_dist_src = loss["corr_coef_dist_src"]

        # update and log metrics
        self.test_l1_dist_src(loss_dist_src)
        self.test_corrcoef_dist_src(corr_coef_dist_src)

        self.log(
            "test/loss/dist_src",
            self.test_l1_dist_src,
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
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """Perform a single prediction step on a batch of data from the test set.

        :param batch: A batch of data (a dict) containing the input tensor and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        # network forward pass
        net_output = self.forward(**batch["net_input"])

        dist_src_hat = net_output["dist_src_hat"]
        padding_mask = net_output["padding_mask"]

        if padding_mask is not None and padding_mask.any():
            # B x T
            input_lengths = (1 - padding_mask.long()).sum(-1)

            # apply conv formula to get real output_lengths
            output_lengths = self.predict_tools._get_param_pred_output_lengths(
                input_lengths=input_lengths
            )

            padding_mask = torch.zeros(
                dist_src_hat.shape[:2],
                dtype=dist_src_hat.dtype,
                device=dist_src_hat.device,
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
            dist_src_hat = (dist_src_hat * reverse_padding_mask).sum(
                dim=1
            ) / reverse_padding_mask.sum(dim=1)

        else:
            # ----------------- no padding mask handling -----------------
            dist_src_hat = dist_src_hat.mean(dim=1)

        preds = {"dist_src_hat": dist_src_hat}

        return preds

    def configure_optimizers(self) -> Dict[str, Any]:
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
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "network.yaml")
    _ = hydra.utils.instantiate(cfg.model.distSrcRegressorModule)

    _ = DistSrcRegressorModule(cfg.model.distSrcRegressorModule)
