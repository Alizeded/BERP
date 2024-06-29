from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric, MaxMetric

from src.criterions.huber_loss import HuberLoss
from src.utils.unitary_linear_norm import unitary_norm_inv

# ======================== RIR regression module ========================


class rirRegressorModule(LightningModule):
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

        self.Th_weight = self.hparams.optim_cfg.Th_weight
        self.Tt_weight = self.hparams.optim_cfg.Tt_weight

        # loss function
        self.criterion_train = HuberLoss(phase="train", module="rir")
        self.criterion_val = HuberLoss(phase="val", module="rir")
        self.criterion_test = HuberLoss(phase="test", module="rir")

        # for averaging loss across batches
        self.train_huber_Th = MeanMetric()
        self.val_l1_Th = MeanMetric()
        self.test_l1_Th = MeanMetric()

        self.train_huber_Tt = MeanMetric()
        self.val_l1_Tt = MeanMetric()
        self.test_l1_Tt = MeanMetric()

        self.val_corrcoef_Th = MeanMetric()
        self.test_corrcoef_Th = MeanMetric()

        self.val_corrcoef_Tt = MeanMetric()
        self.test_corrcoef_Tt = MeanMetric()

        # for tracking best so far validation best metrics
        self.val_l1_best_Th = MinMetric()
        self.val_l1_best_Tt = MinMetric()

        self.val_corrcoef_best_Th = MaxMetric()
        self.val_corrcoef_best_Tt = MaxMetric()

        self.predict_tools = HuberLoss(phase="infer", module="rir")

    def forward(
        self,
        source: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Perform a forward pass.

        :param x: A tensor of waveform
        :return: A tensor of estimated Th, Tt.
        """
        return self.rirRegressor(source, padding_mask)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_l1_Th.reset()
        self.val_l1_Tt.reset()
        self.val_l1_best_Th.reset()
        self.val_l1_best_Tt.reset()

        self.val_corrcoef_Th.reset()
        self.val_corrcoef_Tt.reset()
        self.val_corrcoef_best_Th.reset()
        self.val_corrcoef_best_Tt.reset()

    def model_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a dict) containing the input tensor
            of raw, target Th, target Tt.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """

        # network forward pass
        net_output = self.forward(**batch["net_input"])

        loss = self.criterion_train(net_output, batch["groundtruth"])

        loss_Th = loss["loss_Th"]
        loss_Tt = loss["loss_Tt"]

        return loss_Th, loss_Tt

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a dict) containing the input tensor of raw,
             target Th, target Tt.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss_Th, loss_Tt = self.model_step(batch)

        # update and log metrics
        self.train_huber_Th(loss_Th)
        self.train_huber_Tt(loss_Tt)

        self.log(
            "train/loss/Th",
            self.train_huber_Th,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "train/loss/Tt",
            self.train_huber_Tt,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        total_loss = self.Th_weight * loss_Th + self.Tt_weight * loss_Tt

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
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a dict) containing the input tensor of raw,
            target Th, target Tt.
        :param batch_idx: The index of the current batch.
        """

        # network forward pass
        net_output = self.forward(**batch["net_input"])

        loss = self.criterion_val(net_output, batch["groundtruth"])

        loss_Th = loss["loss_Th"]
        loss_Tt = loss["loss_Tt"]
        corr_coef_Th = loss["corr_coef_Th"]
        corr_coef_Tt = loss["corr_coef_Tt"]

        # update and log metrics
        self.val_l1_Th(loss_Th)
        self.val_l1_Tt(loss_Tt)

        self.val_corrcoef_Th(corr_coef_Th)
        self.val_corrcoef_Tt(corr_coef_Tt)

        self.log(
            "val/loss/Th",
            self.val_l1_Th,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "val/corrcoef/Th",
            self.val_corrcoef_Th,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "val/loss/Tt",
            self.val_l1_Tt,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "val/corrcoef/Tt",
            self.val_corrcoef_Tt,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        total_loss = self.Th_weight * loss_Th + self.Tt_weight * loss_Tt

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
        metric_l1_Th = self.val_l1_Th.compute()  # get current val metric
        metric_l1_Tt = self.val_l1_Tt.compute()  # get current val metric

        metric_corrcoef_Th = self.val_corrcoef_Th.compute()
        metric_corrcoef_Tt = self.val_corrcoef_Tt.compute()

        self.val_l1_best_Th(metric_l1_Th)  # update best so far val metric
        self.val_l1_best_Tt(metric_l1_Tt)  # update best so far val metric

        self.val_corrcoef_best_Th(metric_corrcoef_Th)  # update best so far val metric
        self.val_corrcoef_best_Tt(metric_corrcoef_Tt)  # update best so far val metric

        total_loss_best = (
            self.val_l1_best_Th * self.Th_weight + self.val_l1_best_Tt * self.Tt_weight
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
            self.val_l1_best_Th.compute(),
            sync_dist=True,
            prog_bar=True,
        )

        self.log(
            "val/coorcoef_best/Th",
            self.val_corrcoef_best_Th.compute(),
            sync_dist=True,
            prog_bar=True,
        )

        self.log(
            "val/loss_best/Tt",
            self.val_l1_best_Tt.compute(),
            sync_dist=True,
            prog_bar=True,
        )

        self.log(
            "val/coorcoef_best/Tt",
            self.val_corrcoef_best_Tt.compute(),
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
        # network forward pass
        net_output = self.forward(**batch["net_input"])

        loss = self.criterion_test(net_output, batch["groundtruth"])

        loss_Th = loss["loss_Th"]
        loss_Tt = loss["loss_Tt"]
        corr_coef_Th = loss["corr_coef_Th"]
        corr_coef_Tt = loss["corr_coef_Tt"]

        # update and log metrics
        self.test_l1_Th(loss_Th)
        self.test_l1_Tt(loss_Tt)

        self.test_corrcoef_Th(corr_coef_Th)
        self.test_corrcoef_Tt(corr_coef_Tt)

        self.log(
            "test/loss/Th",
            self.test_l1_Th,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "test/loss/Tt",
            self.test_l1_Tt,
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
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        norm_span: Dict[str, Tuple[float, float]] = None,
    ) -> Dict[str, torch.Tensor]:
        # sourcery skip: inline-immediately-returned-variable, merge-dict-assign
        """Perform a single prediction step on a batch of data from the test set.

        :param batch: A batch of data (a dict) containing the input tensor and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        # network forward pass
        net_output = self.forward(**batch["net_input"])

        Th_hat = net_output["Th_hat"]
        Tt_hat = net_output["Tt_hat"]
        padding_mask = net_output["padding_mask"]

        if padding_mask is not None and padding_mask.any():

            # B x T
            input_lengths = (1 - padding_mask.long()).sum(-1)

            # apply conv formula to get real output_lengths
            output_lengths = self.predict_tools._get_param_pred_output_lengths(
                input_lengths=input_lengths
            )

            padding_mask = torch.zeros(
                Tt_hat.shape[:2], dtype=Tt_hat.dtype, device=Tt_hat.device
            )  # B x T

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
            ).bool()  # for padded values, set to True

            reverse_padding_mask = padding_mask.logical_not()

        if padding_mask is not None and not padding_mask.any():

            # ----------------- padding mask handling -----------------

            Th_hat = (Th_hat * reverse_padding_mask).sum(
                dim=1
            ) / reverse_padding_mask.sum(
                dim=1
            )  # Collapse as a single value intead of a straight-line prediction
            Tt_hat = (Tt_hat * reverse_padding_mask).sum(
                dim=1
            ) / reverse_padding_mask.sum(
                dim=1
            )  # Collapse as a single value intead of a straight-line prediction

        else:

            # ----------------- no padding mask handling -----------------

            Th_hat = Th_hat.mean(
                dim=1
            )  #  collapse as a single value intead of a straight-line prediction
            Tt_hat = Tt_hat.mean(
                dim=1
            )  #  collapse as a single value intead of a straight-line prediction

        # inverse unitary normalization
        if norm_span is not None:
            lb_Th, ub_Th = norm_span["Th"]
            Th_hat = unitary_norm_inv(Th_hat, lb=lb_Th, ub=ub_Th)

        else:  # default values
            Th_hat = unitary_norm_inv(Th_hat, lb=0.005, ub=0.276)

        preds = {"Th_hat": Th_hat}
        preds["Tt_hat"] = Tt_hat

        return preds

    def configure_optimizers(self) -> Dict[str, Any]:
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
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "network.yaml")
    _ = hydra.utils.instantiate(cfg.model.rirRegressorModule)

    _ = rirRegressorModule(cfg.model.rirRegressorModule)
