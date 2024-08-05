from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, MinMetric

from src.criterions.label_smmothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterion,
)

# ========================= numEstimator Module =========================


class numEstimatorModule(LightningModule):
    """a `LightningModule`

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        optim_cfg: Optional[dict],
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    ):
        """Initialize a denoiserClassiferModule.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to metricess init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore="net")

        self.numEstimator = net

        # loss function
        self.criterion = LabelSmoothedCrossEntropyCriterion(label_smoothing=0.1)

        # metric objects for calculating and averaging metricuracy across batches

        self.train_l1 = MeanMetric()
        self.val_l1 = MeanMetric()
        self.test_l1 = MeanMetric()

        self.train_f1 = MeanMetric()
        self.val_f1 = MeanMetric()
        self.test_f1 = MeanMetric()

        self.val_accu = MeanMetric()
        self.test_accu = MeanMetric()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation metricuracy
        self.val_l1_best = MinMetric()
        self.val_f1_best = MaxMetric()
        self.val_accu_best = MaxMetric()

        # activate manual optimization
        self.automatic_optimization = True

    def forward(self, source: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model self.numEstimator.
        and self.numEstimator.

        :param x: A tensor of mixed audio.
        :return: A tensor of predicted logits.
        """
        return self.numEstimator(source, padding_mask)

    def model_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a Dict) containing
        the input tensor of mixed audio and target mixed speech label and padding mask.

        :return: A tuple containing the model predictions, loss, l1 distance, metricuracy and f1 score.

        """

        # # unmask the padded labels
        # mixed_label = mixed_label.masked_select(~padding_mask)

        net_output = self.forward(**batch["net_input"])
        log_prob = self.numEstimator.get_normalized_probs(net_output, log_prob=True)
        loss, l1, accu, f1 = self.criterion(
            log_prob,
            batch["groundtruth"]["target"],
            net_output["padding_mask"],
            batch["groundtruth"]["target_padding_mask"],
            reduce=True,
        )

        return log_prob, loss, l1, accu, f1

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks

        self.val_loss.reset()

        self.val_l1.reset()
        self.val_l1_best.reset()

        self.val_f1.reset()
        self.val_f1_best.reset()

        self.val_accu.reset()
        self.val_accu_best.reset()

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a Dict) containing
         the input tensor of mixed audio and target mixed speech label and padding mask.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets
         for numEstimator.
        """

        # forward pass and calculate loss
        _, loss, l1, accu, f1 = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_l1(l1)
        self.train_f1(f1)

        self.log(
            "train/loss",
            self.train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train/l1_dist",
            self.train_l1,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train/F1_score",
            self.train_f1,
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

        :param batch: A batch of data (a Dict) containing
         the input tensor of mixed audio and target mixed speech label and padding mask.
        :param batch_idx: The index of the current batch.
        """
        (
            _,
            loss,
            l1,
            accu,
            f1,
        ) = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)

        self.val_l1(l1)

        self.val_f1(f1)

        self.val_accu(accu)

        self.log(
            "val/loss",
            self.val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "val/l1_dist",
            self.val_l1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "val/F1_score",
            self.val_f1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "val/accuracy",
            self.val_accu,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."

        l1s = self.val_l1.compute()  # get current val metric
        self.val_l1_best(l1s)  # update best val metric

        f1s = self.val_f1.compute()  # get current val metric
        self.val_f1_best(f1s)  # update best val metric

        accus = self.val_accu.compute()  # get current val metric
        self.val_accu_best(accus)  # update best val metric

        # log `val_metric_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log(
            "val/loss_l1_best",
            self.val_l1_best.compute(),
            sync_dist=True,
            prog_bar=True,
        )

        self.log(
            "val/F1_score_best",
            self.val_f1_best.compute(),
            sync_dist=True,
            prog_bar=True,
        )

        self.log(
            "val/accuracy_best",
            self.val_accu_best.compute(),
            sync_dist=True,
            prog_bar=True,
        )

    def test_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a Dict) containing
         the input tensor of mixed audio and target mixed speech label and padding mask.
        :param batch_idx: The index of the current batch.
        """

        # forward pass and calculate loss
        _, loss, l1, f1, accu = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)

        self.test_l1(l1)

        self.test_f1(f1)

        self.test_accu(accu)

        self.log(
            "test/loss",
            self.test_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.log(
            "test/l1_dist",
            self.test_l1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.log(
            "test/F1_score",
            self.test_f1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.log(
            "test/accuracy",
            self.test_accu,
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
        # sourcery skip: inline-immediately-returned-variable
        """Perform a single prediction step on a batch of data from the test set.

        :param batch: A batch of data (a Dict) containing
         the input tensor of mixed audio and target mixed speech label and padding mask.

        :param batch_idx: The index of the current batch.

        :return: A Dict containing the model predictions.
        """
        log_prob, _, _, _, _ = self.model_step(batch)

        pred_label = torch.argmax(log_prob, dim=-1)  # B T C -> B T

        return pred_label

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configures optimizers and learning-rate schedulers to be used for training.

        Normally you'd need one, but in the case of GANs or similar you might need multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        # optimizer for denoiser
        optimizer = self.hparams.optimizer(self.numEstimator.parameters())

        # ================ scheduler for denoiser and numEstimator ================
        if self.hparams.scheduler is not None:
            init_lr = self.hparams.optim_cfg.init_lr
            peak_lr = self.hparams.optim_cfg.peak_lr
            max_steps = self.trainer.estimated_stepping_batches
            final_lr_scale = self.hparams.optim_cfg.final_lr_scale

            # max_steps = self.trainer.estimated_stepping_batches
            # T_max = self.hparams.optim_cfg.T_max
            # eta_min = 1e-8

            # scheduler for numEstimator
            scheduler = self.hparams.scheduler(
                optimizer=optimizer,
                init_lr=init_lr,
                peak_lr=peak_lr,
                max_steps=max_steps,
                final_lr_scale=final_lr_scale,
            )

            # for using the ReduceLROnPlateau scheduler or other value-conditioned schedulers
            return (
                {  # optimizer for denoiser
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "step",
                        "monitor": "val/l1_dist",
                        "frequency": 1,
                    },
                },
            )

        return optimizer


if __name__ == "__main__":
    import hydra
    import omegaconf
    import rootutils

    root = rootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "network.yaml")
    _ = hydra.utils.instantiate(cfg.numEstimatorModule)

    _ = numEstimatorModule(cfg.numEstimatorModule)
