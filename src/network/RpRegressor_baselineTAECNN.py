import math
from typing import Any, Optional

import torch
from lightning import LightningModule
from torch.nn import L1Loss, MSELoss
from torchmetrics import MeanMetric, MinMetric, PearsonCorrCoef

from src.utils.AcousticParameterUtils import (
    CenterTime,
    Clarity,
    Definition,
    EarlyDecayTime,
    ExtenededRIRModel,
    PercentageArticulationLoss,
    RapidSpeechTransmissionIndex,
    ReverberationTime,
)


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
        optim_cfg: Optional[dict] = None,
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

        # for averaging loss across batches
        self.train_loss_Th = MeanMetric()
        self.val_loss_Th = MeanMetric()

        self.train_loss_Tt = MeanMetric()
        self.val_loss_Tt = MeanMetric()

        # for tracking best so far validation best metrics
        self.val_loss_best_Th = MinMetric()
        self.val_loss_best_Tt = MinMetric()

        self.test_loss_Th = MeanMetric()
        self.test_loss_Tt = MeanMetric()

        self.test_corrcoef_Th = MeanMetric()
        self.test_corrcoef_Tt = MeanMetric()

        self.test_loss_sti = MeanMetric()
        self.test_loss_alcons = MeanMetric()
        self.test_loss_tr = MeanMetric()
        self.test_loss_edt = MeanMetric()
        self.test_loss_c80 = MeanMetric()
        self.test_loss_c50 = MeanMetric()
        self.test_loss_d50 = MeanMetric()
        self.test_loss_ts = MeanMetric()

        self.test_corrcoef_sti = MeanMetric()
        self.test_corrcoef_alcons = MeanMetric()
        self.test_corrcoef_tr = MeanMetric()
        self.test_corrcoef_edt = MeanMetric()
        self.test_corrcoef_c80 = MeanMetric()
        self.test_corrcoef_c50 = MeanMetric()
        self.test_corrcoef_d50 = MeanMetric()
        self.test_corrcoef_ts = MeanMetric()

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
        self.val_loss_best_Th.reset()
        self.val_loss_best_Tt.reset()

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

        total_hat = self.forward(raw)

        Th_hat = total_hat[:, 0]
        Tt_hat = total_hat[:, 1]

        loss_Th = math.sqrt(self.Th_weight) * self.joint_loss(Th_hat, Th)
        loss_Tt = math.sqrt(self.Tt_weight) * self.joint_loss(Tt_hat, Tt)

        return loss_Th, loss_Tt

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

        :param batch: A batch of data (a dict) containing the input tensor of raw,
            target Th, target Tt, target volume, target distSrc, target azimuthSrc,
            target elevationSrc.
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

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        raw = batch["raw"]
        Th = batch["Th"]
        Tt = batch["Tt"]
        sti = batch["sti"]
        alcons = batch["alcons"]
        tr = batch["t60"]
        edt = batch["edt"]
        c80 = batch["c80"]
        c50 = batch["c50"]
        d50 = batch["d50"]
        ts = batch["ts"]

        total_hat = self.forward(raw)

        sti_calc = RapidSpeechTransmissionIndex()
        alcons_calc = PercentageArticulationLoss()
        tr_calc = ReverberationTime()
        edt_calc = EarlyDecayTime()
        c80_calc = Clarity(clarity_mode="C80")
        c50_calc = Clarity(clarity_mode="C50")
        d50_calc = Definition()
        ts_calc = CenterTime()

        Th_hat = total_hat[:, 0]
        Tt_hat = total_hat[:, 1]

        sti_hat = torch.zeros_like(Th_hat)
        alcons_hat = torch.zeros_like(Th_hat)
        tr_hat = torch.zeros_like(Th_hat)
        edt_hat = torch.zeros_like(Th_hat)
        c80_hat = torch.zeros_like(Th_hat)
        c50_hat = torch.zeros_like(Th_hat)
        d50_hat = torch.zeros_like(Th_hat)
        ts_hat = torch.zeros_like(Th_hat)
        for i in range(len(Th_hat)):
            syn_rir = ExtenededRIRModel(
                Th_hat[i],
                Tt_hat[i],
                fs=16000,
            )()

            # calculate acoustic parameters
            sti_hat[i] = sti_calc(syn_rir, fs=16000)
            alcons_hat[i] = alcons_calc(sti_hat[i])
            tr_hat[i] = tr_calc(syn_rir, fs=16000)
            edt_hat[i] = edt_calc(syn_rir, fs=16000)
            c80_hat[i] = c80_calc(syn_rir, fs=16000)
            c50_hat[i] = c50_calc(syn_rir, fs=16000)
            d50_hat[i] = d50_calc(syn_rir, fs=16000) / 100  # remove percentage
            ts_hat[i] = ts_calc(syn_rir, fs=16000)

        # calculate loss
        loss_Th = self.joint_loss_test(Th_hat, Th)
        loss_Tt = self.joint_loss_test(Tt_hat, Tt)
        loss_sti = self.joint_loss_test(sti_hat, sti)
        loss_alcons = self.joint_loss_test(alcons_hat, alcons)
        loss_tr = self.joint_loss_test(tr_hat, tr)
        loss_edt = self.joint_loss_test(edt_hat, edt)
        loss_c80 = self.joint_loss_test(c80_hat, c80)
        loss_c50 = self.joint_loss_test(c50_hat, c50)
        loss_d50 = self.joint_loss_test(d50_hat, d50)
        loss_ts = self.joint_loss_test(ts_hat, ts)

        corrcoef_Th = self.joint_corrcoef_test(Th_hat, Th).abs()
        corrcoef_Tt = self.joint_corrcoef_test(Tt_hat, Tt).abs()
        corrcoef_sti = self.joint_corrcoef_test(sti_hat, sti).abs()
        corrcoef_alcons = self.joint_corrcoef_test(alcons_hat, alcons).abs()
        corrcoef_tr = self.joint_corrcoef_test(tr_hat, tr).abs()
        corrcoef_edt = self.joint_corrcoef_test(edt_hat, edt).abs()
        corrcoef_c80 = self.joint_corrcoef_test(c80_hat, c80).abs()
        corrcoef_c50 = self.joint_corrcoef_test(c50_hat, c50).abs()
        corrcoef_d50 = self.joint_corrcoef_test(d50_hat, d50).abs()
        corrcoef_ts = self.joint_corrcoef_test(ts_hat, ts).abs()

        # update and log metrics
        self.test_loss_Th(loss_Th)
        self.test_loss_Tt(loss_Tt)
        self.test_loss_sti(loss_sti)
        self.test_loss_alcons(loss_alcons)
        self.test_loss_tr(loss_tr)
        self.test_loss_edt(loss_edt)
        self.test_loss_c80(loss_c80)
        self.test_loss_c50(loss_c50)
        self.test_loss_d50(loss_d50)
        self.test_loss_ts(loss_ts)

        self.test_corrcoef_Th(corrcoef_Th)
        self.test_corrcoef_Tt(corrcoef_Tt)
        self.test_corrcoef_sti(corrcoef_sti)
        self.test_corrcoef_alcons(corrcoef_alcons)
        self.test_corrcoef_tr(corrcoef_tr)
        self.test_corrcoef_edt(corrcoef_edt)
        self.test_corrcoef_c80(corrcoef_c80)
        self.test_corrcoef_c50(corrcoef_c50)
        self.test_corrcoef_d50(corrcoef_d50)
        self.test_corrcoef_ts(corrcoef_ts)

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
            "test/loss/sti",
            self.test_loss_sti,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "test/corrcoef/sti",
            self.test_corrcoef_sti,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "test/loss/alcons",
            self.test_loss_alcons,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "test/corrcoef/alcons",
            self.test_corrcoef_alcons,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "test/loss/tr",
            self.test_loss_tr,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "test/corrcoef/tr",
            self.test_corrcoef_tr,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "test/loss/edt",
            self.test_loss_edt,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "test/corrcoef/edt",
            self.test_corrcoef_edt,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "test/loss/c80",
            self.test_loss_c80,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "test/corrcoef/c80",
            self.test_corrcoef_c80,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "test/loss/c50",
            self.test_loss_c50,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "test/corrcoef/c50",
            self.test_corrcoef_c50,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "test/loss/d50",
            self.test_loss_d50,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "test/corrcoef/d50",
            self.test_corrcoef_d50,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "test/loss/ts",
            self.test_loss_ts,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "test/corrcoef/ts",
            self.test_corrcoef_ts,
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

        sti_calc = RapidSpeechTransmissionIndex()
        alcons_calc = PercentageArticulationLoss()
        tr_calc = ReverberationTime()
        edt_calc = EarlyDecayTime()
        c80_calc = Clarity(clarity_mode="C80")
        c50_calc = Clarity(clarity_mode="C50")
        d50_calc = Definition()
        ts_calc = CenterTime()

        sti_hat = torch.zeros_like(Th_hat)
        alcons_hat = torch.zeros_like(Th_hat)
        tr_hat = torch.zeros_like(Th_hat)
        edt_hat = torch.zeros_like(Th_hat)
        c80_hat = torch.zeros_like(Th_hat)
        c50_hat = torch.zeros_like(Th_hat)
        d50_hat = torch.zeros_like(Th_hat)
        ts_hat = torch.zeros_like(Th_hat)
        for i in range(len(Th_hat)):
            syn_rir = ExtenededRIRModel(
                Th_hat[i],
                Tt_hat[i],
                fs=16000,
            )()

            # calculate acoustic parameters
            sti_hat[i] = sti_calc(syn_rir, fs=16000)
            alcons_hat[i] = alcons_calc(sti_hat[i])
            tr_hat[i] = tr_calc(syn_rir, fs=16000)
            edt_hat[i] = edt_calc(syn_rir, fs=16000)
            c80_hat[i] = c80_calc(syn_rir, fs=16000)
            c50_hat[i] = c50_calc(syn_rir, fs=16000)
            d50_hat[i] = d50_calc(syn_rir, fs=16000)
            ts_hat[i] = ts_calc(syn_rir, fs=16000)

        preds = {
            "Th_hat": Th_hat,
            "Tt_hat": Tt_hat,
            "sti_hat": sti_hat,
            "alcons_hat": alcons_hat,
            "t60_hat": tr_hat,
            "edt_hat": edt_hat,
            "c80_hat": c80_hat,
            "c50_hat": c50_hat,
            "d50_hat": d50_hat,
            "ts_hat": ts_hat,
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
