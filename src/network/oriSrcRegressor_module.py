from typing import Any, Dict, Tuple, Optional

import torch
from lightning import LightningModule
from torchmetrics import (
    MinMetric,
    MaxMetric,
    MeanMetric,
)
from torchmetrics.classification.accuracy import BinaryAccuracy

from src.criterions.see_saw_loss import SeeSawLoss
from src.utils.unitary_linear_norm import unitary_norm_inv

# ======================== OriSrc regression module ========================


class OriSrcRegressorModule(LightningModule):
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

        self.criterion_train = SeeSawLoss(phase="train")
        self.criterion_val = SeeSawLoss(phase="val")
        self.criterion_test = SeeSawLoss(phase="test")

        self.bc_loss_weight = self.hparams.optim_cfg.bc_loss_weight
        self.loss_azimuth_weight = self.hparams.optim_cfg.loss_azimuth_weight
        self.loss_elevation_weight = self.hparams.optim_cfg.loss_elevation_weight
        self.loss_ori_src_weight = self.hparams.optim_cfg.loss_ori_src_weight
        self.bc_loss_weight_alt = self.hparams.optim_cfg.bc_loss_weight_alt

        # for averaging loss across batches
        self.train_bce_bc_azimuth = MeanMetric()
        self.val_bce_bc_azimuth = MeanMetric()
        self.test_bce_bc_azimuth = MeanMetric()

        self.train_huber_azimuth = MeanMetric()
        self.val_l1_azimuth = MeanMetric()
        self.test_l1_azimuth = MeanMetric()

        self.train_acc_bc_azimuth = BinaryAccuracy()
        self.val_acc_bc_azimuth = BinaryAccuracy()
        self.test_acc_bc_azimuth = BinaryAccuracy()

        self.train_acc_bc_elevation = BinaryAccuracy()
        self.val_acc_bc_elevation = BinaryAccuracy()
        self.test_acc_bc_elevation = BinaryAccuracy()

        self.train_huber_elevation = MeanMetric()
        self.val_l1_elevation = MeanMetric()
        self.test_l1_elevation = MeanMetric()

        self.train_bce_bc_elevation = MeanMetric()
        self.val_bce_bc_elevation = MeanMetric()
        self.test_bce_bc_elevation = MeanMetric()

        self.val_corrcoef_azimuth = MeanMetric()
        self.val_corrcoef_elevation = MeanMetric()

        self.test_corrcoef_azimuth = MeanMetric()
        self.test_corrcoef_elevation = MeanMetric()

        # for tracking best so far validation best metrics
        self.val_acc_bc_best_azimuth = MaxMetric()
        self.val_acc_bc_best_elevation = MaxMetric()

        self.val_bce_bc_best_azimuth = MinMetric()
        self.val_bce_bc_best_elevation = MinMetric()

        self.val_l1_best_azimuth = MinMetric()
        self.val_l1_best_elevation = MinMetric()

        self.val_corrcoef_best_azimuth = MaxMetric()
        self.val_corrcoef_best_elevation = MaxMetric()

        self.predict_tools = SeeSawLoss(phase="infer")

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
        self.val_bce_bc_azimuth.reset()
        self.val_bce_bc_elevation.reset()
        self.val_l1_azimuth.reset()
        self.val_l1_elevation.reset()
        self.val_acc_bc_azimuth.reset()
        self.val_acc_bc_elevation.reset()
        self.val_corrcoef_azimuth.reset()
        self.val_corrcoef_elevation.reset()
        self.val_bce_bc_best_azimuth.reset()
        self.val_bce_bc_best_elevation.reset()
        self.val_l1_best_azimuth.reset()
        self.val_l1_best_elevation.reset()
        self.val_corrcoef_best_azimuth.reset()
        self.val_corrcoef_best_elevation.reset()

    def model_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
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
        loss_ori_src = self.criterion_train(net_output, batch["groundtruth"])

        loss_azimuth = loss_ori_src["loss_azimuth"]
        loss_elevation = loss_ori_src["loss_elevation"]

        loss_azimuth_pp = loss_azimuth["loss_pred_all"]["loss_pred"]  # list
        loss_azimuth_bc = loss_azimuth["loss_pred_all"]["bce_loss_judge"][0]
        judge_pred_azimuth = loss_azimuth["judge_pred_label"]

        loss_elevation_pp = loss_elevation["loss_pred_all"]["loss_pred"]  # list
        loss_elevation_bc = loss_elevation["loss_pred_all"]["bce_loss_judge"][0]
        judge_pred_elevation = loss_elevation["judge_pred_label"]

        loss_ori_src = {
            "loss_pp_azimuth": loss_azimuth_pp,
            "loss_pp_elevation": loss_elevation_pp,
            "loss_bc_azimuth": loss_azimuth_bc,
            "loss_bc_elevation": loss_elevation_bc,
        }

        pred_label_ori_src = {
            "preds_label_azimuth": judge_pred_azimuth,
            "preds_label_elevation": judge_pred_elevation,
        }

        return loss_ori_src, pred_label_ori_src

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
        loss_ori_src, preds_label_ori_src = self.model_step(batch)

        # cross entropy loss of binary bias correction
        loss_bc_azimuth = loss_ori_src["loss_bc_azimuth"]
        loss_bc_elevation = loss_ori_src["loss_bc_elevation"]

        # update metrics
        self.train_bce_bc_azimuth(loss_bc_azimuth)
        self.train_bce_bc_elevation(loss_bc_elevation)

        # accuracy of binary bias correction
        target_labels_azimuth = batch["groundtruth"]["azimuth_classif"].long()
        target_labels_elevation = batch["groundtruth"]["elevation_classif"].long()

        # convert target label to long to calculate binary accuracy
        # update metrics
        self.train_acc_bc_azimuth(
            preds_label_ori_src["preds_label_azimuth"], target_labels_azimuth
        )
        self.train_acc_bc_elevation(
            preds_label_ori_src["preds_label_elevation"], target_labels_elevation
        )

        # update metrics determining whether ori_azimuth_hat go through the regressor
        if len(loss_ori_src["loss_pp_azimuth"]) > 0:

            loss_ori_azimuth = loss_ori_src["loss_pp_azimuth"][0]

            self.train_huber_azimuth(loss_ori_azimuth)
        else:
            loss_ori_azimuth = torch.tensor(1e-6).to(loss_bc_azimuth.device)

        # update metrics determining whether ori_elevation_hat go through the regressor
        if len(loss_ori_src["loss_pp_elevation"]) > 0:

            loss_ori_elevation = loss_ori_src["loss_pp_elevation"][0]

            self.train_huber_elevation(loss_ori_elevation)
        else:
            loss_ori_elevation = torch.tensor(1e-6).to(loss_bc_elevation.device)

        loss_bc = self.bc_loss_weight * (loss_bc_azimuth + loss_bc_elevation)

        loss = loss_bc + (
            self.loss_ori_src_weight
            * (
                self.loss_azimuth_weight * loss_ori_azimuth
                + self.loss_elevation_weight * loss_ori_elevation
            )
        ) / (1 + self.bc_loss_weight_alt * loss_bc)

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
            "train/loss/BC_azimuth",
            self.train_bce_bc_azimuth,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "train/loss/BC_elevation",
            self.train_bce_bc_elevation,
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

        self.log(
            "train/acc/BC_azimuth",
            self.train_acc_bc_azimuth,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "train/acc/BC_elevation",
            self.train_acc_bc_elevation,
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
    ) -> None:  # sourcery skip: move-assign
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a dict) containing the input tensor of raw,
            target ori_src.
        :param batch_idx: The index of the current batch.
        """
        # network forward pass
        net_output = self.forward(**batch["net_input"])

        # compute loss
        loss = self.criterion_val(net_output, batch["groundtruth"])

        loss_azimuth = loss["loss_azimuth"]
        loss_elevation = loss["loss_elevation"]

        loss_azimuth_pp = loss_azimuth["loss_pred_all"]["loss_pred"]  # list of losses
        loss_azimuth_bc = loss_azimuth["loss_pred_all"]["bce_loss_judge"][0]
        corrcoef_azimuth = loss_azimuth["corr_coef_pred"][
            "corr_pred"
        ]  # list of corrcoef
        pred_azimuth_label = loss_azimuth["judge_pred_label"]

        loss_elevation_pp = loss_elevation["loss_pred_all"][
            "loss_pred"
        ]  # list of losses
        loss_elevation_bc = loss_elevation["loss_pred_all"]["bce_loss_judge"][0]
        corrcoef_elevation = loss_elevation["corr_coef_pred"][
            "corr_pred"
        ]  # list of corrcoef
        pred_elevation_label = loss_elevation["judge_pred_label"]

        # ------------------- BC cross entropy loss calculation ------------------- #

        # update metrics
        self.val_bce_bc_azimuth(loss_azimuth_bc)
        self.val_bce_bc_elevation(loss_elevation_bc)

        # ----------- accuracy of binary classification ----------- #
        target_azimuth_label = batch["groundtruth"]["azimuth_classif"].long()
        target_elevation_label = batch["groundtruth"]["elevation_classif"].long()

        # update metrics
        self.val_acc_bc_azimuth(pred_azimuth_label, target_azimuth_label)
        self.val_acc_bc_elevation(pred_elevation_label, target_elevation_label)

        loss_bc = self.bc_loss_weight * (loss_azimuth_bc + loss_elevation_bc)

        if len(loss_azimuth_pp) > 0:

            # update metrics
            loss_ori_azimuth = loss_azimuth_pp[0]
            corr_coef_azimuth = corrcoef_azimuth[0]

            self.val_l1_azimuth(loss_ori_azimuth)
            self.val_corrcoef_azimuth(corr_coef_azimuth)

        else:
            loss_ori_azimuth = torch.tensor(1e-6).to(loss_azimuth_bc.device)

        # validate elevation by going through the regressor
        if len(loss_elevation_pp) > 0:

            loss_ori_elevation = loss_elevation_pp[0]

            corr_coef_elevation = corrcoef_elevation[0]

            # update metrics
            self.val_l1_elevation(loss_ori_elevation)
            self.val_corrcoef_elevation(corr_coef_elevation)

        else:
            loss_ori_elevation = torch.tensor(1e-6).to(loss_elevation_bc.device)

        loss_all = loss_bc + (
            (
                self.loss_ori_src_weight
                * (
                    self.loss_azimuth_weight * loss_ori_azimuth
                    + self.loss_elevation_weight * loss_ori_elevation
                )
            )
            / (1 + self.bc_loss_weight_alt * loss_bc)
        )

        loss_ori_src = loss_ori_azimuth + loss_ori_elevation

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
            "val/loss/elevation",
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
            "val/loss/BC_azimuth",
            self.val_bce_bc_azimuth,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "val/loss/BC_elevation",
            self.val_bce_bc_elevation,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "val/acc/BC_azimuth",
            self.val_acc_bc_azimuth,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "val/acc/BC_elevation",
            self.val_acc_bc_elevation,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "val/loss/ori_src",
            loss_ori_src,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "val/loss/total",
            loss_all,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        metric_acc_azimuth_bc = (
            self.val_acc_bc_azimuth.compute()
        )  # get current val metric of binary classification for azimuth
        metric_acc_elevation_bc = (
            self.val_acc_bc_elevation.compute()
        )  # get current val metric of binary classification for elevation

        metric_loss_azimuth_bc = self.val_bce_bc_azimuth.compute()
        metric_loss_elevation_bc = self.val_bce_bc_elevation.compute()

        metric_loss_azimuth = (
            self.val_l1_azimuth.compute()
        )  # get current val metric of loss for azimuth
        metric_loss_elevation = (
            self.val_l1_elevation.compute()
        )  # get current val metric of loss for elevation

        metric_corrcoef_azimuth = (
            self.val_corrcoef_azimuth.compute()
        )  # get current val metric of corrcoef for azimuth
        metric_corrcoef_elevation = (
            self.val_corrcoef_elevation.compute()
        )  # get current val metric of corrcoef for elevation

        self.val_acc_bc_best_azimuth(
            metric_acc_azimuth_bc
        )  # update best so far val metric
        self.val_acc_bc_best_elevation(
            metric_acc_elevation_bc
        )  # update best so far val metric

        self.val_l1_best_azimuth(
            metric_loss_azimuth
        )  # update best so far val metric for azimuth loss
        self.val_l1_best_elevation(
            metric_loss_elevation
        )  # update best so far val metric for elevation loss

        self.val_bce_bc_best_azimuth(
            metric_loss_azimuth_bc
        )  # update best so far val metric for azimuth BC loss
        self.val_bce_bc_best_elevation(
            metric_loss_elevation_bc
        )  # update best so far val metric for elevation BC loss

        self.val_corrcoef_best_azimuth(
            metric_corrcoef_azimuth
        )  # update best so far val metric for azimuth corrcoef
        self.val_corrcoef_best_elevation(
            metric_corrcoef_elevation
        )  # update best so far val metric for elevation corrcoef

        metric_bc_best = self.bc_loss_weight * (
            self.val_bce_bc_best_azimuth + self.val_bce_bc_best_elevation
        )

        metric_oriSrc_best = self.loss_ori_src_weight * (
            self.loss_azimuth_weight * self.val_l1_best_azimuth
            + self.loss_elevation_weight * self.val_l1_best_elevation
        )

        metric_total_best = metric_bc_best + (
            metric_oriSrc_best / (1 + self.bc_loss_weight_alt * metric_bc_best)
        )

        corrcoef_total_best = (
            self.val_corrcoef_best_azimuth + self.val_corrcoef_best_elevation
        )

        # log `val_loss_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch

        self.log(
            "val/loss_best/azimuth",
            self.val_l1_best_azimuth.compute(),
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
            "val/loss_best/BC_azimuth",
            self.val_bce_bc_best_azimuth.compute(),
            sync_dist=True,
            prog_bar=True,
        )

        self.log(
            "val/loss_best/BC_elevation",
            self.val_bce_bc_best_elevation.compute(),
            sync_dist=True,
            prog_bar=True,
        )

        self.log(
            "val/corrcoef_best/ori_azimuth_corrcoef",
            self.val_corrcoef_best_azimuth.compute(),
            sync_dist=True,
            prog_bar=True,
        )

        self.log(
            "val/corrcoef_best/ori_elevation_corrcoef",
            self.val_corrcoef_best_elevation.compute(),
            sync_dist=True,
            prog_bar=True,
        )

        self.log(
            "val/acc_best/BC_azimuth",
            self.val_acc_bc_best_azimuth.compute(),
            sync_dist=True,
            prog_bar=True,
        )

        self.log(
            "val/acc_best/BC_elevation",
            self.val_acc_bc_best_elevation.compute(),
            sync_dist=True,
            prog_bar=True,
        )

        self.log(
            "val/loss_best/total",
            metric_total_best.compute(),
            sync_dist=True,
            prog_bar=True,
        )

        self.log(
            "val/loss_best/ori_src",
            metric_oriSrc_best.compute(),
            sync_dist=True,
            prog_bar=True,
        )

        self.log(
            "val/corrcoef_best/ori_src_corrcoef",
            corrcoef_total_best.compute(),
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

        # compute loss
        loss = self.criterion_test(net_output, batch["groundtruth"])

        loss_azimuth = loss["loss_azimuth"]
        loss_elevation = loss["loss_elevation"]
        corr_coef_azimuth = loss["corr_coef_azimuth"]
        corr_coef_elevation = loss["corr_coef_elevation"]

        loss_ori_src = loss_azimuth + loss_elevation

        # update and log metrics
        self.test_l1_azimuth(loss_azimuth)
        self.test_l1_elevation(loss_elevation)

        self.test_corrcoef_azimuth(corr_coef_azimuth)
        self.test_corrcoef_elevation(corr_coef_elevation)

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
        # network forward pass
        net_output = self.forward(**batch["net_input"])

        azimuth_hat = net_output["azimuth_hat"]
        elevation_hat = net_output["elevation_hat"]
        judge_azimuth_prob = net_output["judge_azimuth_prob"]
        judge_elevation_prob = net_output["judge_elevation_prob"]
        padding_mask = net_output["padding_mask"]

        assert azimuth_hat.shape == elevation_hat.shape

        if padding_mask is not None and padding_mask.any():
            # B x T
            input_lengths = (1 - padding_mask.long()).sum(-1)

            # apply conv formula to get real output_lengths
            output_lengths = self.predict_tools._get_param_pred_output_lengths(
                input_lengths=input_lengths
            )

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
            # -------------------- padding mask handling ------------------- #
            azimuth_hat = (azimuth_hat * reverse_padding_mask).sum(
                dim=1
            ) / reverse_padding_mask.sum(dim=1)
            elevation_hat = (elevation_hat * reverse_padding_mask).sum(
                dim=1
            ) / reverse_padding_mask.sum(dim=1)

        else:
            azimuth_hat = azimuth_hat.mean(dim=1)
            elevation_hat = elevation_hat.mean(dim=1)

        idx_azimuth_false = torch.where(judge_azimuth_prob <= 0.5)[0]
        if len(idx_azimuth_false) > 0:
            azimuth_hat[idx_azimuth_false] = torch.tensor(0.494)

        idx_elevation_false = torch.where(judge_elevation_prob <= 0.5)[0]
        if len(idx_elevation_false) > 0:
            elevation_hat[idx_elevation_false] = torch.tensor(0.604)

        # inverse unitary normalization
        if norm_span is not None:
            lb_azimuth, ub_azimuth = norm_span["azimuth"]
            lb_elevation, ub_elevation = norm_span["elevation"]

            azimuth_hat = unitary_norm_inv(azimuth_hat, lb=lb_azimuth, ub=ub_azimuth)
            elevation_hat = unitary_norm_inv(
                elevation_hat, lb=lb_elevation, ub=ub_elevation
            )

        ori_azimuth_hat = unitary_norm_inv(azimuth_hat, lb=-1.000, ub=1.000)
        ori_azimuth_hat = ori_azimuth_hat * torch.pi

        ori_elevation_hat = unitary_norm_inv(elevation_hat, lb=-0.733, ub=0.486)
        ori_azimuth_hat = ori_azimuth_hat * torch.pi

        preds = {"ori_azimuth_hat": ori_azimuth_hat}
        preds["ori_elevation_hat"] = ori_elevation_hat

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
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "network.yaml")
    _ = hydra.utils.instantiate(cfg.model.oriSrcRegressorModule)

    _ = OriSrcRegressorModule(cfg.model.oriSrcRegressorModule)
