from typing import Dict

import torch
import torch.nn as nn

from torch.nn import SmoothL1Loss, L1Loss, BCELoss
from torchmetrics import PearsonCorrCoef

import einops

from src.utils.unitary_linear_norm import unitary_norm_inv


class SeeSawLoss(nn.Module):

    def __init__(self, phase: str):
        super(SeeSawLoss, self).__init__()
        self.phase = phase
        self.huber_loss = SmoothL1Loss()
        self.l1_loss = L1Loss()
        self.bce_loss = BCELoss()
        self.pearson_corr = PearsonCorrCoef()

        assert (
            self.phase == "train"
            or self.phase == "val"
            or self.phase == "test"
            or self.phase == "infer"
        ), "phase should be either 'train' or 'val' or 'test' or 'infer'"

    def _get_param_pred_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride, padding):
            return torch.floor(
                (input_length + 2 * padding - (kernel_size - 1) - 1) / stride + 1
            )

        room_parametric_predictor_layers = [
            (384, 3, 1, 1),
            (384, 3, 1, 1),
            (384, 5, 3, 0),
            (384, 3, 1, 1),
        ]

        for i in range(len(room_parametric_predictor_layers)):
            input_lengths = _conv_out_length(
                input_lengths,
                room_parametric_predictor_layers[i][1],
                room_parametric_predictor_layers[i][2],
                room_parametric_predictor_layers[i][3],
            )

        return input_lengths.to(torch.long)

    def see_saw_loss_calculate(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        judge_prob: torch.Tensor,
        judge_label: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate the see-saw loss
        """

        if padding_mask is not None and padding_mask.any():
            # B x T
            input_lengths = (1 - padding_mask.long()).sum(-1)
            # apply conv formula to get real output_lengths
            output_lengths = self._get_param_pred_output_lengths(input_lengths)

            padding_mask = torch.zeros(
                pred.shape[:2], dtype=pred.dtype, device=pred.device
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
            ).bool()  # padded value is True, else False

            # reverse the padding mask, non-padded value is True, else False
            reverse_padding_mask = padding_mask.logical_not()

            # -------------------- padding mask handling --------------------
        if self.phase == "train" or self.phase == "val":

            # ----------------- BCE Loss for bias corrector -----------------
            # Convert the judge_label to float to calculate the BCE loss
            judge_label = judge_label.long().float().to(judge_prob.device)

            # Calculate the BCE loss
            bce_loss = self.bce_loss(judge_prob, judge_label)

            loss_pred_all = dict(
                bce_loss_judge=[],
                loss_pred=[],
            )

            corr_coef_pred_val = dict(
                corr_pred=[],
            )

            loss_pred_all["bce_loss_judge"].append(bce_loss)

            if self.phase == "train":
                # threshold = 0.4 for the judge_prob
                idx_pred = torch.where(judge_prob >= 0.4)[0]
            elif self.phase == "val":
                idx_pred = torch.where(judge_prob >= 0.5)[0]
            else:
                raise ValueError(
                    f"Invalid phase: {self.phase}, should be either 'train' or 'val'"
                )

            if len(idx_pred) > 0:

                pred = pred[idx_pred, ...]
                target = target[idx_pred]

                if padding_mask is not None and padding_mask.any():

                    reverse_padding_mask = reverse_padding_mask[idx_pred, ...]

                    if self.phase == "train":
                        # repeat the target to match the shape of pred
                        target = einops.repeat(target, "B -> B T", T=pred.shape[1])

                        # Apply padding mask to the pred and target values
                        pred = pred.masked_select(reverse_padding_mask)
                        target = target.masked_select(reverse_padding_mask)

                        # calculate the huber loss
                        loss_pred_pp = self.huber_loss(pred, target)

                    elif self.phase == "val":
                        # Collapse the pred value to match the target value
                        pred = (pred * reverse_padding_mask).sum(
                            dim=1
                        ) / reverse_padding_mask.sum(dim=1)

                        # calculate the huber loss
                        loss_pred_pp = self.l1_loss(pred, target)

                        # Compute the Pearson Correlation Coefficient
                        corr_pred_pp = self.pearson_corr(pred, target).abs()

                else:
                    # -------------------- no padding mask handling --------------------
                    if self.phase == "train":
                        # repeat the target values to match the pred values
                        target = einops.repeat(target, "B -> B T", T=pred.shape[1])

                        # Calculate the loss
                        loss_pred_pp = self.huber_loss(pred, target)

                    elif self.phase == "val":
                        # Collapse the pred value to match the target value
                        pred = pred.mean(dim=1)

                        # Calculate the loss
                        loss_pred_pp = self.l1_loss(pred, target)

                        # Compute the Pearson Correlation Coefficient
                        corr_pred_pp = self.pearson_corr(pred, target).abs()

                loss_pred_all["loss_pred"].append(loss_pred_pp)

                if self.phase == "val":
                    corr_coef_pred_val["corr_pred"].append(corr_pred_pp)

                else:
                    NotImplementedError(
                        "Correlation Coefficient is only calculated during validation phase"
                    )

            judge_pred_label = torch.where(judge_prob > 0.5, 1, 0)

            if self.phase == "train":
                return {
                    "loss_pred_all": loss_pred_all,
                    "judge_pred_label": judge_pred_label,
                }
            elif self.phase == "val":
                return {
                    "loss_pred_all": loss_pred_all,
                    "corr_coef_pred": corr_coef_pred_val,
                    "judge_pred_label": judge_pred_label,
                }

        elif self.phase == "test":

            if padding_mask is not None and padding_mask.any():

                # collapse the pred value to match the target value
                pred = (pred * reverse_padding_mask).sum(
                    dim=1
                ) / reverse_padding_mask.sum(dim=1)

            else:
                # collapse the pred value to match the target value
                pred = pred.mean(dim=1)

            return {
                "pred": pred,
                "judge_prob": judge_prob,
            }

    def forward(
        self,
        param_hat: Dict[str, torch.Tensor],
        param_groundtruth: Dict[str, torch.Tensor],
    ):

        azimuth_hat = param_hat["azimuth_hat"]
        azimuth = param_groundtruth["azimuth"]
        azimuth_label = param_groundtruth["azimuth_classif"]

        elevation_hat = param_hat["elevation_hat"]
        elevation = param_groundtruth["elevation"]
        elevation_label = param_groundtruth["elevation_classif"]

        judge_prob_azimuth = param_hat["judge_prob_azimuth"]
        judge_prob_elevation = param_hat["judge_prob_elevation"]

        padding_mask = param_hat["padding_mask"]

        assert azimuth_hat.shape == elevation_hat.shape

        if self.phase == "train":

            loss_azimuth = self.see_saw_loss_calculate(
                azimuth_hat, azimuth, judge_prob_azimuth, azimuth_label, padding_mask
            )

            loss_elevation = self.see_saw_loss_calculate(
                elevation_hat,
                elevation,
                judge_prob_elevation,
                elevation_label,
                padding_mask,
            )

            return {
                "loss_azimuth": loss_azimuth,
                "loss_elevation": loss_elevation,
            }

        elif self.phase == "val":

            loss_azimuth = self.see_saw_loss_calculate(
                azimuth_hat, azimuth, judge_prob_azimuth, azimuth_label, padding_mask
            )

            loss_elevation = self.see_saw_loss_calculate(
                elevation_hat,
                elevation,
                judge_prob_elevation,
                elevation_label,
                padding_mask,
            )

            return {
                "loss_azimuth": loss_azimuth,
                "loss_elevation": loss_elevation,
            }

        elif self.phase == "test":

            output_azimuth = self.see_saw_loss_calculate(
                azimuth_hat, azimuth, judge_prob_azimuth, azimuth_label, padding_mask
            )

            azimuth_hat = output_azimuth["pred"]
            judge_prob_azimuth = output_azimuth["judge_prob"]

            idx_azimuth_false = torch.where(judge_prob_azimuth < 0.5)[0]
            if len(idx_azimuth_false) > 0:
                azimuth_hat[idx_azimuth_false] = torch.tensor(0.494)

            output_elevation = self.see_saw_loss_calculate(
                elevation_hat,
                elevation,
                judge_prob_elevation,
                elevation_label,
                padding_mask,
            )

            elevation_hat = output_elevation["pred"]
            judge_prob_elevation = output_elevation["judge_prob"]

            idx_elevation_false = torch.where(judge_prob_elevation < 0.5)[0]
            if len(idx_elevation_false) > 0:
                elevation_hat[idx_elevation_false] = torch.tensor(0.604)

            # inverse the unitary normalization
            azimuth_hat = unitary_norm_inv(azimuth_hat, lb=-1.000, ub=1.000)
            azimuth = unitary_norm_inv(azimuth, lb=-1.000, ub=1.000)
            elevation_hat = unitary_norm_inv(elevation_hat, lb=-0.733, ub=0.486)
            elevation = unitary_norm_inv(elevation, lb=-0.733, ub=0.486)

            # MAE metric
            loss_azimuth = self.l1_loss(azimuth_hat, azimuth)
            loss_elevation = self.l1_loss(elevation_hat, elevation)

            # Compute the Pearson Correlation Coefficient
            corr_coef_azimuth = self.pearson_corr(azimuth_hat, azimuth).abs()
            corr_coef_elevation = self.pearson_corr(elevation_hat, elevation).abs()

            return {
                "loss_azimuth": loss_azimuth,
                "loss_elevation": loss_elevation,
                "corr_coef_azimuth": corr_coef_azimuth,
                "corr_coef_elevation": corr_coef_elevation,
            }
