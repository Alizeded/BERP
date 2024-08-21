from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import SmoothL1Loss, L1Loss

from torchmetrics import PearsonCorrCoef

from src.utils.unitary_linear_norm import unitary_norm_inv


class JointEstimationEvaluation(nn.Module):
    def __init__(self):
        super(JointEstimationEvaluation, self).__init__()

        self.criterion = SmoothL1Loss()
        self.l1_loss = L1Loss()
        self.pearson_corr_coef = PearsonCorrCoef()

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

    def forward(
        self,
        param_hat: Dict[str, torch.Tensor],
        param_groundtruth: Dict[str, torch.Tensor],
    ):
        # extract predicted parameters
        # force cast to float to avoid mismatch in data type
        Th_hat = param_hat["Th_hat"]
        Tt_hat = param_hat["Tt_hat"]
        volume_hat = param_hat["volume_hat"]
        dist_src_hat = param_hat["dist_src_hat"]
        azimuth_hat = param_hat["azimuth_hat"]
        elevation_hat = param_hat["elevation_hat"]
        judge_logits_azimuth = param_hat["judge_logits_azimuth"]
        judge_logits_elevation = param_hat["judge_logits_elevation"]
        padding_mask = param_hat["padding_mask"]

        # extract groundtruth parameters
        # force cast to float to avoid mismatch in data type
        Th = param_groundtruth["Th"]
        Tt = param_groundtruth["Tt"]
        volume = param_groundtruth["volume"]
        dist_src = param_groundtruth["dist_src"]
        azimuth = param_groundtruth["azimuth"]
        elevation = param_groundtruth["elevation"]

        assert (
            Th_hat.shape
            == Tt_hat.shape
            == volume_hat.shape
            == dist_src_hat.shape
            == azimuth_hat.shape
            == elevation_hat.shape
        )

        # ----------------- obtain padding mask ----------------- #
        if padding_mask is not None and padding_mask.any():
            # B x T
            input_lengths = (1 - padding_mask.long()).sum(-1)
            # apply conv formula to get real output_lengths
            output_lengths = self._get_param_pred_output_lengths(input_lengths)

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
            ).bool()  # padded value is True, else False

            # reverse the padding mask, non-padded value is True, else False
            reverse_padding_mask = padding_mask.logical_not()

        #! ------------------- compute evaluation metrics ------------------- #
        # ------------------------- polynomial loss ------------------------- #
        # collapse all the predicted parameters to batch size 1D tensor
        if padding_mask is not None and padding_mask.any():
            # ---------------------- padding mask handling ---------------------- #
            azimuth_hat = (azimuth_hat * reverse_padding_mask).sum(
                dim=1
            ) / reverse_padding_mask.sum(dim=1)

            elevation_hat = (elevation_hat * reverse_padding_mask).sum(
                dim=1
            ) / reverse_padding_mask.sum(dim=1)

            Th_hat = (Th_hat * reverse_padding_mask).sum(
                dim=1
            ) / reverse_padding_mask.sum(dim=1)

            Tt_hat = (Tt_hat * reverse_padding_mask).sum(
                dim=1
            ) / reverse_padding_mask.sum(dim=1)

            volume_hat = (volume_hat * reverse_padding_mask).sum(
                dim=1
            ) / reverse_padding_mask.sum(dim=1)

            dist_src_hat = (dist_src_hat * reverse_padding_mask).sum(
                dim=1
            ) / reverse_padding_mask.sum(dim=1)

        else:
            # ---------------------- no padding mask handling ---------------------- #
            azimuth_hat = azimuth_hat.mean(dim=1)
            elevation_hat = elevation_hat.mean(dim=1)
            Th_hat = Th_hat.mean(dim=1)
            Tt_hat = Tt_hat.mean(dim=1)
            volume_hat = volume_hat.mean(dim=1)
            dist_src_hat = dist_src_hat.mean(dim=1)

        # Calculate judge probability for azimuth and elevation
        judge_prob_azimuth = F.sigmoid(judge_logits_azimuth)
        judge_prob_elevation = F.sigmoid(judge_logits_elevation)

        idx_azimuth_pp_false = torch.where(judge_prob_azimuth < 0.5)[0]
        if len(idx_azimuth_pp_false) > 0:
            azimuth_hat[idx_azimuth_pp_false] = torch.tensor(0.4986)

        idx_elevation_pp_false = torch.where(judge_prob_elevation < 0.5)[0]
        if len(idx_elevation_pp_false) > 0:
            elevation_hat[idx_elevation_pp_false] = torch.tensor(0.5977)

        # ------------------- inverse unitary normalization -------------------
        Th_hat = unitary_norm_inv(Th_hat, lb=0.005, ub=0.276)
        Th = unitary_norm_inv(Th, lb=0.005, ub=0.276)
        volume_hat = unitary_norm_inv(volume_hat, lb=1.5051, ub=3.9542)
        volume = unitary_norm_inv(volume, lb=1.5051, ub=3.9542)
        dist_src_hat = unitary_norm_inv(dist_src_hat, lb=0.191, ub=28.350)
        dist_src = unitary_norm_inv(dist_src, lb=0.191, ub=28.350)
        azimuth_hat = unitary_norm_inv(azimuth_hat, lb=-1.000, ub=1.000)
        azimuth = unitary_norm_inv(azimuth, lb=-1.000, ub=1.000)
        elevation_hat = unitary_norm_inv(elevation_hat, lb=-0.733, ub=0.486)
        elevation = unitary_norm_inv(elevation, lb=-0.733, ub=0.486)

        # MAE metric for all the predicted parameters
        loss_Th = self.l1_loss(Th_hat, Th)
        loss_Tt = self.l1_loss(Tt_hat, Tt)
        loss_volume = self.l1_loss(volume_hat, volume)
        loss_dist_src = self.l1_loss(dist_src_hat, dist_src)
        loss_azimuth = self.l1_loss(azimuth_hat, azimuth)
        loss_elevation = self.l1_loss(elevation_hat, elevation)

        # Pearson correlation coefficient for all the predicted parameters
        corr_Th = self.pearson_corr_coef(Th_hat, Th).abs()
        corr_Tt = self.pearson_corr_coef(Tt_hat, Tt).abs()
        corr_volume = self.pearson_corr_coef(volume_hat, volume).abs()
        corr_dist_src = self.pearson_corr_coef(dist_src_hat, dist_src).abs()
        corr_azimuth = self.pearson_corr_coef(azimuth_hat, azimuth).abs()
        corr_elevation = self.pearson_corr_coef(elevation_hat, elevation).abs()

        return {
            "loss_Th": loss_Th,
            "loss_Tt": loss_Tt,
            "loss_volume": loss_volume,
            "loss_dist_src": loss_dist_src,
            "loss_azimuth": loss_azimuth,
            "loss_elevation": loss_elevation,
            "corr_Th": corr_Th,
            "corr_Tt": corr_Tt,
            "corr_volume": corr_volume,
            "corr_dist_src": corr_dist_src,
            "corr_azimuth": corr_azimuth,
            "corr_elevation": corr_elevation,
        }
