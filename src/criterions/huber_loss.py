from typing import Dict

import torch
import torch.nn as nn

from torch.nn import SmoothL1Loss, L1Loss
from torchmetrics import PearsonCorrCoef

import einops

from src.utils.unitary_linear_norm import unitary_norm_inv


class HuberLoss(nn.Module):

    def __init__(
        self,
        phase: str,
        module: str,
        ablation: bool = False,
    ):
        super(HuberLoss, self).__init__()
        self.phase = phase  # train or val or test
        self.module = module  # rir, volume, distance
        self.ablation = ablation
        self.huber_loss = SmoothL1Loss()
        self.l1_loss = L1Loss()
        self.pearson_corr = PearsonCorrCoef()

        assert (
            self.phase == "train"
            or self.phase == "val"
            or self.phase == "test"
            or self.phase == "infer"
        ), "phase should be either 'train' or 'val' or 'test' or 'infer'"

        assert (
            self.module == "rir"
            or self.module == "volume"
            or self.module == "distance"
            or self.module == "orientation"
        ), "module should be either rir or volume or distance or orientation"

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

    def _re_padding_mask(self, pred: torch.Tensor, padding_mask: torch.Tensor):
        """Re padding mask after convolutional layers"""
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

        return padding_mask, reverse_padding_mask

    def huber_loss_calculate(
        self, pred: torch.Tensor, target: torch.Tensor, padding_mask: torch.Tensor
    ):

        if self.ablation is False:  # repadding mask for no ablation
            if padding_mask is not None and padding_mask.any():

                padding_mask, reverse_padding_mask = self._re_padding_mask(
                    pred=pred, padding_mask=padding_mask
                )
        else:
            if padding_mask is not None and padding_mask.any():
                reverse_padding_mask = padding_mask.logical_not()

        if padding_mask is not None and padding_mask.any():
            # -------------------- padding mask handling --------------------
            if self.phase == "train":
                # repeat the target values to match the pred values
                target = einops.repeat(target, "B -> B T", T=pred.shape[1])

                # Apply padding mask to the pred and target values
                pred = pred.masked_select(reverse_padding_mask)
                target = target.masked_select(reverse_padding_mask)

                # Calculate the loss
                loss = self.huber_loss(pred, target)

            elif self.phase == "val" or self.phase == "test":
                # Collapse the pred value to match the target value
                pred = (pred * reverse_padding_mask).sum(
                    dim=1
                ) / reverse_padding_mask.sum(dim=1)

                # Calculate the loss
                loss = self.l1_loss(pred, target)

                # Compute the Pearson Correlation Coefficient
                corr_coef = self.pearson_corr(pred, target).abs()

        else:
            # -------------------- no padding mask handling --------------------
            if self.phase == "train":
                # repeat the target values to match the pred values
                target = einops.repeat(target, "B -> B T", T=pred.shape[1])

                # Calculate the loss
                loss = self.huber_loss(pred, target)

            elif self.phase == "val" or self.phase == "test":
                # Collapse the pred value to match the target value
                pred = pred.mean(dim=1)

                # Calculate the loss
                loss = self.l1_loss(pred, target)

                # Compute the Pearson Correlation Coefficient
                corr_coef = self.pearson_corr(pred, target).abs()

        if self.phase == "train":
            return {"loss": loss}

        elif self.phase == "val" or self.phase == "test":
            return {"loss": loss, "corr_coef": corr_coef}

        else:
            NotImplementedError(
                "Phase not implemented, please choose either 'train' or 'val' or 'test'"
            )

    def forward(
        self,
        param_hat: Dict[str, torch.Tensor],
        param_groundtruth: Dict[str, torch.Tensor],
    ):

        if self.module == "rir":
            Th_hat, Tt_hat = param_hat["Th_hat"], param_hat["Tt_hat"]
            Th, Tt = param_groundtruth["Th"], param_groundtruth["Tt"]
            padding_mask = param_hat["padding_mask"]

            if self.phase == "train":
                # Calculate the loss
                loss_Th = self.huber_loss_calculate(Th_hat, Th, padding_mask)
                loss_Tt = self.huber_loss_calculate(Tt_hat, Tt, padding_mask)

                return {
                    "loss_Th": loss_Th["loss"],
                    "loss_Tt": loss_Tt["loss"],
                }

            elif self.phase == "val":
                # Calculate the loss and correlation coefficient
                loss_Th = self.huber_loss_calculate(Th_hat, Th, padding_mask)
                loss_Tt = self.huber_loss_calculate(Tt_hat, Tt, padding_mask)

                return {
                    "loss_Th": loss_Th["loss"],
                    "loss_Tt": loss_Tt["loss"],
                    "corr_coef_Th": loss_Th["corr_coef"],
                    "corr_coef_Tt": loss_Tt["corr_coef"],
                }

            elif self.phase == "test":

                if self.ablation is False:  # repadding mask for no ablation
                    if padding_mask is not None and padding_mask.any():
                        padding_mask, reverse_padding_mask = self._re_padding_mask(
                            pred=Th_hat, padding_mask=padding_mask
                        )
                else:
                    if padding_mask is not None and padding_mask.any():
                        reverse_padding_mask = padding_mask.logical_not()

                if padding_mask is not None and padding_mask.any():
                    # Collapse the pred value to match the target value
                    Th_hat = (Th_hat * reverse_padding_mask).sum(
                        dim=1
                    ) / reverse_padding_mask.sum(dim=1)

                else:
                    Th_hat = Th_hat.mean(dim=1)

                # inverse unitary normalization
                Th_hat = unitary_norm_inv(Th_hat, lb=0.005, ub=0.276)
                Th = unitary_norm_inv(Th, lb=0.005, ub=0.276)

                loss_Th = self.l1_loss(Th_hat, Th)
                corr_coef_Th = self.pearson_corr(Th_hat, Th).abs()

                loss_Tt = self.huber_loss_calculate(Tt_hat, Tt, padding_mask)

                return {
                    "loss_Th": loss_Th,
                    "loss_Tt": loss_Tt["loss"],
                    "corr_coef_Th": corr_coef_Th,
                    "corr_coef_Tt": loss_Tt["corr_coef"],
                }

            else:
                NotImplementedError(
                    "Phase not implemented, please choose either train or val or test"
                )

        elif self.module == "volume":
            volume_hat = param_hat["volume_hat"]
            volume = param_groundtruth["volume"]
            padding_mask = param_hat["padding_mask"]

            if self.phase == "train":
                # Calculate the loss
                loss = self.huber_loss_calculate(volume_hat, volume, padding_mask)

                return {"loss_volume": loss["loss"]}

            elif self.phase == "val":
                # Calculate the loss
                loss = self.huber_loss_calculate(volume_hat, volume, padding_mask)

                return {
                    "loss_volume": loss["loss"],
                    "corr_coef_volume": loss["corr_coef"],
                }

            elif self.phase == "test":

                if self.ablation is False:  # no ablation for repadding mask
                    if padding_mask is not None and padding_mask.any():
                        padding_mask, reverse_padding_mask = self._re_padding_mask(
                            pred=volume_hat, padding_mask=padding_mask
                        )
                else:
                    if padding_mask is not None and padding_mask.any():
                        reverse_padding_mask = padding_mask.logical_not()

                if padding_mask is not None and padding_mask.any():
                    # Collapse the pred value to match the target value
                    volume_hat = (volume_hat * reverse_padding_mask).sum(
                        dim=1
                    ) / reverse_padding_mask.sum(dim=1)

                else:
                    volume_hat = volume_hat.mean(dim=1)

                # inverse unitary normalization
                volume_hat = unitary_norm_inv(volume_hat, lb=1.5051, ub=3.9542)
                volume = unitary_norm_inv(volume, lb=1.5051, ub=3.9542)

                loss_volume = self.l1_loss(volume_hat, volume)

                corr_coef_volume = self.pearson_corr(volume_hat, volume).abs()

                return {
                    "loss_volume": loss_volume,
                    "corr_coef_volume": corr_coef_volume,
                }

            else:
                NotImplementedError(
                    "Phase not implemented, please choose either train or val or test"
                )

        elif self.module == "distance":
            dist_src_hat = param_hat["dist_src_hat"]
            dist_src = param_groundtruth["dist_src"]
            padding_mask = param_hat["padding_mask"]

            if self.phase == "train":
                # Calculate the loss
                loss = self.huber_loss_calculate(dist_src_hat, dist_src, padding_mask)

                return {"loss_dist_src": loss["loss"]}

            elif self.phase == "val" or self.phase == "test":
                # Calculate the loss
                loss = self.huber_loss_calculate(dist_src_hat, dist_src, padding_mask)

                return {
                    "loss_dist_src": loss["loss"],
                    "corr_coef_dist_src": loss["corr_coef"],
                }

            else:
                NotImplementedError(
                    "Phase not implemented, please choose either train or val or test"
                )

        elif self.module == "orientation":
            azimuth_hat, elevation_hat = (
                param_hat["azimuth_hat"],
                param_hat["elevation_hat"],
            )
            azimuth, elevation = (
                param_groundtruth["azimuth"],
                param_groundtruth["elevation"],
            )
            padding_mask = param_hat["padding_mask"]

            if self.phase == "train":
                # Calculate the loss
                loss_azimuth = self.huber_loss_calculate(
                    azimuth_hat, azimuth, padding_mask
                )
                loss_elevation = self.huber_loss_calculate(
                    elevation_hat, elevation, padding_mask
                )

                return {
                    "loss_azimuth": loss_azimuth["loss"],
                    "loss_elevation": loss_elevation["loss"],
                }

            elif self.phase == "val":
                # Calculate the loss
                loss_azimuth = self.huber_loss_calculate(
                    azimuth_hat, azimuth, padding_mask
                )
                loss_elevation = self.huber_loss_calculate(
                    elevation_hat, elevation, padding_mask
                )

                return {
                    "loss_azimuth": loss_azimuth["loss"],
                    "loss_elevation": loss_elevation["loss"],
                    "corr_coef_azimuth": loss_azimuth["corr_coef"],
                    "corr_coef_elevation": loss_elevation["corr_coef"],
                }

            elif self.phase == "test":

                if self.ablation is False:
                    if padding_mask is not None and padding_mask.any():
                        padding_mask, reverse_padding_mask = self._re_padding_mask(
                            pred=azimuth_hat, padding_mask=padding_mask
                        )

                else:
                    if padding_mask is not None and padding_mask.any():
                        reverse_padding_mask = padding_mask.logical_not()

                # Collapse the pred value to match the target value
                if padding_mask is not None and padding_mask.any():
                    # -------------------- padding mask handling --------------------
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
                azimuth_hat = unitary_norm_inv(azimuth_hat, lb=-1.000, ub=1.000)
                azimuth = unitary_norm_inv(azimuth, lb=-1.000, ub=1.000)
                elevation_hat = elevation_hat = unitary_norm_inv(
                    elevation_hat, lb=-0.733, ub=0.486
                )
                elevation = unitary_norm_inv(elevation, lb=-0.733, ub=0.486)

                loss_azimuth = self.l1_loss(azimuth_hat, azimuth)
                corr_coef_azimuth = self.pearson_corr(azimuth_hat, azimuth).abs()

                loss_elevation = self.l1_loss(elevation_hat, elevation)
                corr_coef_elevation = self.pearson_corr(elevation_hat, elevation).abs()

                return {
                    "loss_azimuth": loss_azimuth,
                    "loss_elevation": loss_elevation,
                    "corr_coef_azimuth": corr_coef_azimuth,
                    "corr_coef_elevation": corr_coef_elevation,
                }
