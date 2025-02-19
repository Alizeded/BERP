import einops
import torch
import torch.nn as nn
from torch.nn import L1Loss, SmoothL1Loss
from torchmetrics import PearsonCorrCoef

from src.utils.unitary_linear_norm import unitary_norm_inv


class MultiTaskLoss(nn.Module):

    def __init__(self, phase: str):
        super().__init__()
        self.smooth_l1_loss = SmoothL1Loss()
        self.l1_loss = L1Loss()
        self.pearson_corr_coef = PearsonCorrCoef()
        self.phase = phase

        assert (
            self.phase == "train" or self.phase == "val"
        ), f"Invalid phase: {self.phase}, should be either 'train' or 'val'"

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
        param_hat: dict[str, torch.Tensor],
        param_groundtruth: dict[str, torch.Tensor],
    ):
        # extract predicted parameters
        # force cast to float to avoid mismatch in data type
        Th_hat = param_hat["Th_hat"]
        Tt_hat = param_hat["Tt_hat"]
        volume_hat = param_hat["volume_hat"]
        dist_src_hat = param_hat["dist_src_hat"]
        padding_mask = param_hat["padding_mask"]

        # extract groundtruth parameters
        # force cast to float to avoid mismatch in data type
        Th = param_groundtruth["Th"]
        Tt = param_groundtruth["Tt"]
        volume = param_groundtruth["volume"]
        dist_src = param_groundtruth["dist_src"]

        assert Th_hat.shape == Tt_hat.shape == volume_hat.shape == dist_src_hat.shape

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

        #! ------------------------- compute loss for training and validation phase ------------------------- #
        if self.phase == "train" or self.phase == "val":

            # ------------------------- mullti-task loss ------------------------- #
            if padding_mask is not None and padding_mask.any():
                # ---------------------- padding mask handling ---------------------- #
                if self.phase == "train":  #! for training phase
                    # repeat for other parameters without bias corrector
                    Th = einops.repeat(Th, "B -> B T", T=Th_hat.shape[1])
                    Tt = einops.repeat(Tt, "B -> B T", T=Tt_hat.shape[1])
                    volume = einops.repeat(volume, "B -> B T", T=volume_hat.shape[1])
                    dist_src = einops.repeat(
                        dist_src, "B -> B T", T=dist_src_hat.shape[1]
                    )

                    # Apply padding mask to Th_hat
                    Th_hat_masked = Th_hat.masked_select(reverse_padding_mask)
                    Tt_hat_masked = Tt_hat.masked_select(reverse_padding_mask)
                    volume_hat_masked = volume_hat.masked_select(reverse_padding_mask)
                    dist_src_hat_masked = dist_src_hat.masked_select(
                        reverse_padding_mask
                    )

                    # Apply padding mask to repeated Th, Tt, volume, dist_src
                    Th_masked = Th.masked_select(reverse_padding_mask)
                    Tt_masked = Tt.masked_select(reverse_padding_mask)
                    volume_masked = volume.masked_select(reverse_padding_mask)
                    dist_src_masked = dist_src.masked_select(reverse_padding_mask)

                    # Compute losses for room parametric predictor at training phase
                    loss_Th = self.smooth_l1_loss(Th_hat_masked, Th_masked)
                    loss_Tt = self.smooth_l1_loss(Tt_hat_masked, Tt_masked)
                    loss_volume = self.smooth_l1_loss(volume_hat_masked, volume_masked)
                    loss_dist_src = self.smooth_l1_loss(
                        dist_src_hat_masked, dist_src_masked
                    )
                elif self.phase == "val":  #! for validation phase
                    # Collapse Th_hat, Tt_hat, volume_hat, dist_src_hat to batch size 1D tensor
                    Th_hat = (Th_hat * reverse_padding_mask).sum(dim=1) / (
                        reverse_padding_mask
                    ).sum(dim=1)

                    Tt_hat = (Tt_hat * reverse_padding_mask).sum(dim=1) / (
                        reverse_padding_mask
                    ).sum(dim=1)

                    volume_hat = (volume_hat * reverse_padding_mask).sum(
                        dim=1
                    ) / reverse_padding_mask.sum(dim=1)

                    dist_src_hat = (dist_src_hat * reverse_padding_mask).sum(
                        dim=1
                    ) / reverse_padding_mask.sum(dim=1)

                    # Compute losses for room parametric predictor at validation phase
                    loss_Th = self.l1_loss(Th_hat, Th)
                    loss_Tt = self.l1_loss(Tt_hat, Tt)
                    loss_volume = self.l1_loss(volume_hat, volume)
                    loss_dist_src = self.l1_loss(dist_src_hat, dist_src)

                    # Compute correlation for room parametric predictor at validation phase
                    corr_Th = self.pearson_corr_coef(Th_hat, Th).abs()
                    corr_Tt = self.pearson_corr_coef(Tt_hat, Tt).abs()
                    corr_volume = self.pearson_corr_coef(volume_hat, volume).abs()
                    corr_dist_src = self.pearson_corr_coef(dist_src_hat, dist_src).abs()

            else:
                # ---------------------- no padding mask handling ---------------------- #
                if self.phase == "train":  #! for training phase
                    # repeat for other parameters without bias corrector
                    Th = einops.repeat(Th, "B -> B T", T=Th_hat.shape[1])
                    Tt = einops.repeat(Tt, "B -> B T", T=Tt_hat.shape[1])
                    volume = einops.repeat(volume, "B -> B T", T=volume_hat.shape[1])
                    dist_src = einops.repeat(
                        dist_src, "B -> B T", T=dist_src_hat.shape[1]
                    )

                    # Compute losses for room parametric predictor at training phase
                    loss_Th = self.smooth_l1_loss(Th_hat, Th)
                    loss_Tt = self.smooth_l1_loss(Tt_hat, Tt)
                    loss_volume = self.smooth_l1_loss(volume_hat, volume)
                    loss_dist_src = self.smooth_l1_loss(dist_src_hat, dist_src)

                elif self.phase == "val":  #! for validation phase
                    # Collapse Th_hat, Tt_hat, volume_hat, dist_src_hat to batch size 1D tensor
                    Th_hat = Th_hat.mean(dim=1)
                    Tt_hat = Tt_hat.mean(dim=1)
                    volume_hat = volume_hat.mean(dim=1)
                    dist_src_hat = dist_src_hat.mean(dim=1)

                    # Compute losses for room parametric predictor at validation phase
                    loss_Th = self.l1_loss(Th_hat, Th)
                    loss_Tt = self.l1_loss(Tt_hat, Tt)
                    loss_volume = self.l1_loss(volume_hat, volume)
                    loss_dist_src = self.l1_loss(dist_src_hat, dist_src)

                    # Compute correlation for room parametric predictor at validation phase
                    corr_Th = self.pearson_corr_coef(Th_hat, Th).abs()
                    corr_Tt = self.pearson_corr_coef(Tt_hat, Tt).abs()
                    corr_volume = self.pearson_corr_coef(volume_hat, volume).abs()
                    corr_dist_src = self.pearson_corr_coef(dist_src_hat, dist_src).abs()

        # ------------------------- return losses and metrics ------------------------- #
        if self.phase == "train":
            return {
                "loss_Th": loss_Th,
                "loss_Tt": loss_Tt,
                "loss_volume": loss_volume,
                "loss_dist_src": loss_dist_src,
            }

        elif self.phase == "val":
            return {
                "loss_Th": loss_Th,
                "loss_Tt": loss_Tt,
                "loss_volume": loss_volume,
                "loss_dist_src": loss_dist_src,
                "corr_Th": corr_Th,
                "corr_Tt": corr_Tt,
                "corr_volume": corr_volume,
                "corr_dist_src": corr_dist_src,
            }
