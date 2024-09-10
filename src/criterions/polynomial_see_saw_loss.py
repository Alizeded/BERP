from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import BCEWithLogitsLoss, SmoothL1Loss, L1Loss

from torchmetrics import PearsonCorrCoef


import einops


class PolynomialSeeSawLoss(nn.Module):
    def __init__(self, phase: str):
        super(PolynomialSeeSawLoss, self).__init__()
        self.bce_loss = BCEWithLogitsLoss()
        self.smooth_l1_loss = SmoothL1Loss()
        self.l1_loss = L1Loss()
        self.pearson_corr_coef = PearsonCorrCoef()
        self.phase = phase

        assert self.phase in {
            "train",
            "val",
        }, f"Invalid phase: {self.phase}, should be either 'train' or 'val'"

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
        azimuth_label = param_groundtruth["azimuth_classif"]
        elevation_label = param_groundtruth["elevation_classif"]

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

        #! ------------------------- compute loss for training and validation phase ------------------------- #
        if self.phase in ["train", "val"]:
            # -------------- BCE loss for bias corrector -------------- #

            # convert azimuth_label and elevation_label to float to calculate BCELoss
            azimuth_label = azimuth_label.long().float().to(judge_logits_azimuth.device)
            elevation_label = (
                elevation_label.long().float().to(judge_logits_elevation.device)
            )

            # calculate BCE loss for azimuth and elevation for bias corrector
            bce_loss_azimuth = self.bce_loss(judge_logits_azimuth, azimuth_label)
            bce_loss_elevation = self.bce_loss(judge_logits_elevation, elevation_label)

            judge_prob_azimuth = F.sigmoid(judge_logits_azimuth)
            judge_prob_elevation = F.sigmoid(judge_logits_elevation)

            # ------------------------- see-saw loss ------------------------- #
            loss_ori_src = dict(
                loss_azimuth_bc=[],  # azimuth loss for bias corrector
                loss_elevation_bc=[],  # elevation loss for bias corrector
                loss_azimuth_pp=[],  # azimuth loss for room parametric predictor
                loss_elevation_pp=[],  # elevation loss for room parametric predictor
            )

            if self.phase == "val":
                corr_ori_src_val = dict(
                    corr_azimuth_pp_val=[],  # azimuth correlation for room parametric predictor
                    corr_elevation_pp_val=[],  # elevation correlation for room parametric predictor
                )

            loss_ori_src["loss_azimuth_bc"].append(bce_loss_azimuth)
            loss_ori_src["loss_elevation_bc"].append(bce_loss_elevation)

            # ------------------- azimuth loss ------------------- #
            if self.phase == "train":
                # threshold = 0.4 to input to room parametric predictor for training phase
                idx_azimuth_pp = torch.where(judge_prob_azimuth >= 0.4)[0]
            elif self.phase == "val":
                # threshold = 0.5 to input to room parametric predictor for validation phase
                idx_azimuth_pp = torch.where(judge_prob_azimuth >= 0.5)[0]
            else:
                raise ValueError(
                    f"Invalid phase: {self.phase}, should be either 'train' or 'val'"
                )

            if len(idx_azimuth_pp) > 0:
                azimuth_hat = azimuth_hat[idx_azimuth_pp, ...]
                azimuth = azimuth[idx_azimuth_pp]

                if padding_mask is not None and padding_mask.any():
                    reverse_padding_mask_azimuth = reverse_padding_mask[
                        idx_azimuth_pp, ...
                    ]

                    # ---------------------- padding mask handling ---------------------- #
                    if self.phase == "train":
                        # Repeat azimuth and elevation to match the shape of azimuth_hat
                        azimuth = einops.repeat(
                            azimuth, "B -> B T", T=azimuth_hat.shape[1]
                        )

                        # Apply padding mask to azimuth_hat
                        azimuth_hat_masked = azimuth_hat.masked_select(
                            reverse_padding_mask_azimuth
                        )

                        # Apply padding mask to repeated azimuth
                        azimuth_masked = azimuth.masked_select(
                            reverse_padding_mask_azimuth
                        )

                        # Compute azimuth losses for room parametric predictor at training phase
                        loss_azimuth_pp = self.smooth_l1_loss(
                            azimuth_hat_masked, azimuth_masked
                        )
                    elif self.phase == "val":
                        # Collapse azimuth_hat to batch size 1D tensor
                        azimuth_hat_masked = (
                            azimuth_hat * reverse_padding_mask_azimuth
                        ).sum(dim=1) / (reverse_padding_mask_azimuth).sum(dim=1)

                        # Compute azimuth losses for room parametric predictor at validation phase
                        loss_azimuth_pp = self.l1_loss(azimuth_hat_masked, azimuth)

                        # Compute azimuth correlation for room parametric predictor at validation phase
                        corr_azimuth_pp = self.pearson_corr_coef(
                            azimuth_hat_masked, azimuth
                        ).abs()

                else:
                    # ---------------------- no padding mask handling ---------------------- #
                    if self.phase == "train":  #! for training phase
                        # azimuth loss for room parametric predictor
                        azimuth = einops.repeat(
                            azimuth, "B -> B T", T=azimuth_hat.shape[1]
                        )

                        # Compute azimuth losses for room parametric predictor at training phase
                        loss_azimuth_pp = self.smooth_l1_loss(azimuth_hat, azimuth)
                    elif self.phase == "val":  #! for validation phase
                        # Collapse azimuth_hat to batch size 1D tensor
                        azimuth_hat = azimuth_hat.mean(dim=1)

                        # Compute azimuth losses for room parametric predictor at validation phase
                        loss_azimuth_pp = self.l1_loss(azimuth_hat, azimuth)

                        # Compute azimuth correlation for room parametric predictor at validation phase
                        corr_azimuth_pp = self.pearson_corr_coef(
                            azimuth_hat, azimuth
                        ).abs()

                loss_ori_src["loss_azimuth_pp"].append(loss_azimuth_pp)

                if self.phase == "val":
                    corr_ori_src_val["corr_azimuth_pp_val"].append(corr_azimuth_pp)

            # ------------------- elevation loss ------------------- #
            if self.phase == "train":  #! for training phase
                # threshold = 0.4 to input to room parametric predictor
                idx_elevation_pp = torch.where(judge_prob_elevation >= 0.4)[0]
            elif self.phase == "val":  #! for validation phase
                # threshold = 0.5 to input to room parametric predictor
                idx_elevation_pp = torch.where(judge_prob_elevation >= 0.5)[0]
            else:
                raise ValueError(
                    f"Invalid phase: {self.phase}, should be either 'train' or 'val'"
                )

            if len(idx_elevation_pp) > 0:
                elevation_hat = elevation_hat[idx_elevation_pp, ...]
                elevation = elevation[idx_elevation_pp]

                if padding_mask is not None and padding_mask.any():
                    reverse_padding_mask_elevation = reverse_padding_mask[
                        idx_elevation_pp, ...
                    ]

                    # ---------------------- padding mask handling ---------------------- #
                    if self.phase == "train":  #! for training phase
                        # Repeat elevation and azimuth to match the shape of elevation_hat
                        elevation = einops.repeat(
                            elevation, "B -> B T", T=elevation_hat.shape[1]
                        )

                        # Apply padding mask to elevation_hat
                        elevation_hat_masked = elevation_hat.masked_select(
                            reverse_padding_mask_elevation
                        )

                        # Apply padding mask to repeated elevation
                        elevation_masked = elevation.masked_select(
                            reverse_padding_mask_elevation
                        )

                        # Compute elevation losses for room parametric predictor
                        loss_elevation_pp = self.smooth_l1_loss(
                            elevation_hat_masked, elevation_masked
                        )

                    elif self.phase == "val":  #! for validation phase
                        # Collapse elevation_hat to batch size 1D tensor
                        elevation_hat_masked = (
                            elevation_hat * reverse_padding_mask_elevation
                        ).sum(dim=1) / (reverse_padding_mask_elevation).sum(dim=1)

                        # Compute elevation losses for room parametric predictor
                        loss_elevation_pp = self.l1_loss(
                            elevation_hat_masked, elevation
                        )

                        corr_elevation_pp = self.pearson_corr_coef(
                            elevation_hat_masked, elevation
                        ).abs()

                else:
                    # ---------------------- no padding mask handling ---------------------- #
                    if self.phase == "train":  #! for training phase
                        # elevation loss for room parametric predictor
                        elevation = einops.repeat(
                            elevation, "B -> B T", T=elevation_hat.shape[1]
                        )
                        loss_elevation_pp = self.smooth_l1_loss(
                            elevation_hat, elevation
                        )
                    elif self.phase == "val":  #! for validation phase
                        # Collapse elevation_hat to batch size 1D tensor
                        elevation_hat = elevation_hat.mean(dim=1)

                        # Compute elevation losses for room parametric predictor
                        loss_elevation_pp = self.l1_loss(elevation_hat, elevation)

                        # Compute elevation correlation for room parametric predictor at validation phase
                        corr_elevation_pp = self.pearson_corr_coef(
                            elevation_hat, elevation
                        ).abs()

                loss_ori_src["loss_elevation_pp"].append(loss_elevation_pp)

                if self.phase == "val":
                    corr_ori_src_val["corr_elevation_pp_val"].append(corr_elevation_pp)

            pred_azimuth_label = torch.where(judge_prob_azimuth > 0.5, 1, 0)
            pred_elevation_label = torch.where(judge_prob_elevation > 0.5, 1, 0)

            pred_label_ori_src = {
                "pred_azimuth_label": pred_azimuth_label,
                "pred_elevation_label": pred_elevation_label,
            }

            # ------------------------- polynomial loss ------------------------- #
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
                "loss_ori_src": loss_ori_src,
                "pred_label_ori_src": pred_label_ori_src,
            }

        elif self.phase == "val":
            return {
                "loss_Th": loss_Th,
                "loss_Tt": loss_Tt,
                "loss_volume": loss_volume,
                "loss_dist_src": loss_dist_src,
                "loss_ori_src": loss_ori_src,
                "pred_label_ori_src": pred_label_ori_src,
                "corr_Th": corr_Th,
                "corr_Tt": corr_Tt,
                "corr_volume": corr_volume,
                "corr_dist_src": corr_dist_src,
                "corr_ori_src": corr_ori_src_val,
            }
