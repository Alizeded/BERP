import einops
import torch
import torch.nn as nn
from torch.nn import L1Loss, SmoothL1Loss
from torchmetrics import PearsonCorrCoef


class RPMultiTaskLoss(nn.Module):

    def __init__(self, phase: str, unrelated_params: bool = False):
        super().__init__()
        self.smooth_l1_loss = SmoothL1Loss()
        self.l1_loss = L1Loss()
        self.pearson_corr_coef = PearsonCorrCoef()
        self.phase = phase
        self.unrelated_params = unrelated_params

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
        if self.unrelated_params:
            ts_hat = param_hat["ts_hat"]
            dist_src_hat = param_hat["dist_src_hat"]
        else:
            tr_hat = param_hat["tr_hat"]
            volume_hat = param_hat["volume_hat"]
        padding_mask = param_hat["padding_mask"]

        # extract groundtruth parameters
        # force cast to float to avoid mismatch in data type
        if self.unrelated_params:
            ts = param_groundtruth["ts"]
            dist_src = param_groundtruth["dist_src"]
        else:
            tr = param_groundtruth["tr"]
            volume = param_groundtruth["volume"]

        if self.unrelated_params:
            assert ts_hat.shape == dist_src_hat.shape
        else:
            assert tr_hat.shape == volume_hat.shape

        # ----------------- obtain padding mask ----------------- #
        if padding_mask is not None and padding_mask.any():
            # B x T
            input_lengths = (1 - padding_mask.long()).sum(-1)
            # apply conv formula to get real output_lengths
            output_lengths = self._get_param_pred_output_lengths(input_lengths)

            padding_mask = torch.zeros(
                ts_hat.shape[:2], dtype=ts_hat.dtype, device=ts_hat.device
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
                    if self.unrelated_params:
                        # repeat for other parameters without bias corrector
                        ts = einops.repeat(ts, "B -> B T", T=ts_hat.shape[1])
                        dist_src = einops.repeat(
                            dist_src, "B -> B T", T=dist_src_hat.shape[1]
                        )

                        # Apply padding mask to ts, volume
                        ts_hat_masked = ts_hat.masked_select(reverse_padding_mask)
                        dist_src_hat_masked = dist_src_hat.masked_select(
                            reverse_padding_mask
                        )

                        # Apply padding mask to repeated ts, volume
                        ts_masked = ts.masked_select(reverse_padding_mask)
                        dist_src_masked = dist_src.masked_select(reverse_padding_mask)

                        # Compute losses for room parametric predictor at training phase
                        loss_ts = self.smooth_l1_loss(ts_hat_masked, ts_masked)
                        loss_dist_src = self.smooth_l1_loss(
                            dist_src_hat_masked, dist_src_masked
                        )
                    else:
                        # repeat for other parameters without bias corrector
                        tr = einops.repeat(tr, "B -> B T", T=tr_hat.shape[1])
                        volume = einops.repeat(
                            volume, "B -> B T", T=volume_hat.shape[1]
                        )

                        # Apply padding mask to tr, volume
                        tr_hat_masked = tr_hat.masked_select(reverse_padding_mask)
                        volume_hat_masked = volume_hat.masked_select(
                            reverse_padding_mask
                        )

                        # Apply padding mask to repeated tr, volume
                        tr_masked = tr.masked_select(reverse_padding_mask)
                        volume_masked = volume.masked_select(reverse_padding_mask)

                        # Compute losses for room parametric predictor at training phase
                        loss_tr = self.smooth_l1_loss(tr_hat_masked, tr_masked)
                        loss_volume = self.smooth_l1_loss(
                            volume_hat_masked, volume_masked
                        )

                elif self.phase == "val":  #! for validation phase
                    # Collapse tr_hat, volume_hat to batch size 1D tensor
                    if self.unrelated_params:
                        ts_hat = (ts_hat * reverse_padding_mask).sum(dim=1) / (
                            reverse_padding_mask
                        ).sum(dim=1)

                        dist_src_hat = (dist_src_hat * reverse_padding_mask).sum(
                            dim=1
                        ) / (reverse_padding_mask).sum(dim=1)

                        # Compute losses for room parametric predictor at validation phase
                        loss_ts = self.l1_loss(ts_hat, ts)
                        loss_dist_src = self.l1_loss(dist_src_hat, dist_src)

                        # Compute correlation for room parametric predictor at validation phase
                        corr_ts = self.pearson_corr_coef(ts_hat, ts).abs()
                        corr_dist_src = self.pearson_corr_coef(
                            dist_src_hat, dist_src
                        ).abs()
                    else:
                        tr_hat = (tr_hat * reverse_padding_mask).sum(dim=1) / (
                            reverse_padding_mask
                        ).sum(dim=1)

                        volume_hat = (volume_hat * reverse_padding_mask).sum(dim=1) / (
                            reverse_padding_mask
                        ).sum(dim=1)

                        # Compute losses for room parametric predictor at validation phase
                        loss_tr = self.l1_loss(tr_hat, tr)
                        loss_volume = self.l1_loss(volume_hat, volume)

                        # Compute correlation for room parametric predictor at validation phase
                        corr_tr = self.pearson_corr_coef(tr_hat, tr).abs()
                        corr_volume = self.pearson_corr_coef(volume_hat, volume).abs()

            else:
                # ---------------------- no padding mask handling ---------------------- #
                if self.phase == "train":  #! for training phase
                    if self.unrelated_params:
                        # repeat for other parameters without bias corrector
                        ts = einops.repeat(ts, "B -> B T", T=ts_hat.shape[1])
                        dist_src = einops.repeat(
                            dist_src, "B -> B T", T=dist_src_hat.shape[1]
                        )

                        # Compute losses for room parametric predictor at training phase
                        loss_ts = self.smooth_l1_loss(ts_hat, ts)
                        loss_dist_src = self.smooth_l1_loss(dist_src_hat, dist_src)
                    else:
                        # repeat for other parameters without bias corrector
                        tr = einops.repeat(tr, "B -> B T", T=tr_hat.shape[1])
                        volume = einops.repeat(
                            volume, "B -> B T", T=volume_hat.shape[1]
                        )

                        # Compute losses for room parametric predictor at training phase
                        loss_tr = self.smooth_l1_loss(tr_hat, tr)
                        loss_volume = self.smooth_l1_loss(volume_hat, volume)

                elif self.phase == "val":  #! for validation phase
                    if self.unrelated_params:
                        # Collapse ts_hat, volume_hat to batch size 1D tensor
                        ts_hat = ts_hat.mean(dim=1)
                        dist_src_hat = dist_src_hat.mean(dim=1)

                        # Compute losses for room parametric predictor at validation phase
                        loss_ts = self.l1_loss(ts_hat, ts)
                        loss_dist_src = self.l1_loss(dist_src_hat, dist_src)

                        # Compute correlation for room parametric predictor at validation phase
                        corr_ts = self.pearson_corr_coef(ts_hat, ts).abs()
                        corr_dist_src = self.pearson_corr_coef(
                            dist_src_hat, dist_src
                        ).abs()
                    else:

                        # Collapse tr_hat, volume_hat to batch size 1D tensor
                        tr_hat = tr_hat.mean(dim=1)
                        volume_hat = volume_hat.mean(dim=1)

                        # Compute losses for room parametric predictor at validation phase
                        loss_tr = self.l1_loss(tr_hat, tr)
                        loss_volume = self.l1_loss(volume_hat, volume)

                        # Compute correlation for room parametric predictor at validation phase
                        corr_tr = self.pearson_corr_coef(tr_hat, tr).abs()
                        corr_volume = self.pearson_corr_coef(volume_hat, volume).abs()

        # ------------------------- return losses and metrics ------------------------- #
        if self.phase == "train":
            if self.unrelated_params:
                return {
                    "loss_ts": loss_ts,
                    "loss_dist_src": loss_dist_src,
                }
            else:
                return {
                    "loss_tr": loss_tr,
                    "loss_volume": loss_volume,
                }

        elif self.phase == "val":
            if self.unrelated_params:
                return {
                    "loss_ts": loss_ts,
                    "loss_dist_src": loss_dist_src,
                    "corr_ts": corr_ts,
                    "corr_dist_src": corr_dist_src,
                }
            else:
                return {
                    "loss_tr": loss_tr,
                    "loss_volume": loss_volume,
                    "corr_tr": corr_tr,
                    "corr_volume": corr_volume,
                }
