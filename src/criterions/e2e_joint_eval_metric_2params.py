import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import L1Loss, SmoothL1Loss
from torchmetrics import PearsonCorrCoef


class RPJointEstimationEvaluation(nn.Module):

    def __init__(
        self,
        unrelated_params: bool = False,
    ):
        super().__init__()

        self.criterion = SmoothL1Loss()
        self.l1_loss = L1Loss()
        self.pearson_corr_coef = PearsonCorrCoef()
        self.unrelated_params = unrelated_params

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

        #! ------------------- compute evaluation metrics ------------------- #
        # ------------------------- polynomial loss ------------------------- #
        # collapse all the predicted parameters to batch size 1D tensor
        if self.unrelated_params:
            if ts_hat.dim() == 2:
                if padding_mask is not None and padding_mask.any():
                    # ---------------------- padding mask handling ---------------------- #
                    ts_hat = (ts_hat * reverse_padding_mask).sum(
                        dim=1
                    ) / reverse_padding_mask.sum(dim=1)

                    dist_src_hat = (dist_src_hat * reverse_padding_mask).sum(
                        dim=1
                    ) / reverse_padding_mask.sum(dim=1)

                else:
                    # ---------------------- no padding mask handling ---------------------- #
                    ts_hat = ts_hat.mean(dim=1)
                    dist_src_hat = dist_src_hat.mean(dim=1)

            elif ts_hat.dim() == 1:
                ts_hat = ts_hat.mean()
                dist_src_hat = dist_src_hat.mean
        else:
            if tr_hat.dim() == 2:
                if padding_mask is not None and padding_mask.any():
                    # ---------------------- padding mask handling ---------------------- #s

                    tr_hat = (tr_hat * reverse_padding_mask).sum(
                        dim=1
                    ) / reverse_padding_mask.sum(dim=1)

                    volume_hat = (volume_hat * reverse_padding_mask).sum(
                        dim=1
                    ) / reverse_padding_mask.sum(dim=1)

                else:
                    # ---------------------- no padding mask handling ---------------------- #
                    tr_hat = tr_hat.mean(dim=1)
                    volume_hat = volume_hat.mean(dim=1)

            elif tr_hat.dim() == 1:
                tr_hat = tr_hat.mean()
                volume_hat = volume_hat.mean()

        # ------------------- compute evaluation metrics -------------------

        # MAE metric for all the predicted parameters
        if self.unrelated_params:
            loss_ts = self.l1_loss(ts_hat, ts)
            loss_dist_src = self.l1_loss(dist_src_hat, dist_src)
        else:
            loss_tr = self.l1_loss(tr_hat, tr)
            loss_volume = self.l1_loss(volume_hat, volume)

        # Pearson correlation coefficient for all the predicted parameters
        if self.unrelated_params:
            corr_ts = self.pearson_corr_coef(ts_hat, ts).abs()
            corr_dist_src = self.pearson_corr_coef(dist_src_hat, dist_src).abs()
        else:
            corr_tr = self.pearson_corr_coef(tr_hat, tr).abs()
            corr_volume = self.pearson_corr_coef(volume_hat, volume).abs()

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
