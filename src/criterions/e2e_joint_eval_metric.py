import torch
import torch.nn as nn
from torch.nn import L1Loss, SmoothL1Loss
from torchmetrics import PearsonCorrCoef

from src.utils.unitary_linear_norm import unitary_norm_inv


class RPJointEstimationEvaluation(nn.Module):

    def __init__(
        self,
    ):
        super().__init__()

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
        param_hat: dict[str, torch.Tensor],
        param_groundtruth: dict[str, torch.Tensor],
    ):

        # extract predicted parameters
        # force cast to float to avoid mismatch in data type
        sti_hat = param_hat["sti_hat"]
        alcons_hat = param_hat["alcons_hat"]
        tr_hat = param_hat["tr_hat"]
        edt_hat = param_hat["edt_hat"]
        c80_hat = param_hat["c80_hat"]
        c50_hat = param_hat["c50_hat"]
        d50_hat = param_hat["d50_hat"]
        ts_hat = param_hat["ts_hat"]
        volume_hat = param_hat["volume_hat"]
        dist_src_hat = param_hat["dist_src_hat"]
        padding_mask = param_hat["padding_mask"]

        # extract groundtruth parameters
        # force cast to float to avoid mismatch in data type
        sti = param_groundtruth["sti"]
        alcons = param_groundtruth["alcons"]
        tr = param_groundtruth["tr"]
        edt = param_groundtruth["edt"]
        c80 = param_groundtruth["c80"]
        c50 = param_groundtruth["c50"]
        d50 = param_groundtruth["d50"]
        ts = param_groundtruth["ts"]
        volume = param_groundtruth["volume"]
        dist_src = param_groundtruth["dist_src"]

        assert (
            sti_hat.shape
            == alcons_hat.shape
            == tr_hat.shape
            == edt_hat.shape
            == c80_hat.shape
            == c50_hat.shape
            == d50_hat.shape
            == ts_hat.shape
            == volume_hat.shape
            == dist_src_hat.shape
        )

        # ----------------- obtain padding mask ----------------- #
        if padding_mask is not None and padding_mask.any():
            # B x T
            input_lengths = (1 - padding_mask.long()).sum(-1)
            # apply conv formula to get real output_lengths
            output_lengths = self._get_param_pred_output_lengths(input_lengths)

            padding_mask = torch.zeros(
                tr_hat.shape[:2], dtype=tr_hat.dtype, device=tr_hat.device
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
        if sti_hat.dim() == 2:
            if padding_mask is not None and padding_mask.any():
                # ---------------------- padding mask handling ---------------------- #s

                sti_hat = (sti_hat * reverse_padding_mask).sum(
                    dim=1
                ) / reverse_padding_mask.sum(dim=1)

                alcons_hat = (alcons_hat * reverse_padding_mask).sum(
                    dim=1
                ) / reverse_padding_mask.sum(dim=1)

                tr_hat = (tr_hat * reverse_padding_mask).sum(
                    dim=1
                ) / reverse_padding_mask.sum(dim=1)

                edt_hat = (edt_hat * reverse_padding_mask).sum(
                    dim=1
                ) / reverse_padding_mask.sum(dim=1)

                c80_hat = (c80_hat * reverse_padding_mask).sum(
                    dim=1
                ) / reverse_padding_mask.sum(dim=1)

                c50_hat = (c50_hat * reverse_padding_mask).sum(
                    dim=1
                ) / reverse_padding_mask.sum(dim=1)

                d50_hat = (d50_hat * reverse_padding_mask).sum(
                    dim=1
                ) / reverse_padding_mask.sum(dim=1)

                ts_hat = (ts_hat * reverse_padding_mask).sum(
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
                sti_hat = sti_hat.mean(dim=1)
                alcons_hat = alcons_hat.mean(dim=1)
                tr_hat = tr_hat.mean(dim=1)
                edt_hat = edt_hat.mean(dim=1)
                c80_hat = c80_hat.mean(dim=1)
                c50_hat = c50_hat.mean(dim=1)
                d50_hat = d50_hat.mean(dim=1)
                ts_hat = ts_hat.mean(dim=1)
                volume_hat = volume_hat.mean(dim=1)
                dist_src_hat = dist_src_hat.mean(dim=1)

        elif sti_hat.dim() == 1:
            sti_hat = sti_hat.mean()
            alcons_hat = alcons_hat.mean()
            tr_hat = tr_hat.mean()
            edt_hat = edt_hat.mean()
            c80_hat = c80_hat.mean()
            c50_hat = c50_hat.mean()
            d50_hat = d50_hat.mean()
            ts_hat = ts_hat.mean()
            volume_hat = volume_hat.mean()
            dist_src_hat = dist_src_hat.mean()

        # ------------------- compute evaluation metrics -------------------

        # MAE metric for all the predicted parameters
        loss_sti = self.l1_loss(sti_hat, sti)
        loss_alcons = self.l1_loss(alcons_hat, alcons)
        loss_tr = self.l1_loss(tr_hat, tr)
        loss_edt = self.l1_loss(edt_hat, edt)
        loss_c80 = self.l1_loss(c80_hat, c80)
        loss_c50 = self.l1_loss(c50_hat, c50)
        loss_d50 = self.l1_loss(d50_hat, d50)
        loss_ts = self.l1_loss(ts_hat, ts)
        loss_volume = self.l1_loss(volume_hat, volume)
        loss_dist_src = self.l1_loss(dist_src_hat, dist_src)

        # Pearson correlation coefficient for all the predicted parameters
        corr_sti = self.pearson_corr_coef(sti_hat, sti).abs()
        corr_alcons = self.pearson_corr_coef(alcons_hat, alcons).abs()
        corr_tr = self.pearson_corr_coef(tr_hat, tr).abs()
        corr_edt = self.pearson_corr_coef(edt_hat, edt).abs()
        corr_c80 = self.pearson_corr_coef(c80_hat, c80).abs()
        corr_c50 = self.pearson_corr_coef(c50_hat, c50).abs()
        corr_d50 = self.pearson_corr_coef(d50_hat, d50).abs()
        corr_ts = self.pearson_corr_coef(ts_hat, ts).abs()
        corr_volume = self.pearson_corr_coef(volume_hat, volume).abs()
        corr_dist_src = self.pearson_corr_coef(dist_src_hat, dist_src).abs()

        return {
            "loss_sti": loss_sti,
            "loss_alcons": loss_alcons,
            "loss_tr": loss_tr,
            "loss_edt": loss_edt,
            "loss_c80": loss_c80,
            "loss_c50": loss_c50,
            "loss_d50": loss_d50,
            "loss_ts": loss_ts,
            "loss_volume": loss_volume,
            "loss_dist_src": loss_dist_src,
            "corr_sti": corr_sti,
            "corr_alcons": corr_alcons,
            "corr_tr": corr_tr,
            "corr_edt": corr_edt,
            "corr_c80": corr_c80,
            "corr_c50": corr_c50,
            "corr_d50": corr_d50,
            "corr_ts": corr_ts,
            "corr_volume": corr_volume,
            "corr_dist_src": corr_dist_src,
        }
