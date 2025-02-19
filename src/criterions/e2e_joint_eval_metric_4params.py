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
        tr_hat = param_hat["tr_hat"]
        c80_hat = param_hat["c80_hat"]
        c50_hat = param_hat["c50_hat"]
        padding_mask = param_hat["padding_mask"]

        # extract groundtruth parameters
        # force cast to float to avoid mismatch in data type
        sti = param_groundtruth["sti"]
        tr = param_groundtruth["tr"]
        c80 = param_groundtruth["c80"]
        c50 = param_groundtruth["c50"]

        assert sti_hat.shape == tr_hat.shape == c80_hat.shape == c50_hat.shape

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

                tr_hat = (tr_hat * reverse_padding_mask).sum(
                    dim=1
                ) / reverse_padding_mask.sum(dim=1)

                c80_hat = (c80_hat * reverse_padding_mask).sum(
                    dim=1
                ) / reverse_padding_mask.sum(dim=1)

                c50_hat = (c50_hat * reverse_padding_mask).sum(
                    dim=1
                ) / reverse_padding_mask.sum(dim=1)

            else:
                # ---------------------- no padding mask handling ---------------------- #
                sti_hat = sti_hat.mean(dim=1)
                tr_hat = tr_hat.mean(dim=1)
                c80_hat = c80_hat.mean(dim=1)
                c50_hat = c50_hat.mean(dim=1)

        elif sti_hat.dim() == 1:
            sti_hat = sti_hat.mean()
            tr_hat = tr_hat.mean()
            c80_hat = c80_hat.mean()
            c50_hat = c50_hat.mean()

        # ------------------- compute evaluation metrics -------------------

        # MAE metric for all the predicted parameters
        loss_sti = self.l1_loss(sti_hat, sti)
        loss_tr = self.l1_loss(tr_hat, tr)
        loss_c80 = self.l1_loss(c80_hat, c80)
        loss_c50 = self.l1_loss(c50_hat, c50)

        # Pearson correlation coefficient for all the predicted parameters
        corr_sti = self.pearson_corr_coef(sti_hat, sti).abs()
        corr_tr = self.pearson_corr_coef(tr_hat, tr).abs()
        corr_c80 = self.pearson_corr_coef(c80_hat, c80).abs()
        corr_c50 = self.pearson_corr_coef(c50_hat, c50).abs()

        return {
            "loss_sti": loss_sti,
            "loss_tr": loss_tr,
            "loss_c80": loss_c80,
            "loss_c50": loss_c50,
            "corr_sti": corr_sti,
            "corr_tr": corr_tr,
            "corr_c80": corr_c80,
            "corr_c50": corr_c50,
        }
