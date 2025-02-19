import einops
import torch
import torch.nn as nn
from torch.nn import L1Loss, SmoothL1Loss
from torchmetrics import PearsonCorrCoef

from src.utils.unitary_linear_norm import unitary_norm_inv


class HuberLoss(nn.Module):

    def __init__(
        self,
        phase: str,
        module: str,
        ablation: bool = False,
    ):
        super().__init__()
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

            elif self.phase == "val":
                # Collapse the pred value to match the target value
                pred = pred.mean(dim=1)

                # Calculate the loss
                loss = self.l1_loss(pred, target)

                # Compute the Pearson Correlation Coefficient
                corr_coef = self.pearson_corr(pred, target).abs()

        if self.phase == "train":
            return {"loss": loss}

        elif self.phase == "val":
            return {"loss": loss, "corr_coef": corr_coef}

        else:
            raise NotImplementedError(
                "Phase not implemented, please choose either 'train' or 'val' or 'test'"
            )

    def forward(
        self,
        param_hat: torch.Tensor,
        param: torch.Tensor,
        padding_mask: torch.Tensor = None,
    ):

        if self.phase == "train":
            # Calculate the loss
            loss = self.huber_loss_calculate(param_hat, param, padding_mask)

            return {"loss": loss["loss"]}

        elif self.phase == "val":
            # Calculate the loss
            loss = self.huber_loss_calculate(param_hat, param, padding_mask)

            return {
                "loss": loss["loss"],
                "corr_coef": loss["corr_coef"],
            }

        elif self.phase == "test":

            if self.ablation is False:  # no ablation for repadding mask
                if padding_mask is not None and padding_mask.any():
                    padding_mask, reverse_padding_mask = self._re_padding_mask(
                        pred=param_hat, padding_mask=padding_mask
                    )
            else:
                if padding_mask is not None and padding_mask.any():
                    reverse_padding_mask = padding_mask.logical_not()

            if padding_mask is not None and padding_mask.any():
                # Collapse the pred value to match the target value
                param_hat = (param_hat * reverse_padding_mask).sum(
                    dim=1
                ) / reverse_padding_mask.sum(dim=1)

            else:
                param_hat = param_hat.mean(dim=1)

            loss_l1 = self.l1_loss(param_hat, param)

            corr_coef = self.pearson_corr(param_hat, param).abs()

            return {
                "loss": loss_l1,
                "corr_coef": corr_coef,
            }

        else:
            raise NotImplementedError(
                "Phase not implemented, please choose either train or val or test"
            )
