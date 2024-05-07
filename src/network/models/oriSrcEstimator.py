from typing import Optional

import torch
import torch.nn as nn

from src.network.components.conv_prenet import ConvFeatureExtractionModel
from src.network.components.room_feature_encoder import RoomFeatureEncoder
from src.network.components.multiheaded_attention import (
    RotaryPositionMultiHeadedAttention,
)
from src.network.components.parametric_predictor import ParametricPredictor


# ------------------------------- binary bias corrector ------------------------------- #
class BinaryClassifier(nn.Module):
    """Binary classifier for bias correction"""

    def __init__(self, in_dim: int = 512, out_dim: int = 1):
        super(BinaryClassifier, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.fc1 = nn.Linear(in_dim, in_dim)
        self.fc2 = nn.Linear(in_dim, out_dim)

        self.rotary_slf_attn = RotaryPositionMultiHeadedAttention(
            embed_dim=in_dim,
            nums_heads=8,
            dropout_prob=0.1,
        )

        self.temporal_collapsing = nn.AdaptiveAvgPool1d(output_size=1)

        self.activation = nn.GELU()

        self.dropout = nn.Dropout(0.25)

        self.output_activation = nn.Sigmoid()

    def forward(
        self, x: torch.Tensor, padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.rotary_slf_attn(
            query=x, key=x, value=x, key_padding_mask=padding_mask
        )  # (B, T, C)
        x = x.permute(0, 2, 1)  # (B, T, C) -> (B, C, T)
        if padding_mask is not None and padding_mask.any():
            temp_avg = x.masked_fill(padding_mask.unsqueeze(1), 0.0).sum(
                dim=-1
            ) / padding_mask.logical_not().sum(dim=-1, keepdim=True)

            temp_avg = temp_avg.view(x.shape[0], x.shape[1])
        else:
            temp_avg = self.temporal_collapsing(x)  # (B, C, T) -> (B, C, 1)
        x = temp_avg.squeeze()  # (B, C, 1) -> (B, C)
        logits = self.fc2(x)
        prob = self.output_activation(logits)

        return prob.squeeze()  # (B, 1) -> (B,)


# ------------------------------- ori_src estimator ------------------------------- #


class OriSrcEstimator(nn.Module):
    """Orientation module."""

    def __init__(
        self,
        ch_in: int,
        ch_out: int = 1,
        num_layers: int = 8,
        num_heads: int = 8,
        embed_dim: int = 512,
        ch_scale: int = 4,
        dropout_prob: float = 0.1,
        conv_feature_layers=[
            (512, 10, 4),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2),
        ],
        feat_type: str = "gammatone",  # "gammatone", "mel", "mfcc", "waveform"
        decoder_type: str = "parametric_predictor",  # parametric_predictor only
        num_channels_decoder: Optional[int] = 384,
        kernel_size_decoder: Optional[int] = 3,
        num_layers_decoder: int = 3,
        dropout_decoder: float = 0.5,
        bias_correction: bool = True,
    ):  # sourcery skip: collection-into-set, default-mutable-arg, merge-comparisons
        """Orientation module.

        Args:
            ch_in (int): input channel
            ch_out (int, optional): output channel. Defaults to 1.
            num_layers (int, optional): number of room feature encoder layers. Defaults to 8.
            num_heads (int, optional): number of heads in multiheaded attention. Defaults to 8.
            embed_dim (int, optional): dimension of embedding. Defaults to 512.
            ch_scale (int, optional): channel scale for feedforward layer. Defaults to 4.
            dropout_prob (float, optional): dropout probability of room feature encoder. Defaults to 0.1.
            conv_feature_layers (list, optional): conv prenet featurizer. Defaults to [ (512, 10, 4), (512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 2, 2), (512, 2, 2), ].
            feat_type (str, optional): feature type. Defaults to "gammatone".
            kernel_size_decoder (Optional[int], optional): kernel size of predictor decoder. Defaults to 3.
            num_layers_decoder (int, optional): number of layers in predictor decoder. Defaults to 3.
            dropout_decoder (float, optional): dropout probability of predictor decoder. Defaults to 0.5.
            bias_correction (bool, optional): whether to use bias correction. Defaults to True.

        Raises:
            ValueError: _description_
            NotImplementedError: _description_
            NotImplementedError: _description_
        """
        super(OriSrcEstimator, self).__init__()

        # conformer encoder
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.num_layer = num_layers
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.ch_scale = ch_scale  # channel scale for feedforward layer
        self.dropout_prob = dropout_prob
        self.bias_correction = bias_correction

        self.feature_extractor = None
        self.feat_proj = None
        self.conv_feature_layers = conv_feature_layers

        # feature extractor
        if feat_type == "waveform":
            self.feature_extractor = ConvFeatureExtractionModel(
                conv_layers=conv_feature_layers,
                mode="layer_norm",
                dropout=dropout_prob,
                conv_bias=False,
            )

        elif feat_type == "gammatone" or feat_type == "mel" or feat_type == "mfcc":
            self.feat_proj = nn.Linear(ch_in, embed_dim)

        else:
            raise ValueError(f"Feature type {feat_type} not supported")

        # layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)

        # room feature encoder
        self.encoder = RoomFeatureEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ch_scale=ch_scale,
            dropout=dropout_prob,
            num_layers=num_layers,
        )

        # decoder
        self.decoder_type = decoder_type
        self.num_layers_decoder = num_layers_decoder
        self.dropout_decoder = dropout_decoder

        if self.decoder_type != "parametric_predictor":
            raise NotImplementedError("Only parametric predictor is supported")

        self.num_channels_decoder = num_channels_decoder
        self.kernel_size_decoder = kernel_size_decoder
        # binary classifier for bias correction
        self.binary_classifier_azimuth = BinaryClassifier(
            in_dim=embed_dim,
            out_dim=1,
        )

        self.binary_classifier_elevation = BinaryClassifier(
            in_dim=embed_dim,
            out_dim=1,
        )

        if self.decoder_type != "parametric_predictor":
            raise NotImplementedError("Only parametric predictor is supported")
        # parametric predictor for ori_src
        self.parametric_predictor_azimuth = ParametricPredictor(
            in_dim=embed_dim,
            num_layers=num_layers_decoder,
            num_channels=num_channels_decoder,
            kernel_size=kernel_size_decoder,
            dropout_prob=dropout_decoder,
        )

        self.parametric_predictor_elevation = ParametricPredictor(
            in_dim=embed_dim,
            num_layers=num_layers_decoder,
            num_channels=num_channels_decoder,
            kernel_size=kernel_size_decoder,
            dropout_prob=dropout_decoder,
        )

    def binary_classifier_forward(self, x: torch.Tensor):
        """binary classifier forward pass"""

        x_azimuth = self.binary_classifier_azimuth(x)
        x_azimuth = x_azimuth.squeeze()  # B x 1 -> B

        x_elevation = self.binary_classifier_elevation(x)
        x_elevation = x_elevation.squeeze()  # B x 1 -> B

        return x_azimuth, x_elevation

    def parametric_predictor_forward(self, x: torch.Tensor):
        """parametric predictor forward pass"""

        ori_azimuth_hat = self.parametric_predictor_azimuth(x)
        ori_evelation_hat = self.parametric_predictor_elevation(x)

        ori_azimuth_hat = ori_azimuth_hat.squeeze()  # B x T x 1 -> B x T
        ori_evelation_hat = ori_evelation_hat.squeeze()  # B x T x 1 -> B x T

        return ori_azimuth_hat, ori_evelation_hat

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            return torch.floor((input_length - kernel_size) / stride + 1)

        for i in range(len(self.conv_feature_layers)):
            input_lengths = _conv_out_length(
                input_lengths,
                self.conv_feature_layers[i][1],
                self.conv_feature_layers[i][2],
            )

        return input_lengths.to(torch.long)

    def forward(
        self,
        source: torch.Tensor,
        padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass of the model.

        Parameters:1
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len).

        Returns:
            ori_src_hat (torch.Tensor): Estimated ori_src of shape (batch_size,).
        """
        if self.feature_extractor is not None:
            features = self.feature_extractor(source)
        else:
            features = source

        x = features.permute(0, 2, 1)  # [batch, ch_in, len] -> [batch, len, ch_in]

        if self.feat_proj is not None:
            x = self.feat_proj(x)

        x = self.layer_norm(x)

        if (
            padding_mask is not None
            and padding_mask.any()
            and self.feature_extractor is not None
        ):
            # B x T
            input_lengths = (1 - padding_mask.long()).sum(-1)
            # apply conv formula to get real output_lengths
            output_lengths = self._get_feat_extract_output_lengths(input_lengths)

            padding_mask = torch.zeros(x.shape[:2], dtype=x.dtype, device=x.device)

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
            ).bool()  # for padded values, set to True, else False

        if padding_mask is not None and not padding_mask.any():
            padding_mask = None

        # encoder
        x = self.encoder(x, padding_mask)

        # binary classifier forward
        if self.bias_correction:
            judge_prob_azimuth, judge_prob_elevation = self.binary_classifier_forward(x)

        # transformer decoder
        if self.decoder_type == "parametric_predictor":
            ori_azimuth_hat, ori_elevation_hat = self.parametric_predictor_forward(x)
        else:
            raise NotImplementedError("Only parametric predictor is supported")

        if self.bias_correction:

            return {
                "azimuth_hat": ori_azimuth_hat,
                "elevation_hat": ori_elevation_hat,
                "judge_prob_azimuth": judge_prob_azimuth,
                "judge_prob_elevation": judge_prob_elevation,
                "padding_mask": padding_mask,
            }

        else:
            return {
                "azimuth_hat": ori_azimuth_hat,
                "elevation_hat": ori_elevation_hat,
                "padding_mask": padding_mask,
            }
