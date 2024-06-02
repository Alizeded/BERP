from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .multiheaded_attention import (
    RelPositionMultiHeadedAttention,
    RotaryPositionMultiHeadedAttention,
    XposMultiHeadedAttention,
)


class Transpose(nn.Module):
    """Wrapper class of torch.transpose() for Sequential module."""

    def __init__(self, shape: tuple):
        super(Transpose, self).__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.transpose(*self.shape)


class Swish(nn.Module):
    """
    Swish is a smooth, non-monotonic function that consistently matches or outperforms ReLU on deep networks applied
    to a variety of challenging domains such as Image classification and Machine translation.
    """

    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * inputs.sigmoid()


class DepthWiseConv1d(nn.Module):
    """
    When groups == in_channels and out_channels == K * in_channels, where K is a positive integer,
    this operation is termed in literature as depthwise convolution.

    Parameters:
        ch_in: number of input channels
        ch_out: number of output channels
        kernel_size: size of the convolving kernel
        stride: stride of the convolution, default: 1
    """

    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        super(DepthWiseConv1d, self).__init__()
        assert (
            ch_out % ch_in == 0
        ), "out_channels should be constant multiple of in_channels"

        self.depthwise = nn.Conv1d(
            ch_in,
            ch_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=ch_in,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.depthwise(x)


class pointWiseConv1d(nn.Module):
    """
    When kernel size == 1 conv1d, this operation is termed in literature as pointwise convolution.
    This operation often used to match dimensions.

    Parameters:
        ch_in: number of input channels
        ch_out: number of output channels
        stride: stride of the convolution, default: 1
    """

    def __init__(self, ch_in: int, ch_out: int, stride: int = 1):
        super(pointWiseConv1d, self).__init__()
        self.pointwise = nn.Conv1d(ch_in, ch_out, kernel_size=1, stride=stride)

    def forward(self, x: Tensor) -> Tensor:
        return self.pointwise(x)


class ConvBlock(nn.Module):
    """
    Convolution module starts with a pointwise convolution
    and a gated linear unit (GLU).
    This is followed by a single 1-D depthwise convolution layer.
    Batchnorm is deployed just after the convolution
    to aid training deep models.

    Parameters:
        ch_in: number of input channels
        kernel_size: size of the convolving kernel, int or tuple,
                    default: 31
        dropout: dropout probability
    """

    def __init__(
        self,
        ch_in: int,
        kernel_size: int = 31,
        dropout_prob: float = 0.1,
        expansion_factor: int = 2,
    ):
        super(ConvBlock, self).__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size should be odd number"
        assert expansion_factor == 2, "Currently expansion factor is 2"

        self.conv = nn.Sequential(
            nn.LayerNorm(ch_in),
            Transpose(shape=(1, 2)),
            pointWiseConv1d(ch_in, ch_in * expansion_factor, stride=1),
            nn.GLU(dim=1),
            DepthWiseConv1d(
                ch_in, ch_in, kernel_size, stride=1, padding=(kernel_size - 1) // 2
            ),
            nn.BatchNorm1d(ch_in),
            Swish(),
            pointWiseConv1d(ch_in, ch_in, stride=1),
            nn.Dropout(dropout_prob),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        return x.transpose(1, 2)


class PosWiseFeedForwardModule(nn.Module):
    """
    Position-wise Feed-Forward Module

    Feed Forward Module follow pre-norm residual units and apply layer normalization within the residual unit
    and on the input before the first linear layer. This module also apply Swish activation and dropout, which helps
    regularizing the network.

    Parameters:
        embed_dim: int, dimension of embedding
        expansion_factor: int, expansion factor of feed forward module
        dropout_prob: float, dropout probability

    """

    def __init__(
        self, encoder_dim: int, expansion_factor: int = 4, dropout_prob: float = 0.1
    ):
        super(PosWiseFeedForwardModule, self).__init__()
        self.sequentail = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, encoder_dim * expansion_factor, bias=True),
            Swish(),
            nn.Dropout(dropout_prob),
            nn.Linear(encoder_dim * expansion_factor, encoder_dim, bias=True),
            nn.Dropout(dropout_prob),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.sequentail(x)


class residual_connection(nn.Module):
    """
    Residual Connection Module.
    outputs = (module(inputs) x module_factor + inputs x input_factor)
    """

    def __init__(
        self, module: nn.Module, module_factor: float = 1.0, input_factor: float = 1.0
    ):
        super(residual_connection, self).__init__()
        self.module = module
        self.module_factor = module_factor
        self.input_factor = input_factor

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return (self.module(x, *args, **kwargs) * self.module_factor) + (
            x * self.input_factor
        )


class RoomFeatureEncoderLayer(nn.Module):
    def __init__(
        self,
        num_heads: int = 8,
        embed_dim: int = 512,
        ch_scale: int = 2,
        dropout_prob: float = 0.1,
        pos_enc_type: str = "xpos",  # xpos or rel_pos or rope
        half_step_residual: bool = True,
    ):
        """Room Feature Encoder Layer

        Args:
            num_heads (int, optional): number of heads in multiheaded attention. Defaults to 8.
            embed_dim (int, optional): dimension of embedding. Defaults to 512.
            ch_scale (int, optional): channel scale for feedforward layer. Defaults to 2.
            dropout_prob (float, optional): dropout probability. Defaults to 0.1.
            pos_enc_type (str, optional): positional encoding type. Defaults to "xpos".
        """
        super(RoomFeatureEncoderLayer, self).__init__()

        feedforward_residual_factor = 0.5 if half_step_residual else 1.0
        self.pos_enc_type = pos_enc_type

        self.ffn1 = residual_connection(
            module=PosWiseFeedForwardModule(
                encoder_dim=embed_dim,
                expansion_factor=ch_scale,
                dropout_prob=dropout_prob,
            ),
            module_factor=feedforward_residual_factor,
        )

        self.layer_norm = nn.LayerNorm(embed_dim)

        self.self_attn = None

        if self.pos_enc_type == "xpos":
            self.self_attn = residual_connection(
                module=XposMultiHeadedAttention(
                    num_heads=num_heads,
                    embed_dim=embed_dim,
                    dropout_prob=dropout_prob,
                ),
            )
        if self.pos_enc_type == "rel_pos":
            self.self_attn = residual_connection(
                module=RelPositionMultiHeadedAttention(
                    nums_heads=num_heads,
                    xpos_scale_base=embed_dim,
                    embed_dim=embed_dim,
                    dropout_prob=dropout_prob,
                ),
            )
        if self.pos_enc_type == "rope":
            self.self_attn = residual_connection(
                module=RotaryPositionMultiHeadedAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout_prob=dropout_prob,
                ),
            )

        self.conv_module = residual_connection(
            module=ConvBlock(
                ch_in=embed_dim,
                kernel_size=31,
                dropout_prob=dropout_prob,
            ),
        )

        self.ffn2 = residual_connection(
            module=PosWiseFeedForwardModule(
                encoder_dim=embed_dim,
                expansion_factor=ch_scale,
                dropout_prob=dropout_prob,
            ),
            module_factor=feedforward_residual_factor,
        )

        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: Tensor,
        padding_mask: Optional[Tensor],
        pos_emb: Optional[Tensor] = None,
    ) -> Tensor:
        """forward pass of RoomFeatureEncoderLayer

        Args:
            x (Tensor): input tensor [B x T x C]
            padding_mask (Optional[torch.Tensor]): key padding mask [B x T]

        Returns:
            Tensor: B x T x C
        """
        x = self.ffn1(x=x)
        x = self.layer_norm(input=x)

        if self.pos_enc_type == "rel_pos" and pos_emb is not None:
            x = self.self_attn(
                x=x,
                key=x,
                value=x,
                pos_embed=pos_emb,
                key_padding_mask=padding_mask,
            )
        else:
            x = self.self_attn(x=x, key=x, value=x, key_padding_mask=padding_mask)

        x = self.conv_module(x=x)

        x = self.ffn2(x=x)

        x = self.final_layer_norm(input=x)
        return x
