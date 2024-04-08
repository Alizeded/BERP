import torch
import torch.nn as nn


from src.network.components.conv_prenet import ConvFeatureExtractionModel
from src.network.components.parametric_predictor import ParametricPredictor
from src.network.components.room_feature_encoder import RoomFeatureEncoder


from typing import Dict, Optional


# ------------------------------- Th, Tt estimator for RIR ------------------------------- #


class ThTtEstimator(nn.Module):
    """RIR module for Ti, Td of SSIR estimation"""

    def __init__(
        self,
        ch_in: int,
        ch_out: int = 2,
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
    ):
        """RIR module for Ti, Td of SSIR estimation

        Args:
            ch_in (int): input channel
            ch_out (int, optional): output channel. Defaults to 2.
            num_layers (int, optional): number of room feature encoder layers. Defaults to 8
            num_heads (int, optional): number of heads in multiheaded attention. Defaults to 8.
            embed_dim (int, optional): dimension of embedding. Defaults to 512.
            ch_scale (int, optional): channel scale for feedforward layer. Defaults to 4.
            dropout_prob (float, optional): dropout probability of room feature encoder. Defaults to 0.1.
            conv_feature_layers (list, optional): conv prenet featuriztion. Defaults to [ (512, 10, 4), (512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 2, 2), (512, 2, 2), ].
            feat_type (str, optional): feature type. Defaults to "gammatone".
            kernel_size_decoder (Optional[int], optional): kernel size of predictor decoder. Defaults to 3.
            num_layers_decoder (int, optional): number of layers in predictor decoder. Defaults to 3.
            dropout_decoder (float, optional): dropout probability of predictor decoder. Defaults to 0.5.

        Raises:
            ValueError: Feature type should be one of "gammatone", "mel", "mfcc", "waveform"
            NotImplementedError: decoder type should be "parametric_predictor" only
            NotImplementedError: decoder type should be "parametric_predictor" only
        """
        super(ThTtEstimator, self).__init__()

        # room feature space encoder configuration
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.num_layer = num_layers
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.ch_scale = ch_scale  # channel scale for feedforward layer
        self.dropout_prob = dropout_prob

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

        if self.decoder_type == "parametric_predictor":
            self.num_channels_decoder = num_channels_decoder
            self.kernel_size_decoder = kernel_size_decoder
        else:
            raise NotImplementedError("Only parameteric predictor is supported")

        # parametric predictor decoder
        if self.decoder_type == "parametric_predictor":
            # parametric predictor for Th, Tt, volume respectively
            self.parametric_predictor_Th = ParametricPredictor(
                in_dim=embed_dim,
                num_layers=num_layers_decoder,
                num_channels=num_channels_decoder,
                kernel_size=kernel_size_decoder,
                dropout_prob=dropout_decoder,
            )

            self.parametric_predictor_Tt = ParametricPredictor(
                in_dim=embed_dim,
                num_layers=num_layers_decoder,
                num_channels=num_channels_decoder,
                kernel_size=kernel_size_decoder,
                dropout_prob=dropout_decoder,
            )
        else:
            raise NotImplementedError("Only parameteric predictor is supported")

    def parametric_predictor_forward(self, x: torch.Tensor):
        """parametric predictor forward pass"""

        Th_hat = self.parametric_predictor_Th(x)
        Tt_hat = self.parametric_predictor_Tt(x)

        Th_hat, Tt_hat = (
            Th_hat.squeeze(),
            Tt_hat.squeeze(),
        )  # B x T x 1 -> B x T

        return Th_hat, Tt_hat

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
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of RIR module.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, freq_bins, seq_len) or (batch_size, seq_len).
            padding_mask (torch.Tensor, optional): Padding mask of shape (batch_size, seq_len). Defaults to None.

        Returns:
            Th_hat (torch.Tensor): Estimated Th of shape (batch_size, seq_len). Ti of SSIR.
            Tt_hat (torch.Tensor): Estimated Tt of shape (batch_size, seq_len). Td of SSIR.
            padding_mask (torch.Tensor): Padding mask of shape (batch_size, seq_len).
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

        if self.decoder_type == "parametric_predictor":
            Th_hat, Tt_hat = self.parametric_predictor_forward(x)
        else:
            raise NotImplementedError("Only parameteric predictor is supported")

        return {"Th_hat": Th_hat, "Tt_hat": Tt_hat, "padding_mask": padding_mask}
