import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.network.components.conv_prenet import ConvFeatureExtractionModel
from src.network.components.positional_encoding import RelPosEncoding
from src.network.components.room_encoder_layer import (
    RelPositionMultiHeadedAttention,
    RoomFeatureEncoderLayer,
    RotaryPositionMultiHeadedAttention,
    XposMultiHeadedAttention,
)


def init_bert_params(module):  # sourcery skip: merge-isinstance
    """
    Initialize the weights specific to the BERT Model.
    This overrides the default initializations depending on the specified arguments.
        1. If normal_init_linear_weights is set then weights of linear
           layer will be initialized using the normal distribution and
           bais will be set to the specified value.
        2. If normal_init_embed_weights is set then weights of embedding
           layer will be initialized using the normal distribution.
        3. If normal_init_proj_weights is set then weights of
           in_project_weight for MultiHeadAttention initialized using
           the normal distribution (to be validated).
    """

    def normal_(data):
        # with FSDP, module params will be on CUDA, so we cast them back to CPU
        # so that the RNG is consistent with and without FSDP
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if (
        isinstance(module, RelPositionMultiHeadedAttention)
        or isinstance(module, XposMultiHeadedAttention)
        or isinstance(
            module,
            RotaryPositionMultiHeadedAttention,
        )
    ):
        normal_(module.w_q.weight.data)
        normal_(module.w_k.weight.data)
        normal_(module.w_v.weight.data)


# ------------------------------- Encoder for Occupants Number ------------------------------- #


class NumOccEstimator(nn.Module):
    """regressor to map occupants number"""

    def __init__(
        self,
        ch_in: int,
        ch_out: int = 13,
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
        feat_type: str = "gammatone",  # "gammatone", "mel", "MFCC", "cnn_prenet"
    ):  # sourcery skip: collection-into-set, merge-comparisons
        """Occupancy Module

        Args:
            ch_in (int): input channel
            ch_out (int, optional): output logits dimension. Defaults to 13.
            num_layers (int, optional): number of room feature encoder layers. Defaults to 8.
            num_heads (int, optional): number of heads in multiheaded attention. Defaults to 8.
            embed_dim (int, optional): dimension of embedding. Defaults to 512.
            ch_scale (int, optional): channel scale for feedforward layer. Defaults to 4.
            dropout_prob (float, optional): dropout probability. Defaults to 0.1.
            conv_feature_layers (list, optional): conv prenet featurization. Defaults to [ (512, 10, 4), (512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 2, 2), (512, 2, 2), ].
            feat_type (str, optional): feature type. Defaults to "gammatone".
        """
        super(NumOccEstimator, self).__init__()

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

        self.layer_norm = nn.LayerNorm(embed_dim)

        self.encoder = RoomFeatureEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ch_scale=ch_scale,
            dropout=dropout_prob,
            num_layers=num_layers,
        )

        self.classifier = nn.Linear(embed_dim, ch_out)

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
        self, source: torch.Tensor, padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Forward pass for the Occupancy Module

        Args:
            source (torch.Tensor): input tensor. (batch_size, freq_bins, seq_len) or (batch_size, seq_len)
            padding_mask (torch.Tensor, optional): key padding mask (batch_size, seq_len). Defaults to None.

        Returns:
            logits: output logits, (batch_size, seq_len, num_classes)
            padding_mask: key padding mask, (batch_size, seq_len)
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
            padding_mask = (1 - padding_mask.flip([-1]).cumsum(-1).flip([-1])).bool()

        if padding_mask is not None and not padding_mask.any():
            padding_mask = None

        # encoder
        x = self.encoder(x, padding_mask)

        # classifier layer
        x = self.classifier(x)

        return {"logits": x, "padding_mask": padding_mask}

    def get_logits(self, net_output):
        logits = net_output["logits"]

        if net_output["padding_mask"] is not None and net_output["padding_mask"].any():
            number_of_classes = logits.size(-1)
            masking_tensor = torch.ones(
                number_of_classes, device=logits.device
            ) * float("-inf")
            masking_tensor[0] = 0

            if logits.size(1) > net_output["padding_mask"].size(1):
                net_output["padding_mask"] = F.pad(
                    net_output["padding_mask"], (1, 0), value=False
                )

            logits[net_output["padding_mask"]] = masking_tensor.type_as(logits)

        return logits

    def get_normalized_probs(self, net_output, log_prob):
        """Get normalized probabilities (or log probs) from a net's output."""

        logits = self.get_logits(net_output)

        if log_prob:
            return F.log_softmax(logits.float(), dim=-1)
        else:
            return F.softmax(logits.float(), dim=-1)


class RoomFeatureEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        ch_scale: int = 2,
        dropout: int = 0.1,
        num_layers: int = 8,
        pos_enc_type: str = "xpos",  # xpos or rel_pos or rope
        encoder_layerdrop=0.0,  # probability of dropping a tarnsformer layer
        layer_norm_first=True,
        max_position_len: int = 10000,
    ):
        super().__init__()
        self.dropout = dropout
        self.embedding_dim = embed_dim
        self.ch_scale = ch_scale
        self.num_heads = num_heads

        self.pos_enc_type = pos_enc_type

        if self.pos_enc_type == "rel_pos":
            self.embed_positions = RelPosEncoding(max_position_len, self.embedding_dim)

        self.layers = nn.ModuleList(
            [
                RoomFeatureEncoderLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ch_scale=ch_scale,
                    dropout_prob=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.layer_norm_first = layer_norm_first
        self.layer_norm = torch.nn.LayerNorm(self.embedding_dim)
        self.layerdrop = encoder_layerdrop

        self.apply(init_bert_params)

    def forward(self, x, padding_mask=None):
        # B x T x C -> T x B x C
        # x = x.transpose(0, 1)

        # B X T X C here
        position_emb = None
        if self.pos_enc_type == "rel_pos":
            position_emb = self.embed_positions(x)

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        for layer in self.layers:
            dropout_probability = np.random.random()
            if not self.training or (dropout_probability > self.layerdrop):
                x = layer(
                    x,
                    padding_mask=padding_mask,
                    pos_emb=position_emb,
                )

        # T x B x C -> B x T x C
        # x = x.transpose(0, 1)

        return x
