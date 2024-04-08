import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.network.components.positional_encoding import RelPosEncoding
from src.network.components.room_encoder import (
    RelPositionMultiHeadedAttention,
    RotaryPositionMultiHeadedAttention,
    XposMultiHeadedAttention,
    RoomFeatureEncoderLayer,
)


def init_bert_params(module):
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
        """Room Feature Encoder Module

        Args:
            embed_dim (int, optional): dimension of embedding. Defaults to 512.
            num_heads (int, optional): number of heads in multiheaded attention. Defaults to 8.
            ch_scale (int, optional): channel scale for feedforward layer. Defaults to 2.
            dropout (int, optional): dropout probability. Defaults to 0.1.
            num_layers (int, optional): number of layers. Defaults to 8.
            pos_enc_type (str, optional): positional encoding type. Defaults to "xpos".
            max_position_len (int, optional): maximum position length. Defaults to 10000.
        """
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
        self.layer_norm = nn.LayerNorm(self.embedding_dim)
        self.layerdrop = encoder_layerdrop

        self.apply(init_bert_params)

    def forward(self, x, padding_mask=None):
        """Forward pass for the Room Feature Encoder Module

        Args:
            x (torch.Tensor): input tensor. Shape: B x T x C
            padding_mask (toch.Tensor, optional): padding mask. Defaults to None.

        Returns:
            x (torch.Tensor): encoded features. Shape: B x T x C
        """

        # B X T X C here
        position_emb = None
        if self.pos_enc_type == "rel_pos":
            position_emb = self.embed_positions(x)

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        for i, layer in enumerate(self.layers):
            dropout_probability = np.random.random()
            if not self.training or (dropout_probability > self.layerdrop):
                x = layer(
                    x,
                    padding_mask=padding_mask,
                    pos_emb=position_emb,
                )

        return x  # B X T X C
