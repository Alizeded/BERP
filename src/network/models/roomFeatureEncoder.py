from typing import Optional

import torch
import torch.nn as nn

from src.network.components.room_feature_encoder import RoomFeatureEncoder


# ablation study for encoder only
class RoomEncoder(nn.Module):
    """
    Conformer Encoder Module for ablation study.
    """

    def __init__(
        self,
        ch_in: int,
        ch_out: int = 1,
        num_layers: int = 8,
        num_heads: int = 8,
        embed_dim: int = 512,
        ch_scale: int = 4,
        dropout_prob: float = 0.1,
        feat_type: str = "mfcc",  # "gammatone", "mel", "mfcc"
        dist_src_est: Optional[bool] = False,
    ):
        super().__init__()

        # conformer encoder
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.num_layer = num_layers
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.ch_scale = ch_scale  # channel scale for feedforward layer
        self.dropout_prob = dropout_prob
        self.dist_src_est = dist_src_est

        self.feature_extractor = None
        self.feat_proj = None

        # feature extractor
        if feat_type == "gammatone" or feat_type == "mel" or feat_type == "mfcc":
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

        self.decoder_sti = nn.Linear(embed_dim, ch_out)
        self.decoder_alcons = nn.Linear(embed_dim, ch_out)
        self.decoder_tr = nn.Linear(embed_dim, ch_out)
        self.decoder_edt = nn.Linear(embed_dim, ch_out)
        self.decoder_c80 = nn.Linear(embed_dim, ch_out)
        self.decoder_c50 = nn.Linear(embed_dim, ch_out)
        self.decoder_d50 = nn.Linear(embed_dim, ch_out)
        self.decoder_ts = nn.Linear(embed_dim, ch_out)
        self.decoder_volume = nn.Linear(embed_dim, ch_out)
        self.decoder_dist_src = nn.Linear(embed_dim, ch_out)

    def decoder_forward(
        self, x: torch.Tensor, padding_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:

        # decoder
        sti_hat = self.decoder_sti(x)
        alcons_hat = self.decoder_alcons(x)
        tr_hat = self.decoder_tr(x)
        edt_hat = self.decoder_edt(x)
        c80_hat = self.decoder_c80(x)
        c50_hat = self.decoder_c50(x)
        d50_hat = self.decoder_d50(x)
        ts_hat = self.decoder_ts(x)
        volume_hat = self.decoder_volume(x)
        dist_src_hat = self.decoder_dist_src(x)

        return {
            "sti_hat": sti_hat.squeeze(),
            "alcons_hat": alcons_hat.squeeze(),
            "tr_hat": tr_hat.squeeze(),
            "edt_hat": edt_hat.squeeze(),
            "c80_hat": c80_hat.squeeze(),
            "c50_hat": c50_hat.squeeze(),
            "d50_hat": d50_hat.squeeze(),
            "ts_hat": ts_hat.squeeze(),
            "volume_hat": volume_hat.squeeze(),
            "dist_src_hat": dist_src_hat.squeeze(),
            "padding_mask": padding_mask,
        }

    def forward(
        self, source: torch.Tensor, padding_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Inputs:
            x: [batch, len, in_dim]
            padding_mask: [batch, len]
        Returns:
            [batch, len]
        """

        # feature extractor
        features = source

        x = features.permute(0, 2, 1)  # [batch, ch_in, len] -> [batch, len, ch_in]

        if self.feat_proj is not None:
            x = self.feat_proj(x)

        x = self.layer_norm(x)

        if padding_mask is not None and not padding_mask.any():
            padding_mask = None

        # encoder
        x = self.encoder(x, padding_mask)

        # decoder
        return self.decoder_forward(x=x, padding_mask=padding_mask)
