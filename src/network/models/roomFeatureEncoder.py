from typing import Dict, Optional
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
        decoder_type: str = "rir",  # rir, volume, distSrc, oriSrc, joint
        dist_src_est: Optional[bool] = False,
    ):
        super(RoomEncoder, self).__init__()

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

        # decoder type
        self.decoder_type = decoder_type

        # decoder
        self.decoder = nn.Linear(embed_dim, ch_out)

    def decoder_forward(
        self, x: torch.Tensor, padding_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:

        if self.decoder_type == "rir":
            Th_hat = self.decoder(x)
            Tt_hat = self.decoder(x)

            Th_hat, Tt_hat = (
                Th_hat.squeeze(),
                Tt_hat.squeeze(),
            )  # B x C x T -> B x T

            return {
                "Th_hat": Th_hat,
                "Tt_hat": Tt_hat,
                "padding_mask": padding_mask,
            }

        elif self.decoder_type == "volume":
            volume_hat = self.decoder(x)
            volume_hat = volume_hat.squeeze()  # B x C x T -> B x T

            return {"volume_hat": volume_hat, "padding_mask": padding_mask}

        elif self.decoder_type == "distSrc":
            dist_src_hat = self.decoder(x)
            dist_src_hat = dist_src_hat.squeeze()

            return {"dist_src_hat": dist_src_hat, "padding_mask": padding_mask}

        elif self.decoder_type == "oriSrc":
            azimuth_hat = self.decoder(x)
            elevation_hat = self.decoder(x)

            azimuth_hat, elevation_hat = (
                azimuth_hat.squeeze(),
                elevation_hat.squeeze(),
            )

            return {
                "azimuth_hat": azimuth_hat,
                "elevation_hat": elevation_hat,
                "padding_mask": padding_mask,
            }

        elif self.decoder_type == "joint":
            Th_hat = self.decoder(x)
            Tt_hat = self.decoder(x)
            volume_hat = self.decoder(x)
            dist_src_hat = self.decoder(x)
            # azimuth_hat = self.decoder(x)
            # elevation_hat = self.decoder(x)

            Th_hat = Th_hat.squeeze()
            Tt_hat = Tt_hat.squeeze()
            volume_hat = volume_hat.squeeze()
            dist_src_hat = dist_src_hat.squeeze()
            # azimuth_hat = azimuth_hat.squeeze()
            # elevation_hat = elevation_hat.squeeze()

            return {
                "Th_hat": Th_hat,
                "Tt_hat": Tt_hat,
                "volume_hat": volume_hat,
                "dist_src_hat": dist_src_hat,
                # "azimuth_hat": azimuth_hat,
                # "elevation_hat": elevation_hat,
                "padding_mask": padding_mask,
            }

    def forward(
        self, source: torch.Tensor, padding_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
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
        x = self.decoder_forward(x=x, padding_mask=padding_mask)

        return x
