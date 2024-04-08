import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Optional
from einops import rearrange


class PosWiseFeedForward(nn.Module):
    """Position-wise Feedforward"""

    def __init__(
        self, embed_dim: int, expansion_factor: int = 4, dropout_prob: float = 0.1
    ):
        super(PosWiseFeedForward, self).__init__()
        self.sequential = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * expansion_factor),
            nn.PReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(embed_dim * expansion_factor, embed_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.sequential(x)


class CrossMultiHeadedAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(
        self,
        embed_dim: int = 512,
        kdim: int = 512,
        vdim: int = 512,
        num_heads: int = 8,
        dropout_prob: float = 0.1,
    ):
        super(CrossMultiHeadedAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        assert kdim % num_heads == 0, "kdim must be divisible by num_heads"
        assert kdim == vdim, "kdim must be equal to vdim"

        self.embed_dim = embed_dim
        self.kdim = kdim
        self.vdim = vdim
        self.dim_per_head = embed_dim // num_heads
        self.num_heads = num_heads
        self.sqrt_dim = self.dim_per_head**0.5

        self.w_q = nn.Linear(embed_dim, kdim)
        self.w_k = nn.Linear(kdim, kdim)
        self.w_v = nn.Linear(vdim, vdim)

        self.dropout = nn.Dropout(dropout_prob)

        self.out_proj = nn.Linear(vdim, embed_dim)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ):
        q = rearrange(self.w_q(query), "b l (h d) -> b h l d", h=self.num_heads)
        # q: [batch_size, n_head, len_q, d_q]
        k = rearrange(self.w_k(key), "b l (h d) -> b h l d", h=self.num_heads)
        # k: [batch_size, n_head, len_k, d_k]
        v = rearrange(self.w_v(value), "b l (h d) -> b h l d", h=self.num_heads)

        scaled_dot_product = (
            torch.einsum("b h i q, b h j k -> b h i j", q, k) / self.sqrt_dim
        )

        attn = F.softmax(scaled_dot_product, dim=-1)
        attn = self.dropout(attn)

        context = torch.einsum("b h i j, b h j v -> b h i v", attn, v)
        context = rearrange(context, "b h l v -> b l (h v)")

        return self.out_proj(context)


class NonAutoregressiveDecoder(nn.Module):
    """
    NonAutoregressive Transformer Decoder
    """

    def __init__(
        self,
        embed_dim_slfattn: int = 256,
        embed_dim_crossattn: int = 128,
        num_head: int = 8,
        ch_scale: int = 2,
        dropout_prob: float = 0.2,
        is_gated: bool = True,
    ):
        super(NonAutoregressiveDecoder, self).__init__()

        self.embed_dim_slfattn = embed_dim_slfattn
        self.embed_dim_crossattn = embed_dim_crossattn
        self.num_head = num_head
        self.dropout_prob = dropout_prob
        self.is_gated = is_gated

        assert (
            embed_dim_slfattn % num_head == 0
        ), "embed_dim of self-attn must be divisible by num_head"
        assert (
            embed_dim_crossattn % num_head == 0
        ), "embed_dim of cross-attn must be divisible by num_head"

        self.slf_attn_layer = nn.MultiheadAttention(
            embed_dim=embed_dim_slfattn,
            num_heads=num_head,
            dropout=dropout_prob,
            kdim=embed_dim_slfattn,
            vdim=embed_dim_slfattn,
            batch_first=True,
        )

        self.slf_attn_layer_norm = nn.LayerNorm(embed_dim_slfattn)

        self.cross_attn_layer = CrossMultiHeadedAttention(
            embed_dim=embed_dim_crossattn,
            kdim=embed_dim_slfattn,
            vdim=embed_dim_slfattn,
            num_heads=num_head,
            dropout_prob=dropout_prob,
        )

        self.cross_attn_layer_norm = nn.LayerNorm(embed_dim_crossattn)

        self.ffn_layer = PosWiseFeedForward(
            embed_dim=embed_dim_crossattn,
            expansion_factor=ch_scale,
            dropout_prob=dropout_prob,
        )

        self.ffn_layer_norm = nn.LayerNorm(embed_dim_crossattn)

        self.dropout = nn.Dropout(dropout_prob)

    def forward(
        self, src: Tensor, tgt: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # target sequence self-attention
        if mask is not None:
            slf_attn, slf_attn_w = self.slf_attn_layer(
                query=tgt, key=tgt, value=tgt, attn_mask=mask
            )
        else:
            slf_attn, slf_attn_w = self.slf_attn_layer(
                query=tgt, key=tgt, value=tgt, attn_mask=None
            )

        z = self.slf_attn_layer_norm(slf_attn)  # post-norm
        tgt = tgt + self.dropout(z)  # residual connection

        # target and source cross-attention
        cross_attn = self.cross_attn_layer(query=tgt, key=src, value=src)
        x = self.cross_attn_layer_norm(cross_attn)  # post-norm
        x = tgt + self.dropout(x)  # residual connection

        # feed-forward
        ffn = self.ffn_layer(x)
        z = self.ffn_layer_norm(ffn)  # post-norm
        x = x + self.dropout(z)  # residual connection

        return x


class DimUpsampler(nn.Module):

    """Dimension Upsampler"""

    def __init__(
        self,
        ch_in: int = 1,
        ch_out: int = 512,
        dropout_prob: float = 0.2,
    ):
        super(DimUpsampler, self).__init__()

        self.ch_in = ch_in
        self.ch_out = ch_out
        self.dropout_prob = dropout_prob

        self.upsample = nn.Sequential(
            nn.Linear(ch_in, ch_out),
            nn.LayerNorm(ch_out),
            nn.Dropout(dropout_prob),
            nn.PReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters:
            x: Tensor
                input tensor, shape (batch, seq, ch_in)
        Returns:
            out: Tensor
                output tensor, shape (batch, seq, ch_out)
        """
        x = self.upsample(x)

        return x
