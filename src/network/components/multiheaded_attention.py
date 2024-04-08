import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from .positional_encoding import XPOS, RotaryPositionEncoding


def rotate_half(x: Tensor) -> Tensor:
    """Rotate the input tensor by 180 degrees.

    Args:
        x: The input tensor.
    """
    x = rearrange(x, "... (d j) -> ... d j", j=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d j -> ... (d j)")


def rotary_pos_enc(
    q: Tensor, k: Tensor, cos_pos: Tensor, sin_pos: Tensor, offset: int = 0
) -> Tensor:
    """Rotary position encoding.

    Args:
        query: The query tensor. shape: [B, T, H, C]
        key: The key tensor. shape: [B, T, H, C]
        cos_pos: The cosine positional encoding. shape: [1, T, 1, C]
        sin_pos: The sine positional encoding. shape: [1, T, 1, C]
        offset: The offset.
    """
    cos_pos = cos_pos[:, offset : offset + q.shape[1], ...]
    sin_pos = sin_pos[:, offset : offset + q.shape[1], ...]

    q, k = map(lambda x: x * cos_pos + rotate_half(x) * sin_pos, (q, k))

    return q, k


class RotaryPositionMultiHeadedAttention(nn.Module):
    """Rotary Position Multi-Headed Attention

    Args:
        embed_dim: The embedding dimension.
        nums_heads: The number of heads.
        dropout_prob: The dropout probability.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        nums_heads: int = 8,
        dropout_prob: float = 0.1,
        rotary_enc_max_len: int = 10000,
    ):
        super(RotaryPositionMultiHeadedAttention, self).__init__()
        self.embed_dim = embed_dim
        self.nums_heads = nums_heads
        assert (
            embed_dim % nums_heads == 0
        ), "embed_dim should be divisible by nums_heads"
        self.dim_per_head = embed_dim // nums_heads
        self.dropout_prob = dropout_prob
        self.temperature = math.sqrt(self.dim_per_head)

        self.rotary_pos_embed = RotaryPositionEncoding(
            dim=self.dim_per_head,
            max_len=rotary_enc_max_len,
        )

        self.w_q = nn.Linear(embed_dim, embed_dim)
        self.w_k = nn.Linear(embed_dim, embed_dim)
        self.w_v = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout_prob)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        offset: int = 0,
    ):
        """Forward pass.

        Args:
            query: The query tensor. [batch, len_query, embed_dim]
            key: The key tensor. [batch, len_key, embed_dim]
            value: The value tensor. [batch, len_key, embed_dim]
            key_padding_mask: The key_padding_mask tensor. [batch, len_query, len_key] or [batch, 1, len_key]
            offset: The offset. Default: 0
        """
        # rotary positional encoding
        B, T, C = value.size()
        query = rearrange(query, "b l (h d) -> b l h d", h=self.nums_heads)
        # query: [B, T, H, C]
        key = rearrange(key, "b l (h d) -> b l h d", h=self.nums_heads)
        # key: [B, T, H, C]

        cos_pos, sin_pos = self.rotary_pos_embed(value, len_seq=T)
        query, key = rotary_pos_enc(query, key, cos_pos, sin_pos, offset=offset)

        # combine heads
        query = rearrange(query, "b l h d -> b l (h d)")
        # query: [B, T, C]
        key = rearrange(key, "b l h d -> b l (h d)")

        # scaled dot-product attention
        q = rearrange(self.w_q(query), "b l (h d) -> b h l d", h=self.nums_heads)
        # q: [B, H, T, C]
        k = rearrange(self.w_k(key), "b l (h d) -> b h l d", h=self.nums_heads)
        # k: [B, H, T, C]
        v = rearrange(self.w_v(value), "b l (h d) -> b h l d", h=self.nums_heads)
        # v: [B, H, T, C]

        scaled_dot_product = (
            torch.einsum("b h i q, b h j k -> b h i j", q, k) / self.temperature
        )
        # scaled_dot_product: [B, H, T1, T2]

        if key_padding_mask is not None:
            scaled_dot_product.masked_fill_(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf")
            )  # [B, H, T1, T2]

        attn = F.softmax(scaled_dot_product, dim=-1)
        attn = self.dropout(attn)

        context = torch.einsum("b h i j, b h j v -> b h i v", attn, v)
        # context: [B, H, T, C]
        context = rearrange(context, "b h t d -> b t (h d)")
        # context: [B, T, C]

        output = self.output_proj(context)

        return output


class XposMultiHeadedAttention(nn.Module):
    """xPos Multi-Headed Attention Module

    Args:
        embed_dim: int, dimension of embedding
        xpos_scale_base: int, base scale of xpos, default 512
        num_heads: int, number of heads, default 8
        dropout_prob: float, dropout probability, default 0.1
        self_attention: bool, whether self-attention, default True
        subln: bool, whether to apply pre layer normalization, default False
    """

    def __init__(
        self,
        embed_dim: int = 512,
        xpos_scale_base: int = 512,
        num_heads: int = 8,
        dropout_prob: int = 0.1,
        self_attention: bool = True,
        subln: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.dropout = dropout_prob

        self.self_attention = self_attention

        self.w_k = nn.Linear(embed_dim, embed_dim, bias=True)
        self.w_v = nn.Linear(embed_dim, embed_dim, bias=True)
        self.w_q = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.inner_attn_ln = (
            nn.LayerNorm(self.embed_dim) if subln and self.self_attention else None
        )
        self.dropout_module = torch.nn.Dropout(dropout_prob)
        self.xpos = (
            XPOS(self.head_dim, xpos_scale_base) if self.self_attention else None
        )

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w_k.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.w_v.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.w_q.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def attention_ops(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        key_padding_mask=None,
        attn_mask=None,
    ):
        """Attention operations.

        Args:
            q (Tensor): query, [B, T, C]
            k (Tensor): key, [B, T, C]
            v (Tensor): value, [B, T, C]
            key_padding_mask (_type_, optional): key padding [B, T]. Defaults to None.
            attn_mask (_type_, optional): attention mask [B T]. Defaults to None.

        Returns:
            attn: attention output, [B, T, C]
            attn_weights: attention weights, [B, T, T]
        """

        q *= self.scaling
        attn_weights = torch.bmm(q, k.transpose(1, 2))

        if attn_mask is not None:
            attn_weights = torch.nan_to_num(attn_weights)
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            attn_weights = rearrange(
                attn_weights, "(b h) t s -> b h t s", h=self.num_heads
            )
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )
            attn_weights = rearrange(attn_weights, "b h t s -> (b h) t s")

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(
            attn_weights
        )
        attn_probs = self.dropout_module(attn_weights)

        attn = torch.bmm(attn_probs, v)
        attn = rearrange(attn, "(b h) l d -> b l (h d)", h=self.num_heads)

        return attn, attn_weights

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor],
        attn_mask: Optional[Tensor] = None,
        offset=0,
    ):
        """Forward pass.

        Args:
            query (Tensor): query, [B, T, C]
            key (Tensor): key, [B, T, C]
            value (Tensor): value, [B, T, C]
            key_padding_mask (Optional[Tensor]): key padding [B, T]
            attn_mask (Optional[Tensor], optional): attention mask [B, T]. Defaults to None.
            offset (int, optional): offset. Defaults to 0.
        Returns:
            attn: attention output, [B, T, C]
        """
        bsz, tgt_len, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"

        key_bsz, src_len, _ = key.size()
        assert key_bsz == bsz, f"{query.size(), key.size()}"
        assert value is not None
        assert bsz, src_len == value.shape[:2]

        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)

        q = rearrange(q, "b l (h d) -> (b h) l d", h=self.num_heads)
        k = rearrange(k, "b l (h d) -> (b h) l d", h=self.num_heads)
        v = rearrange(v, "b l (h d) -> (b h) l d", h=self.num_heads)

        if self.xpos is not None:
            k = self.xpos(k, offset=0, downscale=True)
            q = self.xpos(q, offset=offset, downscale=False)

        attn, attn_weights = self.attention_ops(
            q,
            k,
            v,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
        )

        if self.inner_attn_ln is not None:
            attn = self.inner_attn_ln(attn)

        attn = self.out_proj(attn)

        return attn


class RelPositionMultiHeadedAttention(nn.Module):
    """
    Relative Position Encoding Multi-Headed Attention Module

    args:
        embed_dim: int, dimension of embedding
        nums_heads: int, number of heads
        dropout_prob: float, dropout probability
        zero_triu: bool, whether to zero the upper triangular part of the matrix

    Inputs:
        query: Tensor, shape [batch_size, len_q, embed_dim]
        key: Tensor, shape [batch_size, len_k, embed_dim]
        value: Tensor, shape [batch_size, len_v, embed_dim]
        pos_embed: Tensor, shape [batch_size, len_q, embed_dim]
        key_padding_mask: Tensor, shape [batch_size, len_q, len_k], or shape [batch_size, 1, len_k]

    """

    def __init__(
        self,
        embed_dim: int,
        nums_heads: int,
        dropout_prob: float = 0.1,
    ):
        super(RelPositionMultiHeadedAttention, self).__init__()
        assert (
            embed_dim % nums_heads == 0
        ), "embed_dim should be divisible by nums_heads"

        self.embed_dim = embed_dim
        self.dim_per_head = embed_dim // nums_heads
        self.nums_heads = nums_heads
        self.tempature = math.sqrt(self.dim_per_head)

        self.w_q = nn.Linear(embed_dim, embed_dim)
        self.w_k = nn.Linear(embed_dim, embed_dim)
        self.w_v = nn.Linear(embed_dim, embed_dim)
        self.w_pos = nn.Linear(embed_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(dropout_prob)
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.nums_heads, self.dim_per_head))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.nums_heads, self.dim_per_head))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.final_dropout = nn.Dropout(dropout_prob)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        pos_embed: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            query (Tensor): query [B, T, C]
            key (Tensor): key [B, T, C]
            value (Tensor): value [B, T, C]
            pos_embed (Tensor): positional embedding [B, T, C]
            key_padding_mask (Optional[Tensor], optional): key padding mask [B, T]. Defaults to None.

        Returns:
            context: attn output [B, T, C]
        """
        q = rearrange(self.w_q(query), "b l (h d) -> b l h d", h=self.nums_heads)
        # q: [batch_size, len_q, n_head, d_k]
        k = rearrange(self.w_k(key), "b l (h d) -> b h l d", h=self.nums_heads)
        # k: [batch_size, n_head, len_k, d_k]
        v = rearrange(self.w_v(value), "b l (h d) -> b h l d", h=self.nums_heads)
        # v: [batch_size, n_head, len_k, d_k]

        pos_embed = rearrange(
            self.w_pos(pos_embed), "b l (h d) -> b h l d", h=self.nums_heads
        )

        content_score = torch.matmul(
            (q + self.pos_bias_u).transpose(1, 2), k.transpose(2, 3)
        )
        # content_score: [batch_size, n_head, len_q, len_k]
        pos_score = torch.matmul(
            (q + self.pos_bias_v).transpose(1, 2), pos_embed.transpose(2, 3)
        )
        pos_score = self._relative_shift(pos_score)

        scores = (content_score + pos_score) / self.tempature

        if key_padding_mask is not None:
            scores = scores.masked_fill_(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),  # [batch_size, n_head, len_q, len_k], B x 1 x 1 x T
            )

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.einsum("b h i j, b h j v -> b h i v", attn, v)
        context = rearrange(context, "b h l d -> b l (h d)")
        context = self.out_proj(context)
        context = self.final_dropout(context)

        return context  # [batch_size, len_q, embed_dim]

    def _relative_shift(self, pos_score: Tensor) -> Tensor:
        batch_size, nums_heads, len_seq1, len_seq2 = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, nums_heads, len_seq1, 1)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)

        padded_pad_score = padded_pos_score.view(
            batch_size, nums_heads, len_seq2 + 1, len_seq1
        )
        pos_score = padded_pad_score[:, :, 1:].view_as(pos_score)

        return pos_score
