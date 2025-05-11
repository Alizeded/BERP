import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import xformers.ops as xops  # type: ignore[import]
from einops import rearrange
from torch import Tensor

from .positional_encoding import (
    XPOS,
    ConditionalPositionEncoding,
    RotaryPositionEncoding,
)


def get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = nn.functional.pad(
        torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
    )
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


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
) -> tuple[Tensor, Tensor]:
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

    q, k = map(lambda x: x * cos_pos + rotate_half(x) * sin_pos, (q, k))  # noqa: C417

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
        super().__init__()
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

        self.w_q = nn.Linear(embed_dim, embed_dim, bias=True)
        self.w_k = nn.Linear(embed_dim, embed_dim, bias=True)
        self.w_v = nn.Linear(embed_dim, embed_dim, bias=True)

        if self.dropout_prob > 0.0:
            self.dropout = nn.Dropout(dropout_prob)
        else:
            self.register_module("dropout", None)

        self.output_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w_k.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.w_v.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.w_q.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

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
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                torch.finfo(scaled_dot_product.dtype).min,
            )  # [B, H, T1, T2]

        attn = F.softmax(scaled_dot_product, dim=-1)

        if self.dropout is not None:
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
        dropout_prob: float = 0.1,
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
        if self.dropout > 0.0:
            self.dropout_module = torch.nn.Dropout(dropout_prob)
        else:
            self.register_module("dropout_module", None)
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
                torch.finfo(attn_weights.dtype).min,
            )
            attn_weights = rearrange(attn_weights, "b h t s -> (b h) t s")

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(
            attn_weights
        )
        if self.dropout_module is not None:
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
        kv_dim: Optional[int] = None,
        num_key_value_heads: Optional[int] = None,
        dropout_prob: float = 0.1,
        group_query_attn: bool = False,
    ):
        super().__init__()
        assert (
            embed_dim % nums_heads == 0
        ), "embed_dim should be divisible by nums_heads"

        self.embed_dim = embed_dim
        self.dim_per_head = embed_dim // nums_heads
        self.nums_heads = nums_heads
        self.tempature = math.sqrt(self.dim_per_head)

        if num_key_value_heads is not None:
            if nums_heads < num_key_value_heads:
                raise ValueError(
                    f"`num_heads` must be greater than or equal to `num_key_value_heads` ({num_key_value_heads}), but is {nums_heads} instead."
                )

            if nums_heads % num_key_value_heads != 0:
                raise ValueError(
                    f"`num_heads` must be a multiple of `num_key_value_heads` ({num_key_value_heads}), but is {nums_heads} instead."
                )
            self.num_key_value_heads = num_key_value_heads

        self.kv_dim = kv_dim or embed_dim

        self.group_query_attn = group_query_attn

        if self.group_query_attn:
            num_query_groups = (
                nums_heads // self.num_key_value_heads
                if self.num_key_value_heads
                else 1
            )

            if num_query_groups * self.num_key_value_heads != nums_heads:
                raise ValueError(
                    f"nums_heads ({nums_heads}) should be divisible by num_key_value_heads ({self.num_key_value_heads})"
                )

            self.num_query_groups = num_query_groups

        self.kv_dim = kv_dim if kv_dim is not None else embed_dim

        self.w_q = nn.Linear(embed_dim, embed_dim, bias=True)
        self.w_k = nn.Linear(
            self.kv_dim,
            (
                self.dim_per_head * self.num_key_value_heads
                if self.num_key_value_heads is not None
                else self.nums_heads
            ),
            bias=True,
        )
        self.w_v = nn.Linear(
            self.kv_dim,
            (
                self.dim_per_head * self.num_key_value_heads
                if self.num_key_value_heads is not None
                else self.nums_heads
            ),
            bias=True,
        )
        self.w_pos = nn.Linear(embed_dim, embed_dim, bias=False)

        self.pos_bias_u = nn.Parameter(torch.Tensor(self.nums_heads, self.dim_per_head))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.nums_heads, self.dim_per_head))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        if dropout_prob > 0.0:
            self.final_dropout = nn.Dropout(dropout_prob)
            self.dropout_prob = dropout_prob
        else:
            self.register_module("final_dropout", None)
            self.dropout_prob = 0.0

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w_k.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.w_v.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.w_q.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        pos_embed: Tensor,
        key_padding_mask: Optional[Tensor],
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
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

        btz, tgt_len, embed_dim = query.size()

        key_bsz, src_len, _ = key.size()

        assert key_bsz == btz, f"{query.size(), key.size()}"
        assert value is not None
        assert btz, src_len == value.shape[:2]

        q = rearrange(self.w_q(query), "b l (h d) -> b l h d", h=self.nums_heads)
        # q: [batch_size, len_q, n_head, d_k]
        k = rearrange(self.w_k(key), "b l (h d) -> b l h d", h=self.nums_heads)
        # k: [batch_size, n_head, len_k, d_k]
        v = rearrange(self.w_v(value), "b l (h d) -> b l h d", h=self.nums_heads)
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

        q_scores = ((content_score + pos_score) / self.tempature).permute(
            0, 2, 1, 3
        )  # [btz, len_q, nums_heads, d]

        attn_output = self._attn_forward(
            query=q_scores,
            key=k,
            value=v,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            softmax_scale=self.tempature,
        )

        attn_output = rearrange(attn_output, "b t h d -> b t (h d)")
        attn_output = self.out_proj(attn_output)

        if self.final_dropout is not None:
            attn_output = self.final_dropout(attn_output)

        return attn_output

    def _relative_shift(self, pos_score: Tensor) -> Tensor:
        batch_size, nums_heads, len_seq1, len_seq2 = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, nums_heads, len_seq1, 1)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)

        padded_pad_score = padded_pos_score.view(
            batch_size, nums_heads, len_seq2 + 1, len_seq1
        )
        pos_score = padded_pad_score[:, :, 1:].view_as(pos_score)

        return pos_score

    def _attn_forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor],
        attn_mask: Optional[Tensor] = None,
        softmax_scale: Optional[float] = None,
    ) -> Tensor:
        if key_padding_mask is not None:
            # [btz, kv_seq_len] -> [btz, 1, 1, kv_seq_len]
            mask = key_padding_mask[:, None, None, :]

            # [btz, 1, 1, kv_seq_len] -> [btz, H, 1, kv_seq_len]
            mask = mask.expand(-1, query.size(1), -1, -1)

            if attn_mask is not None:
                # [btz, H, 1, kv_seq_len] + [[H], q_seq_len, kv_seq_len] -> [btz, H, q_seq_len, kv_seq_len]
                mask += attn_mask
        else:
            mask = None

        if self.group_query_attn:
            # rearrange query, key, value to group query attention as B x T x G x H x D, where G is the number of query groups
            q = rearrange(query, "b t (g h) d -> b t g h d", g=self.num_query_groups)
            k = rearrange(key, "b t (g h) d -> b t g h d", g=self.num_query_groups)
            v = rearrange(value, "b t (g h) d -> b t g h d", g=self.num_query_groups)

            # broadcast key and value to the number of query groups
            k = k.expand(q.size())
            v = v.expand(q.size())

            # broadcast key_padding_mask to the number of query groups
            mask = (
                mask.unsqueeze(2)
                .expand(-1, -1, self.num_query_groups, -1, -1)
                .logical_not()
            )  # [btz, H, kv_seq_len] -> [btz, H, G, 1, kv_seq_len]

            # materialize as -inf
            mask_ = torch.zeros_like(mask, device=query.device).masked_fill(
                mask, float("-inf")
            )

            attn_output = xops.memory_efficient_attention(
                query=q,
                key=k,
                value=v,
                attn_bias=mask_,
                p=self.dropout_prob,
                op=None,
                scale=softmax_scale,
            )  # [btz, kv_seq_len, H, head_dim]

        else:
            # materialize mask as -inf
            mask_ = torch.zeros_like(mask, device=query.device).masked_fill_(
                mask, float("-inf")
            )

            attn_output = xops.memory_efficient_attention(
                query=query,
                key=key,
                value=value,
                attn_bias=mask_,
                p=self.dropout_prob,
                scale=softmax_scale,
            )  # [btz, kv_seq_len, H, head_dim]

        return attn_output



class ConditionalMultiHeadedAttention(nn.Module):
    """Conditional Multi-Headed Attention Module

    Args:
        embed_dim: The embedding dimension.
        nums_heads: The number of heads.
        dropout_prob: The dropout probability.
        group_query_attn: Whether to group query attention. Default: False

    Inputs:
        query: Tensor, shape [batch_size, len_query, embed_dim]
        key: Tensor, shape [batch_size, len_key, embed_dim]
        value: Tensor, shape [batch_size, len_key, embed_dim]
        key_padding_mask: Tensor, shape [batch_size, len_key]
        attn_mask: Tensor, shape [batch_size, len_query, len_key]
    """

    def __init__(
        self,
        embed_dim: int,
        nums_heads: int,
        kv_dim: Optional[int] = None,
        num_key_value_heads: Optional[int] = None,
        dropout_prob: float = 0.1,
        cope_max_enc_len: int = 10000,
        group_query_attn: bool = False,
    ):
        super().__init__()

        assert (
            embed_dim % nums_heads == 0
        ), "embed_dim should be divisible by nums_heads"

        if num_key_value_heads is not None:
            if nums_heads < num_key_value_heads:
                raise ValueError(
                    f"'num_key_value_heads' ({num_key_value_heads}) should be less than 'nums_heads' ({nums_heads})"
                )
            if nums_heads % num_key_value_heads != 0:
                raise ValueError(
                    f"'num_key_value_heads' should be divisible by 'nums_heads' ({nums_heads})"
                )
            self.num_key_value_heads = num_key_value_heads

        self.embed_dim = embed_dim
        self.nums_heads = nums_heads
        self.temperature = math.sqrt(self.dim_per_head)

        if dropout_prob > 0.0:
            self.dropout = nn.Dropout(dropout_prob)
        else:
            self.register_module("dropout", None)

        head_dim = embed_dim // nums_heads

        self.group_query_attn = group_query_attn
        if self.group_query_attn:
            num_query_groups = (
                nums_heads // self.num_key_value_heads
                if self.num_key_value_heads is not None
                else 1
            )

            if num_query_groups * self.num_key_value_heads != nums_heads:
                raise ValueError(
                    f"nums_heads ({nums_heads}) should be divisible by num_key_value_heads ({self.num_key_value_heads})"
                )

            self.num_query_groups = num_query_groups

        self.kv_dim = kv_dim if kv_dim is not None else embed_dim

        self.w_q = nn.Linear(embed_dim, embed_dim, bias=True)
        self.w_k = nn.Linear(
            self.kv_dim,
            (
                head_dim * self.num_key_value_heads
                if self.num_key_value_heads is not None
                else self.nums_heads
            ),
            bias=True,
        )
        self.w_v = nn.Linear(
            self.kv_dim,
            (
                head_dim * self.num_key_value_heads
                if self.num_key_value_heads is not None
                else self.nums_heads
            ),
            bias=True,
        )
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.dropout = nn.Dropout(dropout_prob)

        self.cope = ConditionalPositionEncoding(
            head_dim=head_dim,
            max_len=cope_max_enc_len,
        )

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w_k.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.w_v.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.w_q.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ):
        """Forward pass.

        Args:
            query (Tensor): query, shape [batch_size, len_query, embed_dim]
            key (Tensor): key, shape [batch_size, len_key, embed_dim]
            value (Tensor): value, shape [batch_size, len_key, embed_dim]
            key_padding_mask (Optional[Tensor], optional): key padding mask, shape [batch_size, len_key]. Defaults to None.
            attn_mask (Optional[Tensor], optional): attention mask, shape [batch_size, len_query, len_key]. Defaults to None.
        """
        btz, tgt_len, embed_dim = query.size()

        key_bsz, src_len, _ = key.size()

        assert key_bsz == btz, f"{query.size(), key.size()}"
        assert value is not None
        assert btz, src_len == value.shape[:2]

        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)

        attn_output = self._attn_forward(
            query=q,
            key=k,
            value=v,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
        )

        attn_output = rearrange(attn_output, "b t h d -> b t (h d)")

        if self.dropout is not None:
            attn_output = self.dropout(attn_output)

        attn_output = self.out_proj(attn_output)

        return attn_output

    def _attn_forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor],
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass for attention.

        Args:
            query (Tensor): query, shape [batch_size, len_query, embed_dim]
            key (Tensor): key, shape [batch_size, len_key, embed_dim]
            value (Tensor): value, shape [batch_size, len_key, embed_dim]
            key_padding_mask (Optional[Tensor], optional): key padding mask, shape [batch_size, len_key]. Defaults to None.
            attn_mask (Optional[Tensor], optional): attention mask, shape [batch_size, len_query, len_key]. Defaults to None.
        """
        _, q_seq_len, _ = query.size()
        _, kv_seq_len, _ = key.size()

        q = rearrange(query, "b t (h d) -> b h t d", h=self.nums_heads)
        k = rearrange(key, "b t (h d) -> b h d t", h=self.num_key_value_heads)
        v = rearrange(value, "b t (h d) -> b h d t", h=self.num_key_value_heads)

        if self.group_query_attn:
            attn_output = self._sdp_group_query_attn(
                query=q,
                key=k,
                value=v,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )
        else:
            attn_output = self._sdpa_forward(
                query=q,
                key=k,
                value=v,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )

        return attn_output

    def _sdpa_forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor],
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        q, k, v = query, key, value
        btz, q_n_heads, q_seq_len, _ = q.size()
        _, _, _, kv_seq_len = k.size()

        score = (
            torch.einsum("b h i d, b h d j -> b h i j", q, k) / self.temperature
        )

        if attn_mask is not None and key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [btz, 1, 1, kv_seq_len]

            mask = mask.expand(-1, -1, q_seq_len, -1)  # [btz, 1, q_seq_len, kv_seq_len]

            mask += attn_mask.unsqueeze(1)  # [btz, 1, q_seq_len, kv_seq_len]

            score = score.masked_fill(
                mask[:, :, :q_seq_len, :kv_seq_len].to(torch.bool),
                torch.finfo(score.dtype).min,
            )

        elif key_padding_mask is not None and attn_mask is None:
            score = score.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                torch.finfo(score.dtype).min,
            )

        else:
            score = score

        attn_weights = score + self.cope(q, score)

        attn_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(
            attn_weights
        )

        if self.dropout is not None:
            attn_probs = self.dropout(attn_probs)

        attn_output = torch.einsum("b h i j, b h j v -> b h i v", attn_probs, v)

        return attn_output

    def _sdp_group_query_attn(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor],
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass for group query attention.

        Args:
            query (Tensor): query, shape [batch_size, nums_heads, len_query, head_dim]
            key (Tensor): key, shape [batch_size, nums_heads, head_dim, len_key]
            value (Tensor): value, shape [batch_size, nums_heads, len_key, head_dim]
            bias (Tensor): bias, shape [1, nums_heads, len_query, len_key]
            key_padding_mask (Optional[Tensor], optional): key padding mask, shape [batch_size, len_key]. Defaults to None.
            attn_mask (Optional[Tensor], optional): attention mask, shape [batch_size, len_query, len_key]. Defaults to None.

        Returns:
            attn_output: attention output, shape [batch_size, len_query, embed_dim]
        """

        btz, n_heads, q_seq_len, q_head_dim = query.size()
        _, k_n_heads, k_head_dim, k_seq_len = key.size()
        _, v_n_heads, v_seq_len, v_head_dim = value.size()

        assert (
            q_head_dim == k_head_dim == v_head_dim
        ), "head dim of query, key, value should be the same"

        assert (
            k_n_heads == v_n_heads == self.num_key_value_heads
        ), "key and value should have the same number of heads"

        q = rearrange(query, "b (g h) t d -> b g h t d", g=self.num_query_groups)

        # calculate the similarity score with broadcasted bias for group query attention
        similarity = torch.einsum(
            "b g h i d, b h d j -> b g h i j", q, key
        ) / self.temperature

        if attn_mask is not None and key_padding_mask is not None:
            mask = rearrange(
                key_padding_mask, "b t -> b () () () t"
            )  # [btz, 1, 1, 1, k_seq_len]

            attn_mask = rearrange(attn_mask, "b t1 t2 -> b () () t1 t2")
            mask += attn_mask  # [btz, 1, 1, q_seq_len, k_seq_len]

            similarity = similarity.masked_fill(
                mask[:, :, :q_seq_len, :k_seq_len].to(torch.bool), float("-inf")
            )

        elif key_padding_mask is not None and attn_mask is None:
            key_padding_mask_ = rearrange(key_padding_mask, "b t -> b () () () t")
            similarity = similarity.masked_fill(
                key_padding_mask_.to(torch.bool), torch.finfo(similarity.dtype).min
            )

        else:
            similarity = similarity

        cope_pos = self.cope(
            q, similarity
        )  # [btz, group, n_heads, q_seq_len, k_seq_len]

        attn_weights = similarity + cope_pos

        attn_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(
            attn_weights
        )

        if self.dropout is not None:
            attn_probs = self.dropout(attn_probs)

        attn_output = torch.einsum("b g h i j, b h j v -> b g h i v", attn_probs, value)

        attn_output = rearrange(attn_output, "b g h i v -> b i (g h) v")

        return attn_output
