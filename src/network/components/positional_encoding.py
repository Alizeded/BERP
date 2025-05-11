import math

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor


def fixed_pos_embedding(x):
    """Fixed positional embedding for sinusoidal positional encoding."""
    seq_len, dim = x.shape
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim) / dim))
    sinusoid_inp = torch.einsum(
        "i , j -> i j", torch.arange(0, seq_len, dtype=torch.float), inv_freq
    ).to(x)
    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)


def rotate_every_two(x):
    """Rotate the second dimension of a tensor by 90 degrees."""
    x1 = x[:, :, ::2]
    x2 = x[:, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')\


def duplicate_interleave(m):
    """
    A simple version of `torch.repeat_interleave` for duplicating a matrix while interleaving the copy.
    """
    dim0 = m.shape[0]
    m = m.view(-1, 1)  # flatten the matrix
    m = m.repeat(1, 2)  # repeat all elements into the 2nd dimension
    m = m.view(dim0, -1)  # reshape into a matrix, interleaving the copy
    return m


def apply_rotary_pos_emb(x, sin, cos, scale=1):
    sin, cos = map(lambda t: duplicate_interleave(t * scale), (sin, cos))  # noqa: C417
    # einsum notation for lambda t: repeat(t[offset:x.shape[1]+offset,:], "n d -> () n () (d j)", j=2)
    return (x * cos) + (rotate_every_two(x) * sin)


class XPOS(nn.Module):
    def __init__(self, head_dim, scale_base=512):
        super().__init__()
        self.head_dim = head_dim
        self.scale_base = scale_base
        self.register_buffer(
            "scale", (torch.arange(0, head_dim, 2) + 0.4 * head_dim) / (1.4 * head_dim)
        )

    def forward(self, x, offset=0, downscale=False):
        length = x.shape[1]
        min_pos = -(length + offset) // 2
        max_pos = length + offset + min_pos
        scale = (
            self.scale
            ** torch.arange(min_pos, max_pos, 1)
            .to(self.scale)
            .div(self.scale_base)[:, None]
        )
        sin, cos = fixed_pos_embedding(scale)

        if scale.shape[0] > length:
            scale = scale[-length:]
            sin = sin[-length:]
            cos = cos[-length:]

        if downscale:
            scale = 1 / scale

        x = apply_rotary_pos_emb(x, sin, cos, scale)
        return x


class RelPosEncoding(nn.Module):
    """
    Relative positional encoding module.
    Args:
        d_model: Embedding dimension.
        max_len: Maximum input length.
    """

    def __init__(self, d_model: int = 512, max_len: int = 5000) -> None:
        super().__init__()
        self.d_model = d_model
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x: Tensor) -> None:
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return

        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype) # type: ignore[assignment]

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x : Input tensor B X T X C
        Returns:
            torch.Tensor: Encoded tensor B X T X C
        """
        self.extend_pe(x)
        pos_emb = self.pe[
            :,
            self.pe.size(1) // 2 - x.size(1) + 1 : self.pe.size(1) // 2 + x.size(1), # type: ignore
        ]
        return pos_emb


class RotaryPositionEncoding(nn.Module):
    """Rotary Position Encoding

    :param dim: The dimension of the input.
    :param max_len: The maximum length of the input.

    Reference:
        https://blog.eleuther.ai/rotary-embeddings/
        https://github.com/facebookresearch/fairseq/blob/main/fairseq/modules/rotary_positional_embedding.py
        https://github.com/lucidrains/linear-attention-transformer/blob/master/linear_attention_transformer/linear_attention_transformer.py#L164
        paper: https://arxiv.org/pdf/2104.09864.pdf
    """

    def __init__(self, dim: int, max_len: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.inv_freq = 1.0 / (max_len ** (torch.arange(0, dim, 2).float() / dim))
        self.len_seq_cache = 0
        self.cosine_pos_cache = torch.empty(1, self.len_seq_cache, 1, dim)
        self.sine_pos_cache = torch.empty(1, self.len_seq_cache, 1, dim)

    def forward(self, x: Tensor, len_seq: int = 0) -> tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            x: The input tensor.
            len_seq: The length of the input sequence.
        """
        if len_seq > self.len_seq_cache:
            self.len_seq_cache = len_seq
            time_pos = torch.arange(len_seq, device=x.device).type_as(self.inv_freq)
            sine_inp = torch.einsum("i , j -> i j", time_pos, self.inv_freq)
            pos_emb = torch.cat((sine_inp, sine_inp), dim=-1).to(x.device)
            self.cosine_pos_cache = rearrange(pos_emb.cos(), "n d -> () n () d")
            self.sine_pos_cache = rearrange(pos_emb.sin(), "n d -> () n () d")

        return self.cosine_pos_cache, self.sine_pos_cache
