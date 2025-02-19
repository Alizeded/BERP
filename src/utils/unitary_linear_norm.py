# --------------------------- unitary linear normalization ---------------------------
from typing import Any

import torch


def unitary_norm(x: torch.Tensor) -> torch.Tensor:
    """
    Input: x: (B,), the original tensor

    Return: (B,), normalized tensor
    """
    if x.dim() == 1:
        normalized = (x - x.min()) / (x.max() - x.min())
    else:
        raise ValueError("Input must be 1D tensor")
    return normalized


def unitary_norm_inv(x: Any, lb: Any, ub: Any) -> Any:
    """
    Input: x: (B,) the normalized tensor
           ub: upper bound of the original tensor
           lb: lower bound of the original tensor

    Return: (B,)
    """
    inv_norm_mut = ub - lb
    inv_norm_add = lb
    x = x * inv_norm_mut + inv_norm_add
    return x
