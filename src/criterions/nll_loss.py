import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss


class NLLCriterion(_Loss):
    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor, mean: Tensor, var: Tensor) -> Tensor:
        return (torch.log(var) + (input - mean).pow(2) / var).sum()
