import torch
import scipy.stats as stats


def mae_calculator(x: torch.Tensor, y: torch.Tensor) -> float:
    return round(torch.mean(torch.abs(x - y)).item(), 4)


def corrcoef_calculator(x: torch.Tensor, y: torch.Tensor) -> float:
    return round(abs(stats.pearsonr(x, y)[0]), 4)


def std_error_calculator(x: torch.Tensor, y: torch.Tensor) -> float:
    return round(torch.std((x - y).abs()).item(), 4)
