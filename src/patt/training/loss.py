from __future__ import annotations

import math

import torch


def gaussian_nll(mean: torch.Tensor, log_std: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    var = torch.exp(2.0 * log_std)
    return 0.5 * (((target - mean) ** 2) / var + 2.0 * log_std + math.log(2.0 * math.pi)).mean()
