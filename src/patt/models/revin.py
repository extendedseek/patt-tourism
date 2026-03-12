from __future__ import annotations

import torch
import torch.nn as nn


class RevIN(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-5, affine: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(1, num_channels, 1))
            self.beta = nn.Parameter(torch.zeros(1, num_channels, 1))
        self._cached_mean = None
        self._cached_std = None

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(self.eps)
            self._cached_mean = mean
            self._cached_std = std
            x = (x - mean) / std
            if self.affine:
                x = x * self.gamma + self.beta
            return x
        if mode == "denorm":
            if self.affine:
                x = (x - self.beta) / (self.gamma + self.eps)
            if self._cached_mean is None or self._cached_std is None:
                raise RuntimeError("RevIN statistics are missing. Call with mode='norm' first.")
            return x * self._cached_std + self._cached_mean
        raise ValueError(f"Unknown mode: {mode}")
