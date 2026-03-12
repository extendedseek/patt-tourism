from __future__ import annotations

import torch
import torch.nn as nn


class LocalPerceptionUnit(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.dw = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=padding, groups=d_model)
        self.pw = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [BC, Q, D]
        residual = x
        y = x.transpose(1, 2)
        y = self.dw(y)
        y = self.act(y)
        y = self.pw(y)
        y = self.drop(y).transpose(1, 2)
        return residual + y
