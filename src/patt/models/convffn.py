from __future__ import annotations

import torch
import torch.nn as nn


class ConvFFN(nn.Module):
    def __init__(self, d_model: int, expand: int = 4, kernel_size: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        hidden = d_model * expand
        self.fc1 = nn.Linear(d_model, hidden)
        self.dw = nn.Conv1d(hidden, hidden, kernel_size=kernel_size, padding=kernel_size // 2, groups=hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc1(x)
        y = self.act(y)
        y = y.transpose(1, 2)
        y = self.dw(y)
        y = y.transpose(1, 2)
        y = self.drop(y)
        y = self.fc2(y)
        y = self.drop(y)
        return y


class PlainFFN(nn.Module):
    def __init__(self, d_model: int, expand: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        hidden = d_model * expand
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
