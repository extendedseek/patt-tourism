from __future__ import annotations

import torch
import torch.nn as nn


class RLinear(nn.Module):
    def __init__(self, in_channels: int, input_length: int, horizon: int, probabilistic: bool = True) -> None:
        super().__init__()
        self.probabilistic = probabilistic
        self.linear = nn.Linear(in_channels * input_length, horizon)
        self.log_std = nn.Linear(in_channels * input_length, horizon) if probabilistic else None

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = x.flatten(start_dim=1)
        out = {"mean": self.linear(z)}
        if self.probabilistic:
            out["log_std"] = self.log_std(z).clamp(-5.0, 3.0)
        return out
