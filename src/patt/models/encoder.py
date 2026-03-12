from __future__ import annotations

import torch
import torch.nn as nn

from .convffn import ConvFFN, PlainFFN
from .lpu import LocalPerceptionUnit
from .ogsa import OffsetGuidedSparseAttention


class PATTBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ff_expand: int,
        dropout: float,
        num_samples: int,
        rho: int,
        use_lpu: bool,
        use_convffn: bool,
        attention_mode: str,
        offset_mode: str,
        max_offset: float,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.use_lpu = use_lpu
        self.lpu = LocalPerceptionUnit(d_model, kernel_size=kernel_size, dropout=dropout) if use_lpu else nn.Identity()
        self.attn = OffsetGuidedSparseAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_samples=num_samples,
            rho=rho,
            max_offset=max_offset,
            dropout=dropout,
            mode=attention_mode,
            offset_mode=offset_mode,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = ConvFFN(d_model, expand=ff_expand, kernel_size=kernel_size, dropout=dropout) if use_convffn else PlainFFN(d_model, expand=ff_expand, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lpu(x)
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ffn(x))
        return x


class TemporalDownsample(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x.transpose(1, 2))
        return y.transpose(1, 2)
