from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .embedding import VariateIndependentEmbedding
from .encoder import PATTBlock, TemporalDownsample
from .revin import RevIN


class PATT(nn.Module):
    def __init__(
        self,
        in_channels: int,
        horizon: int,
        d_model: int = 64,
        num_blocks: int = 4,
        num_heads: int = 8,
        ff_expand: int = 4,
        dropout: float = 0.1,
        num_samples: int = 12,
        anchor_rho: int = 2,
        use_lpu: bool = True,
        use_convffn: bool = True,
        attention_mode: str = "sparse",
        offset_mode: str = "learned",
        patch_size: int = 1,
        max_offset: float = 6.0,
        kernel_size: int = 3,
        downsample_between_blocks: bool = True,
        probabilistic: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.horizon = horizon
        self.d_model = d_model
        self.probabilistic = probabilistic

        self.revin = RevIN(in_channels)
        self.embedding = VariateIndependentEmbedding(patch_size=patch_size, d_model=d_model)
        self.blocks = nn.ModuleList([
            PATTBlock(
                d_model=d_model,
                num_heads=num_heads,
                ff_expand=ff_expand,
                dropout=dropout,
                num_samples=num_samples,
                rho=anchor_rho,
                use_lpu=use_lpu,
                use_convffn=use_convffn,
                attention_mode=attention_mode,
                offset_mode=offset_mode,
                max_offset=max_offset,
                kernel_size=kernel_size,
            )
            for _ in range(num_blocks)
        ])
        self.downsample_between_blocks = downsample_between_blocks
        self.downsample = TemporalDownsample(d_model) if downsample_between_blocks else nn.Identity()
        self.head_mean = nn.LazyLinear(horizon)
        self.head_logstd = nn.LazyLinear(horizon) if probabilistic else None

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # x: [B, C, T]
        x = self.revin(x, mode="norm")
        z, q = self.embedding(x)  # [B, C, Q, D]
        b, c, q, d = z.shape
        z = z.view(b * c, q, d)

        for bi, block in enumerate(self.blocks):
            z = block(z)
            if self.downsample_between_blocks and bi < len(self.blocks) - 1 and z.shape[1] >= 4:
                z = self.downsample(z)

        z = z.reshape(b, c, -1).mean(dim=1)
        mean = self.head_mean(z)
        out = {"mean": mean}
        if self.probabilistic:
            log_std = self.head_logstd(z).clamp(-5.0, 3.0)
            out["log_std"] = log_std
        return out

    def get_last_attention(self) -> dict[str, Any]:
        block = self.blocks[-1]
        return {
            "attention": getattr(block.attn, "last_attention", None),
            "positions": getattr(block.attn, "last_positions", None),
        }
