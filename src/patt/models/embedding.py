from __future__ import annotations

import torch
import torch.nn as nn


class VariateIndependentEmbedding(nn.Module):
    def __init__(self, patch_size: int, d_model: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size, d_model)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        # x: [B, C, T]
        b, c, t = x.shape
        p = self.patch_size
        if p > 1:
            pad = (p - (t % p)) % p
            if pad > 0:
                x = torch.nn.functional.pad(x, (0, pad))
            t2 = x.shape[-1]
            q = t2 // p
            x = x.view(b, c, q, p)
        else:
            q = t
            x = x.unsqueeze(-1)  # [B, C, T, 1]
        z = self.proj(x)  # [B, C, Q, D]
        return z, q
