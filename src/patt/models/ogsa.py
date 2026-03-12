from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


def linear_time_sample(sequence: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    # sequence: [B, H, Q, D]
    # positions: [B, H, Q, R] in absolute index space [0, Q-1]
    b, h, q, d = sequence.shape
    r = positions.shape[-1]
    low = positions.floor().clamp(0, q - 1).long()
    high = positions.ceil().clamp(0, q - 1).long()
    frac = (positions - low.float()).unsqueeze(-1)
    seq_flat = sequence.reshape(b * h, q, d)
    low_flat = low.reshape(b * h, q, r)
    high_flat = high.reshape(b * h, q, r)

    gather_low = torch.gather(
        seq_flat.unsqueeze(1).expand(-1, q, -1, -1),
        2,
        low_flat.unsqueeze(-1).expand(-1, -1, -1, d),
    )
    gather_high = torch.gather(
        seq_flat.unsqueeze(1).expand(-1, q, -1, -1),
        2,
        high_flat.unsqueeze(-1).expand(-1, -1, -1, d),
    )
    sampled = gather_low * (1.0 - frac.reshape(b * h, q, r, 1)) + gather_high * frac.reshape(b * h, q, r, 1)
    return sampled.reshape(b, h, q, r, d)


class OffsetGuidedSparseAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        num_samples: int = 12,
        rho: int = 2,
        max_offset: float = 6.0,
        dropout: float = 0.1,
        mode: Literal["sparse", "dense"] = "sparse",
        offset_mode: Literal["learned", "fixed", "random"] = "learned",
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.num_samples = num_samples
        self.rho = rho
        self.max_offset = max_offset
        self.mode = mode
        self.offset_mode = offset_mode

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.offset_dw = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.offset_pw = nn.Conv1d(d_model, num_heads * num_samples, kernel_size=1)
        self.rel_bias_scale = nn.Parameter(torch.tensor(0.1))
        self.last_attention = None
        self.last_positions = None

        fixed = torch.linspace(-1.0, 1.0, steps=num_samples).view(1, 1, 1, num_samples)
        self.register_buffer("fixed_pattern", fixed, persistent=False)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, q, d = x.shape
        return x.view(b, q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def _make_positions(self, q_tokens: torch.Tensor) -> torch.Tensor:
        b, h, q, _ = q_tokens.shape
        device = q_tokens.device
        base = torch.arange(q, device=device).view(1, 1, q, 1).float()
        anchor_pattern = torch.linspace(-self.rho, self.rho, steps=self.num_samples, device=device).view(1, 1, 1, self.num_samples)
        if self.offset_mode == "learned":
            feat = q_tokens.permute(0, 2, 1, 3).reshape(b, q, self.d_model).transpose(1, 2)
            offsets = self.offset_pw(F.gelu(self.offset_dw(feat)))
            offsets = offsets.transpose(1, 2).view(b, q, self.num_heads, self.num_samples).permute(0, 2, 1, 3)
            offsets = torch.tanh(offsets) * self.max_offset
        elif self.offset_mode == "fixed":
            offsets = self.fixed_pattern.to(device) * self.rho
            offsets = offsets.expand(b, h, q, self.num_samples)
        elif self.offset_mode == "random":
            offsets = (torch.rand(b, h, q, self.num_samples, device=device) * 2 - 1) * self.max_offset
        else:
            raise ValueError(f"Unknown offset mode: {self.offset_mode}")
        positions = (base + anchor_pattern + offsets).clamp(0, q - 1)
        return positions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, q, d = x.shape
        qh = self._split_heads(self.q_proj(x))
        kh = self._split_heads(self.k_proj(x))
        vh = self._split_heads(self.v_proj(x))

        if self.mode == "dense":
            scores = torch.matmul(qh, kh.transpose(-1, -2)) / math.sqrt(self.head_dim)
            attn = scores.softmax(dim=-1)
            attn = self.attn_drop(attn)
            out = torch.matmul(attn, vh)
            self.last_attention = attn.detach()
            self.last_positions = torch.arange(q, device=x.device).view(1, 1, 1, q).expand(b, self.num_heads, q, q).detach()
        else:
            positions = self._make_positions(qh)
            sampled_k = linear_time_sample(kh, positions)  # [B,H,Q,R,D]
            sampled_v = linear_time_sample(vh, positions)
            q_exp = qh.unsqueeze(-2)  # [B,H,Q,1,D]
            scores = (q_exp * sampled_k).sum(dim=-1) / math.sqrt(self.head_dim)
            rel = positions - torch.arange(q, device=x.device).view(1, 1, q, 1)
            scores = scores - self.rel_bias_scale * rel.abs()
            attn = scores.softmax(dim=-1)
            attn = self.attn_drop(attn)
            out = (attn.unsqueeze(-1) * sampled_v).sum(dim=-2)
            self.last_attention = attn.detach()
            self.last_positions = positions.detach()

        out = out.permute(0, 2, 1, 3).reshape(b, q, d)
        return self.o_proj(out)
