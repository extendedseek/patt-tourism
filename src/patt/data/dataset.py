from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class WindowSpec:
    input_length: int
    horizon: int
    stride: int = 1
    seasonal_period: int = 12


class WindowedTimeSeriesDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        unit_col: str,
        date_col: str,
        target_col: str,
        feature_cols: list[str],
        mask_cols: list[str] | None,
        spec: WindowSpec,
        add_mask_channels: bool = True,
        forecast_start_min: str | pd.Timestamp | None = None,
        forecast_start_max: str | pd.Timestamp | None = None,
    ) -> None:
        self.df = df.copy().sort_values([unit_col, date_col]).reset_index(drop=True)
        self.unit_col = unit_col
        self.date_col = date_col
        self.target_col = target_col
        self.feature_cols = feature_cols
        self.mask_cols = mask_cols or []
        self.spec = spec
        self.add_mask_channels = add_mask_channels
        self.forecast_start_min = pd.Timestamp(forecast_start_min) if forecast_start_min is not None else None
        self.forecast_start_max = pd.Timestamp(forecast_start_max) if forecast_start_max is not None else None
        self.samples: list[tuple[str, int]] = []
        self.groups: dict[str, pd.DataFrame] = {k: g.reset_index(drop=True) for k, g in self.df.groupby(unit_col)}
        for unit, g in self.groups.items():
            max_start = len(g) - spec.input_length - spec.horizon + 1
            for start in range(0, max_start, spec.stride):
                end_in = start + spec.input_length
                forecast_start = pd.Timestamp(g.loc[end_in, date_col])
                if self.forecast_start_min is not None and forecast_start < self.forecast_start_min:
                    continue
                if self.forecast_start_max is not None and forecast_start > self.forecast_start_max:
                    continue
                self.samples.append((unit, start))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        unit, start = self.samples[idx]
        g = self.groups[unit]
        end_in = start + self.spec.input_length
        end_out = end_in + self.spec.horizon

        x = g.loc[start:end_in - 1, self.feature_cols].to_numpy(dtype=np.float32).T
        y = g.loc[end_in:end_out - 1, self.target_col].to_numpy(dtype=np.float32)

        if self.add_mask_channels and self.mask_cols:
            m = g.loc[start:end_in - 1, self.mask_cols].to_numpy(dtype=np.float32).T
            x = np.concatenate([x, m], axis=0)

        insample = g.loc[start:end_in - 1, self.target_col].to_numpy(dtype=np.float32)
        forecast_start = pd.Timestamp(g.loc[end_in, self.date_col])
        return {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
            "insample_target": torch.from_numpy(insample),
            "unit": unit,
            "start_idx": start,
            "forecast_start": str(forecast_start),
        }
