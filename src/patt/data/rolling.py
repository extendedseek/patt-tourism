from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import pandas as pd


@dataclass
class RollingWindow:
    train_end: pd.Timestamp
    val_end: pd.Timestamp
    test_end: pd.Timestamp


def rolling_origin_splits(
    dates: pd.Series,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    step_months: int = 3,
    min_points: int = 48,
) -> list[RollingWindow]:
    unique_dates = pd.Series(sorted(pd.to_datetime(dates).unique()))
    n = len(unique_dates)
    if n < min_points:
        raise ValueError(f"Need at least {min_points} timestamps, found {n}.")
    train_n = int(n * train_ratio)
    val_n = int(n * val_ratio)
    test_n = n - train_n - val_n
    if test_n <= 0:
        raise ValueError("Split ratios leave no room for test data.")

    windows = []
    start = 0
    while train_n + val_n + test_n <= n:
        train_end = unique_dates.iloc[train_n - 1]
        val_end = unique_dates.iloc[train_n + val_n - 1]
        test_end = unique_dates.iloc[train_n + val_n + test_n - 1]
        windows.append(RollingWindow(train_end=train_end, val_end=val_end, test_end=test_end))
        train_n += step_months
        val_n += step_months
        if train_n + val_n + test_n > n:
            break
    return windows
