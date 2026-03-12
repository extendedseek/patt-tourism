from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class FoldSplit:
    train_end: pd.Timestamp
    val_end: pd.Timestamp
    test_end: pd.Timestamp


def add_month_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_datetime(df[date_col])
    month = dt.dt.month
    df["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    return df


def zscore_by_group(
    df: pd.DataFrame,
    group_col: str,
    value_cols: list[str],
) -> pd.DataFrame:
    df = df.copy()
    for col in value_cols:
        mu = df.groupby(group_col)[col].transform("mean")
        sigma = df.groupby(group_col)[col].transform("std").replace(0, 1.0)
        df[col] = (df[col] - mu) / sigma
    return df


def seasonal_median_impute(
    series: pd.Series,
    dates: pd.Series,
) -> pd.Series:
    s = series.copy()
    dt = pd.to_datetime(dates)
    month = dt.dt.month
    medians = s.groupby(month).median()
    mask = s.isna()
    s.loc[mask] = month.loc[mask].map(medians)
    return s


def forward_fill_limit(series: pd.Series, limit: int = 1) -> pd.Series:
    return series.ffill(limit=limit)
