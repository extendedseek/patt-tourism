from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from .common import add_month_features


def load_panel(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def prepare_panel(
    df: pd.DataFrame,
    date_col: str,
    unit_col: str,
    sort: bool = True,
) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    if sort:
        group_cols = [unit_col] + (["target_name"] if "target_name" in df.columns else [])
        df = df.sort_values(group_cols + [date_col]).reset_index(drop=True)
    df = add_month_features(df, date_col)
    return df


def expand_multi_target_panel(
    df: pd.DataFrame,
    unit_col: str,
    date_col: str,
    target_cols: Sequence[str],
    feature_cols: Sequence[str],
) -> pd.DataFrame:
    rows = []
    keep_cols = [c for c in feature_cols if c in df.columns]
    for target in target_cols:
        dedup_keep_cols = [c for c in keep_cols if c != target]
        cols = [unit_col, date_col, target] + dedup_keep_cols
        tmp = df[cols].copy()
        tmp["target_name"] = target
        tmp["target_value"] = tmp[target]
        rows.append(tmp.drop(columns=[target]))
    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values([unit_col, "target_name", date_col]).reset_index(drop=True)
    return out
