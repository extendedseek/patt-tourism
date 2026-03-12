from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .common import add_month_features


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, np.nan)


def build_airbnb_city_month(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    df["month"] = df["snapshot_date"].dt.to_period("M").dt.to_timestamp()
    df["active"] = (df["availability_30"].fillna(0) > 0).astype(int)
    grouped = df.groupby(["city", "month"], as_index=False)

    out = grouped.agg(
        active_listings=("id", "nunique"),
        entire_n=("room_type", lambda s: (s == "Entire home/apt").sum()),
        occupancy=("availability_30", lambda s: np.mean(1 - s.fillna(30).clip(0, 30) / 30.0)),
        minstay=("minimum_nights", "mean"),
        multi_n=("calculated_host_listings_count", lambda s: (s.fillna(0) >= 2).sum()),
        tenure=("first_seen_months", "mean"),
        dwelling_stock=("dwelling_stock", "first"),
    )
    out["density"] = 1000.0 * _safe_div(out["active_listings"], out["dwelling_stock"]).fillna(0.0)
    out["entire_share"] = _safe_div(out["entire_n"], out["active_listings"]).fillna(0.0)
    out["multi_share"] = _safe_div(out["multi_n"], out["active_listings"]).fillna(0.0)
    out = add_month_features(out, "month")
    return out[["city", "month", "density", "entire_share", "occupancy", "minstay", "multi_share", "tenure", "month_sin", "month_cos"]]
