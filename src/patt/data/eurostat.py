from __future__ import annotations

import pandas as pd

from .common import add_month_features


def prepare_eurostat_monthly(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "month" not in df.columns and "year" in df.columns:
        df["month"] = pd.to_datetime(df["year"].astype(str) + "-01-01")
    df["nights_per_resident"] = df["nights"] / df["population"].replace(0, pd.NA)
    df["nights_per_resident"] = df["nights_per_resident"].fillna(0.0)
    df["nights_per_resident"] = (df["nights_per_resident"] + 1.0).map(float).map(pd.np.log) if hasattr(pd, 'np') else df["nights_per_resident"].map(lambda x: __import__("math").log1p(x))
    df["growth_lag1"] = df.groupby("region_id")["nights_per_resident"].diff()
    df = add_month_features(df, "month")
    return df
