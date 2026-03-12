from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

try:
    from statsmodels.tsa.x13 import x13_arima_analysis
except Exception:  # pragma: no cover
    x13_arima_analysis = None


def stl_adjust(series: pd.Series, period: int = 12) -> tuple[pd.Series, pd.Series]:
    result = STL(series, period=period, robust=True).fit()
    adjusted = series - result.seasonal
    return adjusted, result.seasonal


def x13_adjust(series: pd.Series, x13_binary: str | None = None) -> tuple[pd.Series, pd.Series]:
    if x13_arima_analysis is None:
        raise RuntimeError("statsmodels x13 support is unavailable in this environment.")
    res = x13_arima_analysis(series, x12path=x13_binary, prefer_x13=True)
    adjusted = pd.Series(res.seasadj, index=series.index)
    seasonal = series - adjusted
    return adjusted, seasonal
