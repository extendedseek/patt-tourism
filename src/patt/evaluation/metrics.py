from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

from patt.data.seasonal import stl_adjust


def mase(y_true: np.ndarray, y_pred: np.ndarray, insample: np.ndarray, m: int = 12) -> float:
    denom = np.mean(np.abs(insample[m:] - insample[:-m])) if len(insample) > m else np.mean(np.abs(np.diff(insample)))
    denom = max(float(denom), 1e-8)
    return float(np.mean(np.abs(y_true - y_pred)) / denom)


def gaussian_crps(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    sigma = np.maximum(sigma, 1e-8)
    z = (y - mu) / sigma
    crps = sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
    return float(np.mean(crps))


def lagged_correlation(x: pd.Series, y: pd.Series, max_lag: int = 12, seasonal_adjust: bool = True) -> dict:
    x = pd.Series(x).astype(float)
    y = pd.Series(y).astype(float)
    if seasonal_adjust and len(x) >= 24 and len(y) >= 24:
        x, _ = stl_adjust(x, period=12)
        y, _ = stl_adjust(y, period=12)
    vals = []
    for lag in range(0, max_lag + 1):
        corr = pd.concat([x.shift(lag), y], axis=1).corr().iloc[0, 1]
        vals.append(corr)
    vals = np.array(vals, dtype=float)
    idx = int(np.nanargmax(np.abs(vals)))
    return {"r_max": float(vals[idx]), "tau_star": idx, "curve": vals.tolist()}


def normalized_attention_entropy(attn: torch.Tensor) -> float:
    # sparse: [B,H,Q,R] ; dense: [B,H,Q,Q]
    eps = 1e-12
    probs = attn.clamp_min(eps)
    k = probs.shape[-1]
    ent = -(probs * probs.log()).sum(dim=-1) / math.log(k)
    return float(ent.mean().item())


def bootstrap_ci(values: Iterable[float], alpha: float = 0.05) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return (np.nan, np.nan)
    lo = np.quantile(arr, alpha / 2)
    hi = np.quantile(arr, 1 - alpha / 2)
    return float(lo), float(hi)
