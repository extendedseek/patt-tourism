from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import add_month_features


AUTHENTICITY_WORDS = {"authentic", "local", "heritage", "traditional", "original", "family-run"}
COMMODIFICATION_WORDS = {"touristy", "chain", "generic", "souvenir", "overpriced", "instagrammable"}
NOISE_WORDS = {"loud", "noisy", "shouting", "chaotic", "disturbing"}
CROWDING_WORDS = {"crowded", "packed", "queue", "line", "wait", "busy"}


def _keyword_score(text: str, lexicon: set[str]) -> float:
    tokens = str(text).lower().split()
    if not tokens:
        return 0.0
    matches = sum(t.strip(".,!?;:'\"()[]{}") in lexicon for t in tokens)
    return matches / max(len(tokens), 1)


def build_yelp_city_month(reviews_df: pd.DataFrame, business_df: pd.DataFrame) -> pd.DataFrame:
    reviews_df = reviews_df.copy()
    business_df = business_df.copy()
    reviews_df["date"] = pd.to_datetime(reviews_df["date"])
    reviews_df["month"] = reviews_df["date"].dt.to_period("M").dt.to_timestamp()
    merged = reviews_df.merge(business_df[["business_id", "city"]], on="business_id", how="left")
    merged["authenticity"] = merged["text"].map(lambda x: _keyword_score(x, AUTHENTICITY_WORDS))
    merged["commodification"] = merged["text"].map(lambda x: _keyword_score(x, COMMODIFICATION_WORDS))
    merged["noise"] = merged["text"].map(lambda x: _keyword_score(x, NOISE_WORDS))
    merged["crowding"] = merged["text"].map(lambda x: _keyword_score(x, CROWDING_WORDS))
    out = merged.groupby(["city", "month"], as_index=False)[["authenticity", "commodification", "noise", "crowding"]].mean()
    out = add_month_features(out, "month")
    return out
