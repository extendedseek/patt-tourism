from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from patt.data.panel_builder import expand_multi_target_panel
from patt.data.yelp import build_yelp_city_month


def load_json_lines(path: Path, max_rows: int | None = None) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            rows.append(json.loads(line))
            if max_rows is not None and i + 1 >= max_rows:
                break
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_reviews", type=int, default=None)
    parser.add_argument("--airbnb_panel", default=None, help="Optional processed Airbnb panel to attach STR covariates.")
    parser.add_argument("--eurostat_panel", default=None, help="Optional processed Eurostat panel to attach tourism pressure.")
    parser.add_argument("--city_region_map", default=None, help="Optional CSV with columns: city, region_id.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    business_path = input_dir / "yelp_academic_dataset_business.json"
    review_path = input_dir / "yelp_academic_dataset_review.json"

    if not business_path.exists() or not review_path.exists():
        raise FileNotFoundError("Expected Yelp business and review JSON files in input_dir.")

    business_df = load_json_lines(business_path)
    review_df = load_json_lines(review_path, max_rows=args.max_reviews)
    processed = build_yelp_city_month(review_df, business_df)

    if args.airbnb_panel:
        airbnb = pd.read_parquet(args.airbnb_panel) if args.airbnb_panel.endswith(".parquet") else pd.read_csv(args.airbnb_panel)
        if "target_name" in airbnb.columns:
            # reduce long-form airbnb panel back to wide covariates
            wide = airbnb.pivot_table(index=["city", "month"], columns="target_name", values="target_value").reset_index()
            wide.columns.name = None
            airbnb = wide
        airbnb["month"] = pd.to_datetime(airbnb["month"])
        cols = [c for c in ["density", "entire_share", "occupancy", "multi_share"] if c in airbnb.columns]
        processed = processed.merge(airbnb[["city", "month"] + cols], on=["city", "month"], how="left", suffixes=("", "_airbnb"))
        for col in cols:
            if f"{col}_airbnb" in processed.columns:
                processed[col] = processed[f"{col}_airbnb"].fillna(processed[col])
                processed = processed.drop(columns=[f"{col}_airbnb"])

    if args.eurostat_panel and args.city_region_map:
        euro = pd.read_parquet(args.eurostat_panel) if args.eurostat_panel.endswith(".parquet") else pd.read_csv(args.eurostat_panel)
        cmap = pd.read_csv(args.city_region_map)
        euro["month"] = pd.to_datetime(euro["month"])
        processed = processed.merge(cmap, on="city", how="left")
        processed = processed.merge(
            euro[["region_id", "month", "nights_per_resident"]].rename(columns={"nights_per_resident": "mapped_nights_per_resident"}),
            on=["region_id", "month"],
            how="left",
        )

    for col in ["mapped_nights_per_resident", "density", "entire_share", "occupancy", "multi_share"]:
        if col not in processed.columns:
            processed[col] = 0.0
        else:
            processed[col] = processed[col].fillna(0.0)

    target_cols = ["authenticity", "commodification", "noise", "crowding"]
    long_df = expand_multi_target_panel(
        processed,
        unit_col="city",
        date_col="month",
        target_cols=target_cols,
        feature_cols=[
            "mapped_nights_per_resident",
            "density",
            "entire_share",
            "occupancy",
            "multi_share",
            "authenticity",
            "commodification",
            "noise",
            "crowding",
            "month_sin",
            "month_cos",
        ],
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if args.output.endswith(".csv"):
        long_df.to_csv(args.output, index=False)
    else:
        long_df.to_parquet(args.output, index=False)
    print(f"Saved processed Yelp long-form panel to {args.output}")


if __name__ == "__main__":
    main()
