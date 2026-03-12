from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from patt.data.airbnb import build_airbnb_city_month
from patt.data.panel_builder import expand_multi_target_panel


def _load_snapshot_file(path: Path, city_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["city"] = city_name
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--eurostat_panel", default=None, help="Optional processed Eurostat panel for mapped tourism pressure.")
    parser.add_argument("--city_region_map", default=None, help="Optional CSV with columns: city, region_id.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    frames = []
    for city_dir in sorted(p for p in input_dir.iterdir() if p.is_dir()):
        for csv_path in city_dir.glob("*.csv"):
            frames.append(_load_snapshot_file(csv_path, city_dir.name))

    if not frames:
        raise ValueError("No Airbnb CSV files found.")

    raw = pd.concat(frames, ignore_index=True)
    if "snapshot_date" not in raw.columns:
        if "last_scraped" in raw.columns:
            raw["snapshot_date"] = raw["last_scraped"]
        else:
            raise ValueError("Expected snapshot_date or last_scraped column.")
    if "first_seen_months" not in raw.columns:
        if "host_since" in raw.columns:
            raw["host_since"] = pd.to_datetime(raw["host_since"], errors="coerce")
            raw["snapshot_date"] = pd.to_datetime(raw["snapshot_date"], errors="coerce")
            raw["first_seen_months"] = (
                (raw["snapshot_date"].dt.year - raw["host_since"].dt.year) * 12
                + (raw["snapshot_date"].dt.month - raw["host_since"].dt.month)
            ).fillna(0)
        else:
            raw["first_seen_months"] = 0
    if "dwelling_stock" not in raw.columns:
        raw["dwelling_stock"] = 1_000_000  # replace with city-specific housing stock for exact replication

    processed = build_airbnb_city_month(raw)

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
        processed["mapped_nights_per_resident"] = processed.get("mapped_nights_per_resident", 0.0)
        processed["mapped_nights_per_resident"] = processed["mapped_nights_per_resident"].fillna(0.0)
    else:
        processed["mapped_nights_per_resident"] = 0.0

    target_cols = ["density", "entire_share", "occupancy", "minstay", "multi_share", "tenure"]
    long_df = expand_multi_target_panel(
        processed,
        unit_col="city",
        date_col="month",
        target_cols=target_cols,
        feature_cols=["mapped_nights_per_resident", "density", "entire_share", "occupancy", "minstay", "multi_share", "tenure", "month_sin", "month_cos"],
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if args.output.endswith(".csv"):
        long_df.to_csv(args.output, index=False)
    else:
        long_df.to_parquet(args.output, index=False)
    print(f"Saved processed Airbnb long-form panel to {args.output}")


if __name__ == "__main__":
    main()
