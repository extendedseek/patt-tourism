from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # This script keeps preprocessing conservative because the paper does not
    # provide a complete raw-response schema. It expects a flat CSV/parquet
    # prepared from the Eurostat response, or a JSON file to be manually flattened.
    in_path = Path(args.input)
    if in_path.suffix.lower() == ".csv":
        df = pd.read_csv(in_path)
    elif in_path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(in_path)
    else:
        payload = json.loads(in_path.read_text(encoding="utf-8"))
        raise ValueError(
            "Raw Eurostat JSON needs flattening because dimension names vary by query. "
            "Convert it to CSV/parquet with columns such as region_id, year/month, nights, population."
        )

    required = {"region_id", "month", "nights", "population"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["month"] = pd.to_datetime(df["month"])
    df["nights_per_resident"] = df["nights"] / df["population"].replace(0, pd.NA)
    df["nights_per_resident"] = df["nights_per_resident"].fillna(0.0).map(lambda x: __import__("math").log1p(float(x)))
    df["growth_lag1"] = df.groupby("region_id")["nights_per_resident"].diff().fillna(0.0)
    month = df["month"].dt.month
    import numpy as np
    df["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if args.output.endswith(".csv"):
        df.to_csv(args.output, index=False)
    else:
        df.to_parquet(args.output, index=False)
    print(f"Saved processed Eurostat panel to {args.output}")


if __name__ == "__main__":
    main()
