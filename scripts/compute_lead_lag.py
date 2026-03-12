from __future__ import annotations

import argparse

import pandas as pd

from patt.evaluation.metrics import lagged_correlation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--unit_col", required=True)
    parser.add_argument("--date_col", required=True)
    parser.add_argument("--x_col", required=True)
    parser.add_argument("--y_col", required=True)
    parser.add_argument("--max_lag", type=int, default=12)
    args = parser.parse_args()

    df = pd.read_parquet(args.input) if args.input.endswith(".parquet") else pd.read_csv(args.input)
    df[args.date_col] = pd.to_datetime(df[args.date_col])
    out = []
    for unit, g in df.sort_values([args.unit_col, args.date_col]).groupby(args.unit_col):
        stats = lagged_correlation(g[args.x_col], g[args.y_col], max_lag=args.max_lag, seasonal_adjust=True)
        out.append({args.unit_col: unit, **stats})
    print(pd.DataFrame(out).to_string(index=False))


if __name__ == "__main__":
    main()
