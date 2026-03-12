from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="tour_occ_nin2")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--url",
        default="https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data",
        help="Eurostat STATISTICS API base URL.",
    )
    args = parser.parse_args()

    url = f"{args.url}/{args.dataset}?format=JSON"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(resp.text, encoding="utf-8")
    print(f"Saved raw Eurostat response to {out_path}")


if __name__ == "__main__":
    main()
