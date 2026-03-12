from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

from patt.config import load_config
from patt.data.dataset import WindowSpec, WindowedTimeSeriesDataset
from patt.data.panel_builder import load_panel, prepare_panel
from patt.models.patt import PATT
from patt.training.engine import fit
from patt.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    df = load_panel(cfg["data"]["parquet_path"])
    df = prepare_panel(df, cfg["data"]["date_col"], cfg["data"]["unit_col"])

    dates = pd.to_datetime(df[cfg["data"]["date_col"]]).sort_values().unique()
    dates = pd.Series(dates)
    train_cut = dates.iloc[int(len(dates) * cfg["data"]["train_ratio"]) - 1]
    val_cut = dates.iloc[int(len(dates) * (cfg["data"]["train_ratio"] + cfg["data"]["val_ratio"])) - 1]

    spec = WindowSpec(
        input_length=cfg["data"]["input_length"],
        horizon=cfg["data"]["horizon"],
        stride=cfg["data"]["stride"],
        seasonal_period=cfg["data"]["seasonal_period"],
    )

    common_kwargs = dict(
        df=df,
        unit_col=cfg["data"]["unit_col"],
        date_col=cfg["data"]["date_col"],
        target_col=cfg["data"]["target_col"],
        feature_cols=cfg["data"]["feature_cols"],
        mask_cols=cfg["data"]["mask_cols"],
        spec=spec,
        add_mask_channels=cfg["data"]["add_mask_channels"],
    )

    train_ds = WindowedTimeSeriesDataset(**common_kwargs, forecast_start_max=train_cut)
    val_ds = WindowedTimeSeriesDataset(**common_kwargs, forecast_start_min=train_cut + pd.offsets.MonthBegin(1), forecast_start_max=val_cut)

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError(
            f"Insufficient windows after splitting. train={len(train_ds)}, val={len(val_ds)}. "
            "Reduce input_length/horizon or provide longer histories."
        )

    in_channels = len(cfg["data"]["feature_cols"]) + (len(cfg["data"]["mask_cols"]) if cfg["data"]["add_mask_channels"] else 0)

    model = PATT(
        in_channels=in_channels,
        horizon=cfg["data"]["horizon"],
        **cfg["model"],
    )

    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=cfg["train"]["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=cfg["train"]["num_workers"])

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = fit(model, train_loader, val_loader, cfg, output_dir)
    print(f"Best checkpoint saved to: {best_path}")


if __name__ == "__main__":
    main()
