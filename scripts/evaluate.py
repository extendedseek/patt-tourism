from __future__ import annotations

import argparse

import pandas as pd
from torch.utils.data import DataLoader

from patt.config import load_config
from patt.data.dataset import WindowSpec, WindowedTimeSeriesDataset
from patt.data.panel_builder import load_panel, prepare_panel
from patt.models.patt import PATT
from patt.training.engine import evaluate
from patt.utils.io import load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ckpt = load_checkpoint(args.checkpoint, map_location="cpu")

    df = load_panel(cfg["data"]["parquet_path"])
    df = prepare_panel(df, cfg["data"]["date_col"], cfg["data"]["unit_col"])

    dates = pd.to_datetime(df[cfg["data"]["date_col"]]).sort_values().unique()
    dates = pd.Series(dates)
    val_cut = dates.iloc[int(len(dates) * (cfg["data"]["train_ratio"] + cfg["data"]["val_ratio"])) - 1]

    spec = WindowSpec(
        input_length=cfg["data"]["input_length"],
        horizon=cfg["data"]["horizon"],
        stride=cfg["data"]["stride"],
        seasonal_period=cfg["data"]["seasonal_period"],
    )
    ds = WindowedTimeSeriesDataset(
        df=df,
        unit_col=cfg["data"]["unit_col"],
        date_col=cfg["data"]["date_col"],
        target_col=cfg["data"]["target_col"],
        feature_cols=cfg["data"]["feature_cols"],
        mask_cols=cfg["data"]["mask_cols"],
        spec=spec,
        add_mask_channels=cfg["data"]["add_mask_channels"],
        forecast_start_min=val_cut + pd.offsets.MonthBegin(1),
    )
    if len(ds) == 0:
        raise ValueError("No evaluation windows found. Reduce input_length/horizon or provide longer histories.")
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=False)

    in_channels = len(cfg["data"]["feature_cols"]) + (len(cfg["data"]["mask_cols"]) if cfg["data"]["add_mask_channels"] else 0)
    model = PATT(in_channels=in_channels, horizon=cfg["data"]["horizon"], **cfg["model"])
    model.load_state_dict(ckpt["model_state"])
    metrics = evaluate(model, loader, cfg)
    print(metrics)


if __name__ == "__main__":
    main()
