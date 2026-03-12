from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from patt.evaluation.interpretability import summarize_attention
from patt.evaluation.metrics import gaussian_crps, mase
from patt.training.loss import gaussian_nll
from patt.utils.io import save_checkpoint


def make_optimizer(model, cfg: dict):
    return torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])


def make_scheduler(optimizer, cfg: dict, total_steps: int):
    warmup = max(1, int(total_steps * cfg["warmup_ratio"]))
    def lr_lambda(step: int):
        if step < warmup:
            return float(step + 1) / float(warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _forward_loss(model, batch, device):
    x = batch["x"].to(device)
    y = batch["y"].to(device)
    out = model(x)
    if "log_std" in out:
        loss = gaussian_nll(out["mean"], out["log_std"], y)
        sigma = torch.exp(out["log_std"])
    else:
        loss = torch.nn.functional.l1_loss(out["mean"], y)
        sigma = torch.ones_like(out["mean"])
    return loss, out["mean"], sigma


def fit(model, train_loader, val_loader, cfg, output_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    optimizer = make_optimizer(model, cfg["train"])
    total_steps = cfg["train"]["epochs"] * max(1, len(train_loader))
    scheduler = make_scheduler(optimizer, cfg["train"], total_steps)
    scaler = torch.amp.GradScaler(device="cuda", enabled=bool(cfg["train"].get("amp", True) and device == "cuda"))

    best_val = float("inf")
    bad_epochs = 0
    best_path = Path(output_dir) / "best.pt"

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        train_losses = []
        for batch in tqdm(train_loader, desc=f"train {epoch}", leave=False):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, enabled=bool(cfg["train"].get("amp", True) and device == "cuda")):
                loss, _, _ = _forward_loss(model, batch, device)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_losses.append(float(loss.item()))

        val_metrics = evaluate(model, val_loader, cfg, device=device)
        if val_metrics["crps"] < best_val:
            best_val = val_metrics["crps"]
            bad_epochs = 0
            save_checkpoint(
                {
                    "model_state": model.state_dict(),
                    "cfg": cfg,
                    "val_metrics": val_metrics,
                },
                best_path,
            )
        else:
            bad_epochs += 1

        if bad_epochs >= cfg["train"]["patience"]:
            break
    return best_path


@torch.no_grad()
def evaluate(model, data_loader, cfg, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    y_true, y_pred, y_std, insamples = [], [], [], []
    for batch in data_loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        out = model(x)
        mu = out["mean"]
        sigma = torch.exp(out["log_std"]) if "log_std" in out else torch.ones_like(mu)
        y_true.append(y.cpu().numpy())
        y_pred.append(mu.cpu().numpy())
        y_std.append(sigma.cpu().numpy())
        insamples.append(batch["insample_target"].numpy())

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    y_std = np.concatenate(y_std, axis=0)
    insamples = np.concatenate(insamples, axis=0)

    mase_vals = [
        mase(y_true[i], y_pred[i], insamples[i], m=cfg["data"]["seasonal_period"])
        for i in range(y_true.shape[0])
    ]
    crps = gaussian_crps(y_true, y_pred, y_std)
    attn_summary = summarize_attention(model)
    return {
        "mase": float(np.mean(mase_vals)),
        "crps": float(crps),
        "entropy": attn_summary["entropy"],
    }
