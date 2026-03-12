from __future__ import annotations

import pandas as pd
import torch

from .metrics import normalized_attention_entropy


def summarize_attention(model) -> dict:
    info = model.get_last_attention()
    attn = info.get("attention", None)
    positions = info.get("positions", None)
    if attn is None:
        return {"entropy": None, "positions": None}
    return {
        "entropy": normalized_attention_entropy(attn),
        "positions": None if positions is None else positions.detach().cpu().numpy().tolist(),
    }
