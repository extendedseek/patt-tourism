from __future__ import annotations

import argparse
import copy
from pathlib import Path

from patt.config import deep_update, load_config
from patt.utils.io import save_json


def flatten_variants(cfg: dict):
    ab = cfg.get("ablation", {})
    variants = []
    for i, override in enumerate(ab.get("overrides", [])):
        variant = copy.deepcopy({k: v for k, v in cfg.items() if k != "ablation"})
        variant = deep_update(variant, override)
        variant["output_dir"] = str(Path(variant["output_dir"]) / f"variant_{i}")
        variants.append(variant)
    return variants


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    variants = flatten_variants(cfg)
    out_dir = Path("outputs/ablations")
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, variant in enumerate(variants):
        save_json(variant, out_dir / f"variant_{i}.json")
    print(f"Prepared {len(variants)} ablation variants under {out_dir}.")
    print("Run each variant with scripts/train.py using the emitted JSON/YAML-equivalent settings.")


if __name__ == "__main__":
    main()
