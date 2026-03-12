# PATT: Patch-Agnostic Temporal Transformer for Tourism

This repository Contains the implementation of the paper **“AI-Driven Insights into the Impact of Tourism on Local Cultures: A Machine Learning Approach”**.

It implements the paper’s forecasting pipeline around three public data sources:
- **Eurostat `tour_occ_nin2`** for regional tourism pressure,
- **Inside Airbnb** snapshots for city-scale short-term-rental (STR) micro-intensity, and
- **Yelp Open Dataset** reviews / business metadata for review-derived cultural-perception proxies.

The repository is organized to support:
- monthly panel construction,
- rolling-window forecasting,
- the Patch-Agnostic Temporal Transformer (PATT),
- uncertainty-aware evaluation,
- lead–lag diagnostics, and
- ablation studies.

---

## What this repository implements

### Model
- **RevIN** for per-series reversible normalization
- **Variate-independent input handling** via feature-first sequence tensors
- **Local Perception Unit (LPU)** for short-range temporal denoising
- **Offset-Guided Sparse Attention (OGSA)** for event-selective long-range attention
- **ConvFFN** for local structure refinement after attention
- **Hierarchical encoder** with optional stride-2 downsampling
- **Gaussian output head** for probabilistic multi-horizon forecasting

### Data pipeline
- Eurostat tourism-pressure preprocessing
- Inside Airbnb city-month STR feature construction
- Yelp city-month cultural-perception index construction
- optional city-to-region reconciliation for mapped tourism pressure
- long-form panel expansion for target-wise forecasting
- rolling windows for train / validation / test evaluation

### Evaluation and diagnostics
- **MASE**
- **CRPS** under Gaussian predictive assumptions
- **lagged cross-modal correlation** (`r_max`, `tau*`-style analysis)
- **attention entropy** for temporal selectivity
- ablation switches for dense vs sparse attention, offset modes, `R`, `rho`, LPU, and ConvFFN

---

## Repository layout

```text
patt-tourism/
├── CITATION.cff
├── configs/
│   ├── base.yaml
│   ├── eurostat.yaml
│   ├── airbnb.yaml
│   ├── yelp.yaml
│   └── ablations/
├── data/
│   ├── README.md
│   └── sample/
├── scripts/
├── src/patt/
   ├── baselines/
   ├── data/
   ├── evaluation/
   ├── models/
   ├── training/
   └── utils/
```


---

## Mapping paper sections to code

| Paper component | Main files |
|---|---|
| Data-to-model interface | `src/patt/data/panel_builder.py`, `src/patt/data/dataset.py` |
| RevIN | `src/patt/models/revin.py` |
| LPU | `src/patt/models/lpu.py` |
| Offset-guided sparse attention | `src/patt/models/ogsa.py` |
| ConvFFN | `src/patt/models/convffn.py` |
| Hierarchical encoder | `src/patt/models/encoder.py` |
| PATT model | `src/patt/models/patt.py` |
| Metrics | `src/patt/evaluation/metrics.py` |
| Rolling-window training | `src/patt/training/engine.py`, `scripts/train.py` |
| Evaluation | `scripts/evaluate.py` |
| Lead–lag analysis | `scripts/compute_lead_lag.py` |
| Ablations | `scripts/run_ablation.py`, `configs/ablations/` |
---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

---

## Data sources

### 1) Eurostat
Use the official Eurostat tourism dataset `tour_occ_nin2` as the tourism-pressure source.

```bash
python scripts/download_eurostat.py   --dataset tour_occ_nin2   --out data/raw/eurostat.json
```

Then flatten / curate the response into a table with at least:
- `region_id`
- `month`
- `nights`
- `population`

and run:

```bash
python scripts/prepare_eurostat.py   --input data/raw/eurostat.csv   --output data/processed/eurostat_monthly.parquet
```

### 2) Inside Airbnb
Download city snapshots into `data/raw/inside_airbnb/<city>/`.

Expected listing-level fields for the included preparation script are:
- `id`
- `snapshot_date` or `last_scraped`
- `room_type`
- `availability_30`
- `minimum_nights`
- `calculated_host_listings_count`
- `first_seen_months` or `host_since`
- `dwelling_stock`

Then run:

```bash
python scripts/prepare_airbnb.py   --input_dir data/raw/inside_airbnb   --output data/processed/airbnb_monthly.parquet
```

Optional mapped Eurostat covariates:

```bash
python scripts/prepare_airbnb.py   --input_dir data/raw/inside_airbnb   --eurostat_panel data/processed/eurostat_monthly.parquet   --city_region_map data/city_region_map.csv   --output data/processed/airbnb_monthly.parquet
```

### 3) Yelp Open Dataset
Place the Yelp dataset files under `data/raw/yelp/`.

Expected files:
- `yelp_academic_dataset_business.json`
- `yelp_academic_dataset_review.json`

Then run:

```bash
python scripts/prepare_yelp.py   --input_dir data/raw/yelp   --output data/processed/yelp_monthly.parquet
```

Optional STR and tourism-pressure covariates:

```bash
python scripts/prepare_yelp.py   --input_dir data/raw/yelp   --airbnb_panel data/processed/airbnb_monthly.parquet   --eurostat_panel data/processed/eurostat_monthly.parquet   --city_region_map data/city_region_map.csv   --output data/processed/yelp_monthly.parquet
```

---

## Training

### Eurostat
```bash
python scripts/train.py --config configs/eurostat.yaml
```

### Airbnb
```bash
python scripts/train.py --config configs/airbnb.yaml
```

### Yelp
```bash
python scripts/train.py --config configs/yelp.yaml
```

---

## Evaluation

```bash
python scripts/evaluate.py   --config configs/eurostat.yaml   --checkpoint outputs/eurostat/best.pt
```

---

## Lead-lag diagnostics

```bash
python scripts/compute_lead_lag.py   --input data/processed/yelp_monthly.parquet   --unit_col city   --date_col month   --x_col density   --y_col crowding
```

---

## Ablations

Examples:

```bash
python scripts/run_ablation.py --config configs/ablations/dense_vs_sparse.yaml
python scripts/run_ablation.py --config configs/ablations/lpu.yaml
python scripts/run_ablation.py --config configs/ablations/sparsity_R.yaml
python scripts/run_ablation.py --config configs/ablations/rho.yaml
python scripts/run_ablation.py --config configs/ablations/offset_modes.yaml
```

---
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18980783.svg)](https://doi.org/10.5281/zenodo.18980783)
