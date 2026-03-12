# PATT: Patch-Agnostic Temporal Transformer for Tourism–Culture Forecasting

This repository packages the paper **“AI-Driven Insights into the Impact of Tourism on Local Cultures: A Machine Learning Approach”** as a modular research codebase.

It implements the paper’s forecasting pipeline around three public data sources:
- **Eurostat `tour_occ_nin2`** for regional tourism pressure,
- **Inside Airbnb** snapshots for city-scale short-term-rental (STR) micro-intensity, and
- **Yelp Open Dataset** reviews / business metadata for review-derived cultural-perception proxies.

The repository is organized to support:
- monthly panel construction,
- rolling-window forecasting,
- the **Patch-Agnostic Temporal Transformer (PATT)**,
- uncertainty-aware evaluation,
- lead–lag diagnostics, and
- ablation studies for the core architectural claims.

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

## Reproducibility boundary

This repository is designed to be **faithful to the paper’s method and experiment structure**, but the manuscript still leaves several dataset-specific choices open. In particular, exact one-to-one reproduction would still require the final paper authors to lock down:
- the exact city list used in each panel,
- the final city-to-NUTS concordance table,
- the exact review lexicon / prompting recipe for the Yelp-derived indices,
- the exact holiday / event indicator tables,
- the final inclusion and filtering rules for all panels.

So this repo should be understood as:
- a **complete implementation of the architecture and experiment engine**,
- a **plausible, paper-aligned preprocessing pipeline**, and
- a **public release skeleton ready for GitHub**.

---

## Repository layout

```text
patt-tourism-repo/
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
├── outputs/
├── scripts/
├── src/patt/
│   ├── baselines/
│   ├── data/
│   ├── evaluation/
│   ├── models/
│   ├── training/
│   └── utils/
└── tests/
```

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

## Quickstart with the included synthetic sample

The repository now includes a **tiny synthetic sample bundle** in `data/sample/` so that new users can inspect file formats without redistributing third-party datasets.

Sample contents include:
- a toy Eurostat-style region-month CSV,
- a toy Inside Airbnb city snapshot,
- toy Yelp business and review JSONL files,
- a sample city-to-region mapping, and
- a manifest describing each file.

Example preprocessing commands:

```bash
python scripts/prepare_eurostat.py   --input data/sample/raw/eurostat_sample.csv   --output data/sample/processed/eurostat_monthly.parquet

python scripts/prepare_airbnb.py   --input_dir data/sample/raw/inside_airbnb   --eurostat_panel data/sample/processed/eurostat_monthly.parquet   --city_region_map data/sample/city_region_map_sample.csv   --output data/sample/processed/airbnb_monthly.parquet

python scripts/prepare_yelp.py   --input_dir data/sample/raw/yelp   --airbnb_panel data/sample/processed/airbnb_monthly.parquet   --eurostat_panel data/sample/processed/eurostat_monthly.parquet   --city_region_map data/sample/city_region_map_sample.csv   --output data/sample/processed/yelp_monthly.parquet
```

The sample files are **illustrative only** and are not intended for model quality claims.

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

## Lead–lag diagnostics

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

## Public release checklist

Before publishing this as a final GitHub repository, update:
- repository owner / URL fields,
- real author names and ORCIDs in `CITATION.cff`,
- release tag and version number,
- paper DOI / journal metadata if the article is published,
- exact city lists and concordance tables used in the final study,
- any task-specific lexical resources used for the Yelp indices.

---

## Citation

A root-level `CITATION.cff` file is included so GitHub can expose a **“Cite this repository”** entry for the software release. Update its maintainer and publication metadata before the public release if you want the repository to cite a paper instead of the software artifact.
