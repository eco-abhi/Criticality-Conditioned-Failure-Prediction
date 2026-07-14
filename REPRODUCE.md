# Reproducing paper results

This document describes how to regenerate the synthetic dataset and modeling tables used in the write-up. All paths are relative to the repository root.

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)
- DataCo CSV at `data/DataCoSupplyChainDataset.csv` (or set `DATACO_CSV_PATH`)

```bash
uv sync
uv sync --group modeling   # LightGBM, PyTorch, PyG, SHAP
```

## Tuned configuration (frozen for the paper)

These values were chosen so Layer 2 **conditioned** beats **uniform** on AUC-PR while keeping Layer 1 in an honest evaluation band (price rule macro F1 ≈ 0.45–0.55).

| Setting | Value | Meaning |
|---------|-------|---------|
| `N_PARTS` | **3500** | Catalog size (3500 avoids OOM on laptop GAT runs) |
| `SYNTHETIC_GENERATOR_MODE` | **latent** | ABC labels are noisy views of a latent score |
| `LATENT_TO_LABEL_NOISE` | **0.45** | Blur between A/B/C from latent |
| `LATENT_TO_FEATURE_NOISE` | **1.05** | Observable feature noise vs latent |
| `LAYER1_FEATURES` | **clean** | Drop label-linked BOM columns from Layer 1 |
| `COMPLIANCE_GRAIN` | **part_month** | One row per part × month (~84k rows at 3500 parts; default for paper) |
| `THRESHOLD_SEARCH_MIN` / `MAX` | **0.05** / **0.50** | Validation F1 tuning range (avoids ~0.005 “flag everything”) |
| `BUSINESS_THRESHOLDS` | **0.10,0.15,0.20,0.25** | Fixed ops thresholds for PR tradeoff CSV |
| `CRIT_PROB_SHARPEN` | **0.88** | Sharpens Layer 1 probs before Layer 2 (<1 = more decisive) |
| `L2_NUM_LEAVES` | **127** | Layer 2 LightGBM capacity |
| `LAYER2_SCOPE` | **real_category_only** | Restrict Layer 2 (supplier/compliance) rows to parts with a genuine real DataCo category link (~100-120 of 3500 parts) rather than the full catalog, most of which (UCI-sourced parts) falls back to a random within-category supplier-proxy assignment — see `data_dictionary.md`'s `real_category_link` entry for the UCI/DataCo category-vocabulary mismatch this addresses. Set to `all` to reproduce the full-catalog (larger-N, category-fallback-included) run instead. |

## One-command reproduction

```bash
uv run python scripts/reproduce_paper.py
```

This runs data generation + full modeling (no `MODELING_FAST`). Expect roughly 30–90 minutes depending on hardware.

## Step by step

### 1. Generate data

```bash
export SYNTHETIC_GENERATOR_MODE=latent
export LATENT_TO_LABEL_NOISE=0.45
export LATENT_TO_FEATURE_NOISE=1.05
export N_PARTS=3500

uv run python generate_synthetic_datasets.py
```

Check `outputs/run_manifest.json` for the recorded generator settings.

### 2. Train models (headless, all tables)

```bash
export LAYER1_FEATURES=clean
export COMPLIANCE_GRAIN=part_month
export CRIT_PROB_SHARPEN=0.88
export L2_NUM_LEAVES=127
export N_PARTS=3500
export LAYER2_SCOPE=real_category_only
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

uv run --group modeling python scripts/run_modeling_core.py
```

### 3. Optional: notebook (figures + SHAP)

```bash
# same env vars as above
uv run --group dev --group modeling jupyter nbconvert \
  --to notebook --execute modeling.ipynb --inplace \
  --ExecutePreprocessor.timeout=7200
```

## Main result files

| File | Content |
|------|---------|
| `outputs/modeling/classification_comparison.csv` | Layer 1: weighted/macro F1 + per-class A/B/C |
| `outputs/modeling/full_results_summary_layer1.csv` | Same as classification_comparison |
| `outputs/modeling/compliance_comparison.csv` | Layer 2 @ threshold 0.5 |
| `outputs/modeling/compliance_comparison_val_threshold.csv` | Threshold tuned on validation (0.05–0.50 search) |
| `outputs/modeling/compliance_comparison_business_thresholds.csv` | P/R/F1 at 0.10–0.25 |
| `outputs/modeling/business_value_simulation.csv` | Net value vs threshold |
| `outputs/modeling/modeling_manifest.json` | Layer 2 grain, sharpen, row counts |
| `outputs/modeling/layer2_*.json` | Per-model detail + by-class metrics |

## What to report in the paper

- **Layer 1:** Macro F1 for price rule → tabular LGBM → LGBM+GAT.
- **Layer 2:** AUC-PR and Brier at 0.5; conditioned vs uniform vs oracle.
- **Caveat:** Conditioning gain is **modest**; oracle shows headroom if criticality were perfect.
- **Do not** headline train-max-F1 on the full training panel (rare positives + threshold overfitting).
- **N caveat:** with `LAYER2_SCOPE=real_category_only`, Layer 2 runs on the ~100-120 parts with a genuine real DataCo category link, not the full 3500-part catalog — report this N explicitly; it's a smaller but fully-grounded result, not a like-for-like comparison with a full-catalog run (`LAYER2_SCOPE=all`).

## Retuning (optional)

```bash
uv run --group modeling python scripts/tune_conditioning.py --fast-only
```

Omit `--fast-only` to apply the best cell and run full `run_modeling_core` automatically.
