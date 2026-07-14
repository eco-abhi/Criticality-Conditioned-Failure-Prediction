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

These values were originally chosen via `scripts/tune_conditioning.py` so Layer 2 **conditioned** beats **uniform** on AUC-PR, while keeping Layer 1 in an honest evaluation band (price rule macro F1 ≈ 0.45–0.55). **This is a data-generating-process search, not a hyperparameter search** — be explicit about that in the paper. Bootstrap significance testing (added after tuning; see "What to report" below) shows the searched-for effect is narrower than the tuning objective implies: conditioning improves Brier (calibration) significantly only at `LAYER2_SCOPE=all`, not AUC-PR/F1, and nothing reaches significance at the honestly-scoped `real_category_only` N. Report the tuning process itself as a limitation, not just the config table.

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
| `outputs/modeling/classification_comparison_bootstrap.csv` | Layer 1: part-level bootstrap 95% CI + p-value for each pairwise macro-F1 comparison |
| `outputs/modeling/full_results_summary_layer1.csv` | Same as classification_comparison |
| `outputs/modeling/compliance_comparison.csv` | Layer 2 @ threshold 0.5 |
| `outputs/modeling/compliance_comparison_bootstrap.csv` | Layer 2: part-level bootstrap 95% CI + p-value for each pairwise comparison × {auc_pr, brier, f1@0.5} |
| `outputs/modeling/compliance_comparison_val_threshold.csv` | Threshold tuned on validation (0.05–0.50 search) |
| `outputs/modeling/compliance_comparison_business_thresholds.csv` | P/R/F1 at 0.10–0.25 |
| `outputs/modeling/business_value_simulation.csv` | Net value vs threshold |
| `outputs/modeling/modeling_manifest.json` | Layer 2 grain, sharpen, row counts |
| `outputs/modeling/layer2_*.json` | Per-model detail + by-class metrics |

Bootstrap CIs are part-level cluster bootstraps (`n_boot=2000` default, override via `L2_BOOTSTRAP_N`), not row-level — rows sharing a `part_id` in the part-month panel aren't independent, so row-level resampling would understate variance. See `modeling_lib.bootstrap_part_level_metric_diff`.

## What to report in the paper

**Do not report point estimates alone — every headline comparison below has a bootstrap CI/p-value; report those, not just the deltas.**

- **Layer 1** (`classification_comparison_bootstrap.csv`): tabular LGBM and LGBM+GAT both significantly beat the price-rule baseline (p≈0.007, p≈0.001 in the reference run). **LGBM+GAT vs. plain tabular LGBM is NOT significant** (p≈0.74) — the graph/GAT component's added complexity is not currently earning a statistically detectable improvement over the simpler tabular-only model. State this explicitly if the paper's contribution leans on the graph architecture.
- **Layer 2** (`compliance_comparison_bootstrap.csv`): at `LAYER2_SCOPE=real_category_only` (the honestly-scoped, real-category-linked subset), **no conditioned-vs-uniform or oracle-vs-uniform comparison reaches p<0.05** — this N is underpowered to confirm the effect. At `LAYER2_SCOPE=all` (full catalog, most supplier links are category-blind proxies), conditioned significantly beats uniform **on Brier only** (p≈0.025), not AUC-PR/F1; oracle significantly beats both on AUC-PR and F1 (p<0.05) — real, confirmed headroom from better criticality prediction, even though current conditioning doesn't yet capture much of it.
- **Defensible framing**: not "criticality-conditioning improves compliance prediction" (not supportable — null at the honest N, and null on discrimination even at full N). Instead: "conditioning measurably improves probability calibration at scale, and there is statistically significant headroom versus a criticality oracle" — narrower, but actually true.
- **Do not** headline train-max-F1 on the full training panel (rare positives + threshold overfitting).
- **N caveat:** with `LAYER2_SCOPE=real_category_only`, Layer 2 runs on the ~100-120 parts with a genuine real DataCo category link, not the full 3500-part catalog — report this N explicitly; it's a smaller but fully-grounded result, not a like-for-like comparison with a full-catalog run (`LAYER2_SCOPE=all`).
- **Tuning caveat:** the frozen config (`LATENT_TO_LABEL_NOISE`, `LATENT_TO_FEATURE_NOISE`, `CRIT_PROB_SHARPEN`) was searched to produce a conditioning effect (see "Tuned configuration" above) — disclose this search process, don't present the frozen config as an a priori choice.

## Retuning (optional)

```bash
uv run --group modeling python scripts/tune_conditioning.py --fast-only
```

Omit `--fast-only` to apply the best cell and run full `run_modeling_core` automatically.
