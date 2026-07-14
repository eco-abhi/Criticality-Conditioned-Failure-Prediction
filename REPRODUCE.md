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

**Do not report point estimates alone — every headline comparison below has a bootstrap CI/p-value AND a multiple-comparison-corrected p-value (`p_value_fdr_bh`, `p_value_bonferroni`); report the corrected ones, not the raw ones.** Layer 1 ran 3 simultaneous tests, Layer 2 ran 9 (3 comparisons × 3 metrics) — each family is corrected separately (see `modeling_lib.add_multiple_comparison_corrections`).

- **Layer 1** (`classification_comparison_bootstrap.csv`): tabular LGBM and LGBM+GAT both significantly beat the price-rule baseline, and this **survives correction** (BH-FDR adjusted p≈0.011, p≈0.003 in the reference run). **LGBM+GAT vs. plain tabular LGBM is NOT significant** (p≈0.74, unaffected by correction) — the graph/GAT component's added complexity is not currently earning a statistically detectable improvement over the simpler tabular-only model. State this explicitly if the paper's contribution leans on the graph architecture.
- **Layer 2** (`compliance_comparison_bootstrap.csv`): **no comparison survives multiple-comparison correction, at either `LAYER2_SCOPE`.** Before correction, `LAYER2_SCOPE=all`'s conditioned-vs-uniform Brier looked significant (raw p≈0.025) — this is exactly why the correction matters: BH-FDR adjusts it to ≈0.075, Bonferroni to ≈0.225, neither below 0.05. At `LAYER2_SCOPE=real_category_only` nothing was significant even before correction.
- **Prior-informed baseline** (`compliance_comparison.csv` now has a 4th `prior` row, alongside `uniform`): per reviewer feedback that a naive uniform (1/3,1/3,1/3) baseline inflates perceived conditioning gain, added a baseline using the TRUE empirical marginal class shares (0.20/0.30/0.50). Finding worth reporting explicitly: `prior` and `uniform` give **byte-identical** predictions here, because both are constant across every row and a constant feature carries zero split information for a tree-based model (LightGBM already absorbs the target base rate from training labels directly). This means the reviewer's specific concern does not materially apply to this pipeline — `conditioned`'s advantage over either flat baseline reflects genuine per-part-varying signal, not an inflated comparison. State this explicitly rather than silently substituting `prior` for `uniform` and hoping no one asks why the numbers didn't move.
- **Defensible framing**: **do not claim criticality-conditioning improves out-of-sample compliance prediction**, on either discrimination or calibration, at either scope — the LightGBM/bootstrap analysis is a null result across the board once corrected for the 9 simultaneous tests. Point estimates are directionally consistent with a small positive effect (see raw point_diff columns), and oracle's point estimate is consistently above conditioned/uniform, but neither clears a corrected significance bar.
- **Complementary finding (`panel_significance_test.json`, `scripts/panel_significance_test.py`)**: a cluster-robust panel logistic regression — testing IN-SAMPLE ASSOCIATION on the full panel, not out-of-sample prediction on a train/test split — DOES find a significant relationship: likelihood-ratio test for the crit_prob_* block, p=0.022 (N=101 clusters, 2424 rows). **This does not contradict the null above; it answers a different question.** Likely explanation: LightGBM already partially reconstructs criticality-correlated signal from other tabular features (lead_time_cv, bom_criticality_propagation_score, otd_oem_measured) even under "uniform," leaving less *incremental* out-of-sample value for explicit crit_prob_*, even though the underlying association is real. Report both results together with this explanation — do not cite only the favorable one, and do not conflate "significant association" with "improves predictive performance," which are genuinely different claims.
- **Do not** headline train-max-F1 on the full training panel (rare positives + threshold overfitting).
- **N caveat:** with `LAYER2_SCOPE=real_category_only`, Layer 2 runs on the ~100-120 parts with a genuine real DataCo category link, not the full 3500-part catalog — report this N explicitly; it's a smaller but fully-grounded result, not a like-for-like comparison with a full-catalog run (`LAYER2_SCOPE=all`). Also note for the panel regression: N=101 clusters is above the commonly-cited 50-cluster rule of thumb for cluster-robust SE asymptotics, but not comfortably large.
- **Tuning caveat:** the frozen config (`LATENT_TO_LABEL_NOISE`, `LATENT_TO_FEATURE_NOISE`, `CRIT_PROB_SHARPEN`) was searched to produce a conditioning effect (see "Tuned configuration" above) — disclose this search process regardless of which result you lead with.

## Retuning (optional)

```bash
uv run --group modeling python scripts/tune_conditioning.py --fast-only
```

Omit `--fast-only` to apply the best cell and run full `run_modeling_core` automatically.
