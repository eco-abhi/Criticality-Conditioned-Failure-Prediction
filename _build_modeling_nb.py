"""Build modeling.ipynb — run from repo: uv run --group modeling python _build_modeling_nb.py"""

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parent
cells = []


def md(s: str):
    cells.append(new_markdown_cell(s))


def code(s: str):
    cells.append(new_code_cell(s))


md(
    """# Criticality-aware compliance modeling

This notebook trains and evaluates the two-layer framework (Layer 1: hybrid LightGBM + GAT for ABC criticality; Layer 2: LightGBM compliance risk with criticality conditioning).

**Dependencies:** install with `uv sync --group modeling` from the repo root (see [`pyproject.toml`](pyproject.toml)).

**PyTorch Geometric:** the `modeling` group includes `torch-geometric`. If wheels fail on your platform, see [PyG installation](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

### Methodology notes (from design review)

1. **BOM features and labels:** `bom_n_downstream_A_assemblies` and `bom_criticality_propagation_score` are computed in `bom_graph.py` using **true** `criticality_class` when the synthetic dataset was built. In **`LAYER1_FEATURES=full`**, Layer 1 therefore includes label-derived graph summaries—interpret as learning under **partially observed engineering exposure** consistent with the generator, not as pure ABC discovery from primitives alone. **`LAYER1_FEATURES=clean`** drops those two columns.

   **ABC score inputs vs observables:** In the generator, `criticality_class` is assigned from `abc_price_proxy` and `abc_demand_cv_proxy` before literature augmentation. Those columns are **not** in `CLASSIFICATION_FEATURES`, so tabular LGBM and the GAT stack cannot refit the quantile scoring function; they remain in `part_catalog.csv` for **Baseline 1** (price-quantile proxy) only—an operations team sees part behavior (lead times, suppliers, BOM topology), not the exact score inputs used at label construction.

2. **Inductive GAT training:** During each GAT training run, **only edges whose endpoints are both in the current training part set** are used, so gradients never flow through held-out parts. For embedding extraction and downstream tabular stacking we run one **full-graph** forward pass at evaluation time so nodes see realistic BOM connectivity (weights were still learned without test-part message paths during training).

3. **Layer 2 and stacking:** Criticality probabilities on **training** compliance rows use **5-fold out-of-fold** predictions from the Layer 1 stack to avoid optimistic bias. The final Layer 1 model is refit on **all** training parts; **test** rows use those final probabilities.

4. **Runtime / environment:** Set `MODELING_FAST=1` for a short smoke run (fewer GAT epochs and 2 OOF folds). Full paper runs should omit it. LightGBM is configured with `n_jobs=1` in code to reduce native crashes on some macOS / OpenMP stacks.

5. **Layer 1 feature mode:** Default **`LAYER1_FEATURES=clean`**. Layer 2 always keeps `bom_criticality_propagation_score` (see `LAYER2_BOM_EXTRA_FEATURES`).

6. **Layer 2 grain:** Default **`COMPLIANCE_GRAIN=part_month`** — one row per part × month (~84k rows at 3500 parts). Set `COMPLIANCE_GRAIN=part` for a collapsed part-level view only.

7. **Layer 2 conditioning:** `crit_prob_*` from Layer 1 (uniform / conditioned / oracle) plus `crit_prob_*_x_at_risk`. **`CRIT_PROB_SHARPEN`** (default **0.88**) slightly sharpens Layer 1 probabilities before Layer 2 (see [REPRODUCE.md](REPRODUCE.md) for frozen paper values).

8. **Metrics:** Headline = test **AUC-PR** and **Brier** at threshold **0.5**. Threshold tuning searches **0.05–0.50** only (`compliance_comparison_val_threshold.csv`). Precision–recall at **0.10–0.25** in `compliance_comparison_business_thresholds.csv` and Section 7 business value.
"""
)

md("## Section 1: Setup and imports")

code(
    r"""import importlib
import json
import pickle
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

_REPO = Path.cwd().resolve()
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import modeling_lib as ml
importlib.reload(ml)

OUT = _REPO / "outputs" / "modeling"
OUT.mkdir(parents=True, exist_ok=True)

ml.set_seed(42)

import os

_MODELING_FAST = os.environ.get("MODELING_FAST", "0") == "1"
if _MODELING_FAST:
    print("MODELING_FAST=1: reduced GAT epochs for smoke testing")

print("numpy", np.__version__)
print("pandas", pd.__version__)
print("torch", torch.__version__)
try:
    import torch_geometric

    print("torch_geometric", torch_geometric.__version__)
except Exception as e:
    print("torch_geometric import failed:", e)
try:
    import lightgbm

    print("lightgbm", lightgbm.__version__)
except Exception as e:
    print("lightgbm import failed:", e)
try:
    import shap

    print("shap", shap.__version__)
except Exception as e:
    print("shap import failed:", e)

import sklearn

print("sklearn", sklearn.__version__)

T_START = time.perf_counter()
"""
)

md(
    """## Section 2: Load and prepare data

Load the compliance panel at **part × month** granularity by default (`COMPLIANCE_GRAIN=part_month`, ~24 rows per part). Train/test split is **by part_id** (80/20, stratified on ABC) so all months for a part stay in the same fold.

**Paper defaults:** `N_PARTS=3500`, latent noise `LLN=0.45`, `LFN=1.05`, `CRIT_PROB_SHARPEN=0.88` — see `REPRODUCE.md`.
"""
)

code(
    r"""outputs_dir = _REPO / "outputs"
df, part_catalog, G, CLASS_FEATURES, COMPLIANCE_FEATURE_NAMES = ml.load_and_prepare_data(outputs_dir)

import os

# Effective mode: optional in-notebook override (avoids needing env set before kernel start).
# None = use os.environ["LAYER1_FEATURES"] if set, else "full".
LAYER1_FEATURES_OVERRIDE = None  # set to "clean" or "full" to force without restarting kernel

_ov = (LAYER1_FEATURES_OVERRIDE or "").strip().lower()
_layer1_mode = _ov if _ov in ("full", "clean") else os.environ.get("LAYER1_FEATURES", "clean").strip().lower()
if _layer1_mode not in ("full", "clean"):
    raise ValueError(f"Bad LAYER1_FEATURES (effective): {_layer1_mode!r}; use 'full' or 'clean'")

LAYER1_FEATS = ml.get_layer1_classification_features(_layer1_mode)
LAYER2_TABULAR = ml.get_layer2_tabular_features(_layer1_mode)
L2_FEATURE_COLS = ml.get_layer2_model_feature_columns(layer1_mode=_layer1_mode)
print("LAYER1_FEATURES (effective):", _layer1_mode, "| n_layer1_tabular:", len(LAYER1_FEATS))
print("Layer2 tabular:", len(LAYER2_TABULAR), "| full design matrix:", len(L2_FEATURE_COLS))

part_order = sorted(G.nodes())
assert set(part_order) == set(part_catalog["part_id"].astype(str)), "Graph nodes must match part_catalog"
N = len(part_order)
_n_expected = int(os.environ.get("N_PARTS", "3500"))
print("COMPLIANCE_GRAIN:", ml.get_compliance_grain())
assert N == _n_expected, (N, _n_expected, "Set N_PARTS env to match generate_synthetic_datasets.py --n-parts.")

train_parts, test_parts = ml.part_level_train_test_split(part_catalog, test_size=0.2, random_state=42)
train_set, test_set = set(train_parts), set(test_parts)
assert not train_set & test_set

# Layer 1 (criticality classification) trains/evaluates on the full part-level split above
# (train_indices/test_indices below) regardless of scope. Only the Layer 2 (supplier/compliance)
# panel rows are scoped — see modeling_lib.get_layer2_scope for why (UCI/DataCo category mismatch).
df_train = ml.filter_layer2_scope(df[df["part_id"].isin(train_parts)].copy(), part_catalog)
df_test = ml.filter_layer2_scope(df[df["part_id"].isin(test_parts)].copy(), part_catalog)
print("LAYER2_SCOPE:", ml.get_layer2_scope())
print("Panel rows train/test/total:", len(df_train), len(df_test), len(df))
print("Unique parts train/test (Layer 2 scope):", df_train["part_id"].nunique(), df_test["part_id"].nunique())

print("\nCriticality (part catalog, train):")
print(part_catalog[part_catalog["part_id"].isin(train_parts)]["criticality_class"].value_counts())
print("\nCriticality (part catalog, test):")
print(part_catalog[part_catalog["part_id"].isin(test_parts)]["criticality_class"].value_counts())
print("\nCompliance failure rate train:", df_train["compliance_failure"].mean())
print("Compliance failure rate test:", df_test["compliance_failure"].mean())

assert df_train["part_id"].nunique() <= len(train_parts)
assert df_test["part_id"].nunique() <= len(test_parts)

X_tab, y_part, _ = ml.build_part_arrays(part_catalog, part_order, LAYER1_FEATS)
assert X_tab.shape == (N, len(LAYER1_FEATS))
assert y_part.shape == (N,)
part_to_idx = {p: i for i, p in enumerate(part_order)}
train_indices = np.array([part_to_idx[p] for p in train_parts], dtype=int)
test_indices = np.array([part_to_idx[p] for p in test_parts], dtype=int)

edge_index_full = ml.nx_to_edge_index(G, part_order)
print("Edges (directed):", edge_index_full.shape[1])
"""
)

md(
    """## Section 3: Baseline 1 — Rule-based ABC (price quantiles)

Classify using **train-only** 20th and 50th percentiles of `abc_price_proxy` for 20/30/50-style buckets. Evaluated on **held-out test parts** only.

**Legacy vs latent generator.** When the synthetic catalog is built in **`SYNTHETIC_GENERATOR_MODE=legacy`**, labels are tied to the same price proxy used here, so Baseline 1 can look deceptively strong. In **`latent`** mode, `abc_price_proxy` is only a **noisy view** of the latent score, so **price is genuinely decoupled** from `criticality_class`; a realistic macro-F1 floor for this rule is often **~0.45–0.55**, not a legacy-style **~0.60–0.65**. The sanity check is **Baseline 1 meaningfully below Baseline 2**, not a fixed Baseline 1 target.
"""
)

code(
    r"""b1 = ml.baseline_rule_abc_quantile(part_catalog, train_parts, test_parts)
ml.save_json(OUT / "baseline1_results.json", b1)
print(json.dumps(b1, indent=2)[:2500])
"""
)

md(
    """## Section 4: Baseline 2 — LightGBM tabular (no graph)

Multiclass LightGBM on classification features only. **Scaler fit on training parts only**. SHAP on a training subsample.
"""
)

code(
    r"""scaler_tab = ml.StandardScaler()
scaler_tab.fit(X_tab[train_indices])
X_scaled_tab = scaler_tab.transform(X_tab)

Xtr, Xte = X_scaled_tab[train_indices], X_scaled_tab[test_indices]
ytr, yte = y_part[train_indices], y_part[test_indices]

lgbm_tab, b2_metrics = ml.train_lgbm_multiclass(Xtr, ytr, Xte, yte)
with open(OUT / "lgbm_classifier.pkl", "wb") as f:
    pickle.dump(lgbm_tab, f, protocol=pickle.HIGHEST_PROTOCOL)
ml.save_json(OUT / "baseline2_results.json", b2_metrics)

feat_names = list(LAYER1_FEATS)
ml.shap_bar_top_multiclass(lgbm_tab, Xtr, feat_names, OUT / "shap_baseline2_top10.png", max_display=10)
print(json.dumps(b2_metrics, indent=2)[:2500])
"""
)

md(
    """## Section 5: Layer 1 — LightGBM + GAT (inductive training)

**5a–5c:** GAT with masked CE on training nodes; **training edges** use only endpoints in the active training set (OOF fold or full train). **5d–5e:** Full-graph forward for embeddings, concat tabular + 32-dim emb, LightGBM stack. OOF uses `GAT_EPOCHS_OOF` for speed; final model uses 200 epochs.

**5f:** Classification comparison CSV.
"""
)

code(
    r"""device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OOF_SPLITS = 5 if not _MODELING_FAST else 2
GAT_EPOCHS_OOF = 80 if not _MODELING_FAST else 3
GAT_EPOCHS_FINAL = 200 if not _MODELING_FAST else 5

oof_probs, oof_losses = ml.oof_layer1_gat_lgbm(
    part_order,
    X_tab,
    y_part,
    train_indices,
    edge_index_full,
    n_splits=OOF_SPLITS,
    gat_epochs_oof=GAT_EPOCHS_OOF,
    device=device,
)
ml.plot_loss_curve(
    oof_losses[-1], OUT / "gat_training_loss_oof_last_fold.png", title="GAT loss (last OOF fold)"
)

scaler_gat, gat_final, lgbm_stack, gat_losses_final, _emb_final_unused = ml.fit_final_layer1_gat_lgbm(
    part_order,
    X_tab,
    y_part,
    train_indices,
    edge_index_full,
    gat_epochs_final=GAT_EPOCHS_FINAL,
    device=device,
)
ml.plot_loss_curve(
    gat_losses_final, OUT / "gat_training_loss_final.png", title="GAT training loss (final model)"
)

proba_test = ml.layer1_predict_tabular_gat(
    scaler_gat, gat_final, lgbm_stack, X_tab, edge_index_full, test_indices, device=device
)
y_pred_l1 = np.argmax(proba_test, axis=1)
layer1_metrics = ml.multiclass_metrics_dict(y_part[test_indices], y_pred_l1)
with open(OUT / "lgbm_gat_classifier.pkl", "wb") as f:
    pickle.dump({"scaler": scaler_gat, "gat": gat_final, "lgbm": lgbm_stack}, f, protocol=pickle.HIGHEST_PROTOCOL)
ml.save_json(OUT / "layer1_results.json", layer1_metrics)

X_scaled_all = scaler_gat.transform(X_tab)
emb_tr_full, _ = ml.gat_embeddings_and_logits(
    gat_final,
    torch.tensor(X_scaled_all, dtype=torch.float32),
    edge_index_full,
    device=device,
)
X_stack_tr = np.hstack([X_scaled_all[train_indices], emb_tr_full[train_indices]])
feat_stack = feat_names + [f"gat_emb_{i}" for i in range(32)]
ml.shap_bar_top_multiclass(
    lgbm_stack, X_stack_tr, feat_stack, OUT / "shap_layer1_stack_top10.png", max_display=10
)

cmp = pd.DataFrame(
    [
        ml.classification_metrics_row("Baseline1_rule_price", b1),
        ml.classification_metrics_row("Baseline2_LGBM_tabular", b2_metrics),
        ml.classification_metrics_row("Layer1_LGBM_GAT", layer1_metrics),
    ]
)
cmp.to_csv(OUT / "classification_comparison.csv", index=False)
cmp.to_csv(OUT / "full_results_summary_layer1.csv", index=False)
print(cmp.to_string(index=False))
"""
)

md(
    """## Section 6: Layer 2 — Compliance risk engine

Three ablations (same tabular + supplier features; differ only in criticality conditioning):

| Model | `crit_prob_*` source |
|-------|----------------------|
| **uniform** | (1/3, 1/3, 1/3) |
| **conditioned** | OOF Layer 1 on train rows; final Layer 1 on test rows |
| **oracle** | True-class one-hot (ceiling if Layer 1 were perfect) |

All use `crit_prob_*_x_at_risk` interactions (no true-class `is_*_x_at_risk`).

**Metrics:** test @ **0.5** (AUC-PR, Brier) plus validation-part threshold (`compliance_comparison_val_threshold.csv`). Artifacts written by `modeling_lib.run_layer2_evaluation`.
"""
)

code(
    r"""all_idx = np.arange(N, dtype=int)
final_part_probs_all = ml.layer1_predict_tabular_gat(
    scaler_gat, gat_final, lgbm_stack, X_tab, edge_index_full, all_idx, device=device
)

_crit_sharp = float(os.environ.get("CRIT_PROB_SHARPEN", "0.88"))
pi_train = df_train["part_id"].map(part_to_idx).to_numpy()
train_crit_mat = ml.sharpen_crit_probs(oof_probs[pi_train], _crit_sharp)
pi_test = df_test["part_id"].map(part_to_idx).to_numpy()
test_crit_mat = ml.sharpen_crit_probs(final_part_probs_all[pi_test], _crit_sharp)
print("CRIT_PROB_SHARPEN:", _crit_sharp)

l2_out = ml.run_layer2_evaluation(
    df_train,
    df_test,
    part_catalog,
    train_parts,
    train_crit_mat,
    test_crit_mat,
    _layer1_mode,
    OUT,
)

y_te = l2_out["y_test"]
s_uni_te = l2_out["scores_test"]["uniform"]
s_cond_te = l2_out["scores_test"]["conditioned"]
s_orac_te = l2_out["scores_test"]["oracle"]
m_uni = l2_out["metrics"]["uniform"]
m_cond = l2_out["metrics"]["conditioned"]
m_orac = l2_out["metrics"]["oracle"]
by_cond = l2_out["by_criticality"]["conditioned"]
by_uni = l2_out["by_criticality"]["uniform"]

m_base, m_full = m_uni, m_cond
s_base_te, s_full_te = s_uni_te, s_cond_te
by_base, by_full = by_uni, by_cond

delta_aucpr_by_crit = {
    c: by_cond[c]["auc_pr"] - by_uni[c]["auc_pr"] for c in by_cond if c in by_uni
}
best_crit_gain = max(delta_aucpr_by_crit, key=delta_aucpr_by_crit.get) if delta_aucpr_by_crit else None

print("=== Layer 2 @ 0.5 (written to compliance_comparison.csv) ===")
print(pd.read_csv(OUT / "compliance_comparison.csv").to_string(index=False))
print("Delta AUC-PR (conditioned - uniform):", l2_out["delta_auc_pr_cond_minus_uniform"])
print()
print("=== Validation threshold (search 0.05–0.50) ===")
print(pd.read_csv(OUT / "compliance_comparison_val_threshold.csv").to_string(index=False))
print()
print("=== Business thresholds 0.10–0.25 (precision–recall tradeoff) ===")
print(pd.read_csv(OUT / "compliance_comparison_business_thresholds.csv").to_string(index=False))

ml.plot_roc_pr_pair(
    y_te, s_uni_te, s_cond_te,
    "Uniform crit probs", "Layer1-conditioned",
    OUT / "compliance_roc.png", OUT / "compliance_pr.png",
)

tmp_plot = df_test.copy()
tmp_plot["risk_score"] = s_cond_te
ml.plot_score_by_crit(tmp_plot, "risk_score", OUT / "compliance_risk_by_criticality.png")
ml.plot_calibration(y_te, s_cond_te, OUT / "compliance_calibration.png")

# SHAP for conditioned model (refit once for explainability plots)
L2_COLS = ml.get_layer2_model_feature_columns(layer1_mode=_layer1_mode)
df_tr_f = ml.attach_crit_prob_matrix(df_train, train_crit_mat)
df_te_f = ml.attach_crit_prob_matrix(df_test, test_crit_mat)
y_tr = df_train["compliance_failure"].to_numpy(dtype=int)
from lightgbm import LGBMClassifier
spw = int((y_tr == 0).sum()) / max(int((y_tr == 1).sum()), 1)
l2_cond = LGBMClassifier(objective="binary", **ml._l2_classifier_params(spw))
l2_cond.fit(ml._lgbm_df(df_tr_f[L2_COLS].to_numpy(float)), y_tr)
ml.shap_summary_binary_top(
    l2_cond, df_te_f[L2_COLS].to_numpy(float), L2_COLS,
    OUT / "shap_layer2_top15.png", max_display=15, sample_size=min(2500, len(df_te_f)),
)

crit_te_arr = df_test["criticality_class"].astype(str).to_numpy()
for c in ["A", "B", "C"]:
    ok = ml.plot_calibration_uniform_vs_full_stratum(
        crit_te_arr, y_te, s_uni_te, s_cond_te, c,
        OUT / f"compliance_calibration_uniform_vs_full_stratum_{c}.png",
    )
    print(f"Stratum {c} calibration plot:", ok)
"""
)

md(
    """## Section 7: Business value simulation

Costs and horizon are **constants** (easy to edit). Baseline cost = sum of undetected failure costs for all failures; model cost = FN failure costs + FP intervention costs; **net value** = baseline − model cost. Thresholds: 0.10 … 0.25 on Layer 2 full risk score.
"""
)

code(
    r"""COST_A = 50_000.0
COST_B = 10_000.0
COST_C = 2_000.0
FP_COST = 500.0
HORIZON_MONTHS = 24
N_PARTS = int(N)

cost_by = {"A": COST_A, "B": COST_B, "C": COST_C}
thrs = [0.10, 0.15, 0.20, 0.25]
crit_te = df_test["criticality_class"].astype(str).to_numpy()

biz = ml.business_value_summary(crit_te, y_te, s_full_te, thrs, cost_by, FP_COST)
biz.to_csv(OUT / "business_value_simulation.csv", index=False)
print(biz.to_string(index=False))

plt.figure(figsize=(7, 4))
plt.plot(biz["threshold"], biz["net_value"], marker="o")
plt.xlabel("Decision threshold")
plt.ylabel("Net value (test panel)")
plt.title("Net value vs threshold (Layer 2 full, test)")
plt.axhline(0, color="k", ls="--", alpha=0.3)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "business_value_net_vs_threshold.png", dpi=150, bbox_inches="tight")
plt.close()

_ = HORIZON_MONTHS, N_PARTS  # scenario constants (paper text)
"""
)

md(
    """## Section 8: Results summary

Two **layer-specific** tables (no cross-layer `NaN` columns) are printed below and saved as:

- `outputs/modeling/full_results_summary_layer1.csv` — criticality classification metrics  
- `outputs/modeling/full_results_summary_layer2.csv` — compliance metrics (`auc_roc`, `auc_pr`, …)

A short **results-style** narrative for the paper draft follows.
"""
)

code(
    r"""def _float_metrics(d: dict, skip=("confusion_matrix",)) -> dict:
    return {k: v for k, v in d.items() if k not in skip and isinstance(v, (int, float))}


layer1_rows = [
    {"model": "Baseline1_rule", **_float_metrics(b1)},
    {"model": "Baseline2_LGBM", **_float_metrics(b2_metrics)},
    {"model": "Layer1_LGBM_GAT", **_float_metrics(layer1_metrics)},
]
full_summary_l1 = pd.DataFrame(layer1_rows)
full_summary_l1.to_csv(OUT / "full_results_summary_layer1.csv", index=False)

layer2_rows = [
    {"model": "uniform", **m_uni},
    {"model": "conditioned", **m_cond},
    {"model": "oracle", **m_orac},
]
full_summary_l2 = pd.DataFrame(layer2_rows)
full_summary_l2.to_csv(OUT / "full_results_summary_layer2.csv", index=False)

print("=== Layer 1 (criticality classification, held-out parts) ===")
print(full_summary_l1.to_string(index=False))
print()
print("=== Layer 2 (compliance, held-out parts) ===")
print(full_summary_l2.to_string(index=False))
print()

if best_crit_gain is None:
    best_crit_gain = "?"

draft = (
    "Key findings (draft):\\n"
    "(1) Moving from single-criterion price quantiles (Baseline 1) to tabular LightGBM (Baseline 2) "
    "improves weighted and macro F1 on held-out parts when the label is learnable from tabular/BOM signals.\\n"
    "(2) Adding GAT embeddings (Layer 1 full) can further separate classes versus tabular-only LightGBM; "
    "see classification_comparison.csv and full_results_summary_layer1.csv.\\n"
    "(3) Layer 2 uniform vs conditioned vs oracle @0.5: AUC-PR "
    + f"uniform {m_uni['auc_pr']:.4f}, conditioned {m_cond['auc_pr']:.4f}, oracle {m_orac['auc_pr']:.4f}. "
    "Oracle shows the ceiling if Layer 1 were perfect; compare validation-threshold CSV for F1.\\n"
    "(4) Stratum with the largest AUC-PR delta (conditioned minus uniform) on true criticality: "
    + str(best_crit_gain or "?")
    + " (delta can be negative if conditioning hurts on this run); see layer2_* JSON files.\\n"
)
print(draft)

elapsed = time.perf_counter() - T_START
print(f"Total notebook runtime (approx): {elapsed:.1f}s")
"""
)

nb = new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
)
nb["nbformat"] = 4
nb["nbformat_minor"] = 5
nbformat.write(nb, str(ROOT / "modeling.ipynb"))
print("Wrote", ROOT / "modeling.ipynb", "cells:", len(cells))
