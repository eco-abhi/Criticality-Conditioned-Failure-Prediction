#!/usr/bin/env python3
"""
Hyperparameter search (reviewer feedback item #4a: "Report hyperparameter search procedure
(grid search / random search / Bayesian optimization) and search space.")

Method: RANDOM SEARCH with a fixed seed (chosen over grid search for coverage-per-trial
efficiency at this budget, and over Bayesian optimization for simplicity and exact
reproducibility -- every trial's config is deterministic from the seed, no surrogate model to
serialize). All search happens on TRAINING data only:
  - Layer 1 tabular LGBM and Layer 2 binary LGBM: part-level k-fold CV on train parts.
  - GAT: an 80/20 fit/val split of train parts, using a REDUCED epoch count (30 vs the full 200)
    as a tractable proxy -- standard practice (cheap proxy search, then validate the winning
    config with full-epoch training in the normal pipeline).
The test set is never touched during search, only afterward when the winning configs are run
through the normal scripts/run_modeling_core.py pipeline for the reported numbers.

Search spaces (uniform/log-uniform sampling, N_TRIALS draws each):
  Layer 1 / Layer 2 LGBM:
    n_estimators   in {200, 300, 500, 800}
    learning_rate  ~ log-uniform(0.01, 0.2)
    num_leaves     in {15, 31, 63, 127}
    min_child_samples in {5, 10, 20, 40}
  GAT:
    hidden       in {32, 64, 128}
    embed_dim    in {16, 32, 64}
    dropout      ~ uniform(0.1, 0.5)
    lr           ~ log-uniform(0.001, 0.02)
    weight_decay ~ log-uniform(1e-5, 1e-3)

Run: uv run --group modeling python scripts/hyperparameter_search.py
Writes: outputs/modeling/hparam_search_{layer1_lgbm,layer2_lgbm,gat}.csv, best_hparams.json
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import modeling_lib as ml  # noqa: E402

N_TRIALS = 20
N_TRIALS_GAT = 12
CV_FOLDS = 5
SEARCH_SEED = 777


def sample_lgbm_config(rng: np.random.Generator) -> dict:
    return {
        "n_estimators": int(rng.choice([200, 300, 500, 800])),
        "learning_rate": float(np.exp(rng.uniform(np.log(0.01), np.log(0.2)))),
        "num_leaves": int(rng.choice([15, 31, 63, 127])),
        "min_child_samples": int(rng.choice([5, 10, 20, 40])),
    }


def search_layer1_lgbm(X_tab: np.ndarray, y_part: np.ndarray, train_indices: np.ndarray) -> pd.DataFrame:
    rng = np.random.default_rng(SEARCH_SEED)
    y_tr = y_part[train_indices]
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEARCH_SEED)
    rows = []
    for trial in range(N_TRIALS):
        cfg = sample_lgbm_config(rng)
        fold_scores = []
        for fit_rel, val_rel in skf.split(np.zeros(len(train_indices)), y_tr):
            fit_idx, val_idx = train_indices[fit_rel], train_indices[val_rel]
            scaler = StandardScaler().fit(X_tab[fit_idx])
            Xf, Xv = scaler.transform(X_tab[fit_idx]), scaler.transform(X_tab[val_idx])
            clf = LGBMClassifier(
                objective="multiclass", num_class=3, class_weight="balanced",
                random_state=42, verbosity=-1, n_jobs=1, **cfg,
            )
            clf.fit(ml._lgbm_df(Xf), y_part[fit_idx])
            pred = clf.predict(ml._lgbm_df(Xv))
            fold_scores.append(f1_score(y_part[val_idx], pred, average="macro", zero_division=0))
        rows.append({"trial": trial, **cfg, "cv_macro_f1_mean": float(np.mean(fold_scores)), "cv_macro_f1_std": float(np.std(fold_scores))})
        print(f"[layer1_lgbm] trial {trial+1}/{N_TRIALS} cfg={cfg} cv_macro_f1={rows[-1]['cv_macro_f1_mean']:.4f}")
    return pd.DataFrame(rows).sort_values("cv_macro_f1_mean", ascending=False).reset_index(drop=True)


def search_layer2_lgbm(df_train: pd.DataFrame, l2_cols: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(SEARCH_SEED + 1)
    y_tr_all = df_train["compliance_failure"].to_numpy(dtype=int)
    part_ids = df_train["part_id"].astype(str).to_numpy()
    unique_parts = df_train["part_id"].astype(str).unique()
    part_y = df_train.groupby("part_id")["compliance_failure"].max().reindex(unique_parts).to_numpy()
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEARCH_SEED)
    X_all = df_train[l2_cols].to_numpy(dtype=np.float64)
    rows = []
    for trial in range(N_TRIALS):
        cfg = sample_lgbm_config(rng)
        fold_scores = []
        for fit_rel, val_rel in skf.split(unique_parts, part_y):
            fit_parts, val_parts = set(unique_parts[fit_rel]), set(unique_parts[val_rel])
            fit_mask, val_mask = np.isin(part_ids, list(fit_parts)), np.isin(part_ids, list(val_parts))
            spw = int((y_tr_all[fit_mask] == 0).sum()) / max(int((y_tr_all[fit_mask] == 1).sum()), 1)
            clf = LGBMClassifier(objective="binary", random_state=42, verbosity=-1, n_jobs=1, scale_pos_weight=spw, **cfg)
            clf.fit(ml._lgbm_df(X_all[fit_mask]), y_tr_all[fit_mask])
            s = clf.predict_proba(ml._lgbm_df(X_all[val_mask]))[:, 1]
            fold_scores.append(average_precision_score(y_tr_all[val_mask], s))
        rows.append({"trial": trial, **cfg, "cv_auc_pr_mean": float(np.mean(fold_scores)), "cv_auc_pr_std": float(np.std(fold_scores))})
        print(f"[layer2_lgbm] trial {trial+1}/{N_TRIALS} cfg={cfg} cv_auc_pr={rows[-1]['cv_auc_pr_mean']:.4f}")
    return pd.DataFrame(rows).sort_values("cv_auc_pr_mean", ascending=False).reset_index(drop=True)


def search_gat(
    part_order: list[str], X_tab: np.ndarray, y_part: np.ndarray, train_indices: np.ndarray, edge_index_full: torch.Tensor
) -> pd.DataFrame:
    rng = np.random.default_rng(SEARCH_SEED + 2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = len(part_order)
    fit_idx, val_idx = train_test_split(train_indices, test_size=0.2, random_state=SEARCH_SEED, stratify=y_part[train_indices])
    rows = []
    for trial in range(N_TRIALS_GAT):
        cfg = {
            "hidden": int(rng.choice([32, 64, 128])),
            "embed_dim": int(rng.choice([16, 32, 64])),
            "dropout": float(rng.uniform(0.1, 0.5)),
            "lr": float(np.exp(rng.uniform(np.log(0.001), np.log(0.02)))),
            "weight_decay": float(np.exp(rng.uniform(np.log(1e-5), np.log(1e-3)))),
        }
        ml.set_seed(42)
        scaler = StandardScaler().fit(X_tab[fit_idx])
        X_scaled = scaler.transform(X_tab)
        x_t = torch.tensor(X_scaled, dtype=torch.float32)
        edge_fit = ml.filter_edges_both_endpoints_in(edge_index_full, fit_idx)
        train_mask = torch.zeros(N, dtype=torch.bool)
        train_mask[fit_idx] = True
        y_t = torch.tensor(y_part, dtype=torch.long)
        model, _ = ml.train_gat_classifier(
            x_t, edge_fit, y_t, train_mask, epochs=30, device=device,
            hidden=cfg["hidden"], embed_dim=cfg["embed_dim"], dropout=cfg["dropout"],
            lr=cfg["lr"], weight_decay=cfg["weight_decay"],
        )
        _, logits = ml.gat_embeddings_and_logits(model, x_t, edge_index_full, device=device)
        val_pred = np.argmax(logits[val_idx], axis=1)
        score = f1_score(y_part[val_idx], val_pred, average="macro", zero_division=0)
        rows.append({"trial": trial, **cfg, "val_macro_f1": float(score)})
        print(f"[gat] trial {trial+1}/{N_TRIALS_GAT} cfg={cfg} val_macro_f1={score:.4f}")
    return pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False).reset_index(drop=True)


def main() -> None:
    os.environ.setdefault("LAYER1_FEATURES", "clean")
    os.environ.setdefault("COMPLIANCE_GRAIN", "part_month")
    os.environ.setdefault("LAYER2_SCOPE", "real_category_only")
    os.environ.setdefault("N_PARTS", "3500")
    out_dir = REPO / "outputs" / "modeling"
    out_dir.mkdir(parents=True, exist_ok=True)

    df, part_catalog, G, _, _ = ml.load_and_prepare_data(REPO / "outputs")
    layer1_feats = ml.get_layer1_classification_features("clean")
    train_parts, test_parts = ml.part_level_train_test_split(part_catalog)
    part_order = sorted(G.nodes())
    part_to_idx = {p: i for i, p in enumerate(part_order)}
    X_tab, y_part, _ = ml.build_part_arrays(part_catalog, part_order, layer1_feats)
    train_indices = np.array([part_to_idx[p] for p in train_parts], dtype=int)
    edge_index_full = ml.nx_to_edge_index(G, part_order)

    print("=== Searching Layer 1 tabular LGBM ===")
    l1_results = search_layer1_lgbm(X_tab, y_part, train_indices)
    l1_results.to_csv(out_dir / "hparam_search_layer1_lgbm.csv", index=False)

    print("\n=== Searching Layer 2 binary LGBM ===")
    df_train_l2 = ml.filter_layer2_scope(df[df["part_id"].isin(train_parts)].copy(), part_catalog)
    bundle = pickle.loads((out_dir / "layer1_bundle.pkl").read_bytes())
    pi_train = df_train_l2["part_id"].map(part_to_idx).to_numpy()
    train_crit_mat = ml.sharpen_crit_probs(bundle["oof_probs"][pi_train], float(os.environ.get("CRIT_PROB_SHARPEN", "0.88")))
    df_train_l2 = ml.attach_crit_prob_matrix(df_train_l2, train_crit_mat)
    l2_cols = ml.get_layer2_model_feature_columns(layer1_mode="clean")
    l2_results = search_layer2_lgbm(df_train_l2, l2_cols)
    l2_results.to_csv(out_dir / "hparam_search_layer2_lgbm.csv", index=False)

    print("\n=== Searching GAT (30-epoch proxy) ===")
    gat_results = search_gat(part_order, X_tab, y_part, train_indices, edge_index_full)
    gat_results.to_csv(out_dir / "hparam_search_gat.csv", index=False)

    best = {
        "search_method": "random search, fixed seed, train-only CV/val (test set never touched)",
        "n_trials": {"layer1_lgbm": N_TRIALS, "layer2_lgbm": N_TRIALS, "gat": N_TRIALS_GAT},
        "layer1_lgbm_best": l1_results.iloc[0].to_dict(),
        "layer2_lgbm_best": l2_results.iloc[0].to_dict(),
        "gat_best": gat_results.iloc[0].to_dict(),
    }
    (out_dir / "best_hparams.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    print("\nWrote best_hparams.json:")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
