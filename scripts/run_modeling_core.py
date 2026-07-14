#!/usr/bin/env python3
"""Run Layer 1 + Layer 2 pipeline without Jupyter (reproducible paper path)."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import modeling_lib as ml  # noqa: E402

# Paper-tuned defaults (override via environment).
PAPER_ENV = {
    "SYNTHETIC_GENERATOR_MODE": "latent",
    "LATENT_TO_LABEL_NOISE": "0.45",
    "LATENT_TO_FEATURE_NOISE": "1.05",
    "LAYER1_FEATURES": "clean",
    "COMPLIANCE_GRAIN": "part_month",
    "CRIT_PROB_SHARPEN": "0.88",
    "L2_NUM_LEAVES": "127",
    "N_PARTS": "3500",
    "LAYER2_SCOPE": "real_category_only",
}


def _apply_paper_defaults() -> None:
    for k, v in PAPER_ENV.items():
        os.environ.setdefault(k, v)


def run_pipeline(
    outputs_dir: Path,
    out_dir: Path,
    *,
    layer1_mode: str = "clean",
    fast: bool = False,
    skip_layer1: bool = False,
) -> dict:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_modeling_core")
    # RUN_SEED (default 42, matching the frozen paper config) controls both the train/test split
    # and model training stochasticity (GAT init, LightGBM random_state stays fixed via
    # LGBM_*_PARAMS -- only the split and GAT vary here). Overriding it is how
    # scripts/multi_seed_significance.py gets independent repeated runs for CI estimation across
    # runs/seeds, not just bootstrap-resampling one run's test set.
    run_seed = int(os.environ.get("RUN_SEED", "42"))
    ml.set_seed(run_seed)

    df, part_catalog, G, _, _ = ml.load_and_prepare_data(outputs_dir)
    n_parts = len(part_catalog)
    n_expected = int(os.environ.get("N_PARTS", str(n_parts)))
    if n_parts != n_expected:
        raise ValueError(f"part_catalog has {n_parts} parts but N_PARTS={n_expected}")

    layer1_feats = ml.get_layer1_classification_features(layer1_mode)
    # Layer 1 (criticality classification) always trains/evaluates on the full part-level split
    # below (train_indices/test_indices, derived straight from part_catalog) — it doesn't depend
    # on a DataCo category link. Only the Layer 2 (supplier/compliance) panel rows are scoped.
    train_parts, test_parts = ml.part_level_train_test_split(part_catalog, random_state=run_seed)
    df_train = ml.filter_layer2_scope(df[df["part_id"].isin(train_parts)].copy(), part_catalog)
    df_test = ml.filter_layer2_scope(df[df["part_id"].isin(test_parts)].copy(), part_catalog)

    print(
        f"COMPLIANCE_GRAIN={ml.get_compliance_grain()} | LAYER2_SCOPE={ml.get_layer2_scope()} | "
        f"train/test rows: {len(df_train)}/{len(df_test)} | "
        f"train/test parts: {df_train['part_id'].nunique()}/{df_test['part_id'].nunique()} | "
        f"failure rate: {df_train['compliance_failure'].mean():.4f} / {df_test['compliance_failure'].mean():.4f}"
    )

    part_order = sorted(G.nodes())
    part_to_idx = {p: i for i, p in enumerate(part_order)}
    X_tab, y_part, _ = ml.build_part_arrays(part_catalog, part_order, layer1_feats)
    train_indices = np.array([part_to_idx[p] for p in train_parts], dtype=int)
    test_indices = np.array([part_to_idx[p] for p in test_parts], dtype=int)
    edge_index_full = ml.nx_to_edge_index(G, part_order)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {"n_parts": n_parts}
    l1_metrics: dict | None = None

    b1 = ml.baseline_rule_abc_quantile(part_catalog, train_parts, test_parts)
    ml.save_json(out_dir / "baseline1_results.json", b1)
    summary["baseline1_f1_macro"] = float(b1["f1_macro"])

    scaler_tab = ml.StandardScaler()
    scaler_tab.fit(X_tab[train_indices])
    X_scaled_tab = scaler_tab.transform(X_tab)
    lgbm_tab, b2_metrics = ml.train_lgbm_multiclass(
        X_scaled_tab[train_indices],
        y_part[train_indices],
        X_scaled_tab[test_indices],
        y_part[test_indices],
    )
    ml.save_json(out_dir / "baseline2_results.json", b2_metrics)
    summary["baseline2_f1_macro"] = float(b2_metrics["f1_macro"])

    if not skip_layer1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        oof_splits = 2 if fast else 5
        gat_oof = 3 if fast else 80
        gat_final = 5 if fast else 200

        oof_probs, _ = ml.oof_layer1_gat_lgbm(
            part_order,
            X_tab,
            y_part,
            train_indices,
            edge_index_full,
            n_splits=oof_splits,
            gat_epochs_oof=gat_oof,
            device=device,
            random_state=run_seed,
        )
        scaler_gat, gat_final, lgbm_stack, _, _ = ml.fit_final_layer1_gat_lgbm(
            part_order,
            X_tab,
            y_part,
            train_indices,
            edge_index_full,
            gat_epochs_final=gat_final,
            device=device,
        )
        all_idx = np.arange(len(part_order), dtype=int)
        final_probs_all = ml.layer1_predict_tabular_gat(
            scaler_gat, gat_final, lgbm_stack, X_tab, edge_index_full, all_idx, device=device
        )
        y_pred = np.argmax(final_probs_all[test_indices], axis=1)
        l1_metrics = ml._multiclass_metrics_dict(y_part[test_indices], y_pred)
        ml.save_json(out_dir / "layer1_results.json", l1_metrics)
        summary["layer1_f1_macro"] = float(l1_metrics["f1_macro"])

        pi_train = df_train["part_id"].map(part_to_idx).to_numpy()
        train_crit_mat = oof_probs[pi_train]
        pi_test = df_test["part_id"].map(part_to_idx).to_numpy()
        test_crit_mat = final_probs_all[pi_test]
        with (out_dir / "layer1_bundle.pkl").open("wb") as f:
            pickle.dump(
                {
                    # Full per-part arrays (aligned to part_order), not pre-sliced to a
                    # LAYER2_SCOPE -- so --skip-layer1 can re-slice for whatever scope is active
                    # without retraining the GAT (30-90 min) just to compare LAYER2_SCOPE values.
                    "part_order": list(part_order),
                    "oof_probs": oof_probs,
                    "final_probs_all": final_probs_all,
                    "layer1_f1_macro": summary["layer1_f1_macro"],
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
    else:
        bundle = pickle.loads((out_dir / "layer1_bundle.pkl").read_bytes())
        summary["layer1_f1_macro"] = float(bundle.get("layer1_f1_macro", float("nan")))
        if "oof_probs" not in bundle:
            raise ValueError(
                "layer1_bundle.pkl was written by an older version that only cached pre-sliced "
                "crit-prob matrices for one LAYER2_SCOPE. Re-run once without --skip-layer1 to "
                "regenerate a scope-independent bundle."
            )
        bundle_part_order = bundle["part_order"]
        if bundle_part_order != list(part_order):
            raise ValueError("layer1_bundle.pkl part_order doesn't match the current part_catalog.")
        pi_train = df_train["part_id"].map(part_to_idx).to_numpy()
        train_crit_mat = bundle["oof_probs"][pi_train]
        pi_test = df_test["part_id"].map(part_to_idx).to_numpy()
        test_crit_mat = bundle["final_probs_all"][pi_test]
        final_probs_all = bundle["final_probs_all"]

    # Bootstrap CIs for the Layer 1 classification comparisons. All three prediction arrays must
    # be aligned to the SAME part order for a valid element-wise comparison -- use
    # part_order[test_indices] (the order X_tab/y_part/final_probs_all are already built in) as
    # the canonical order, reindexing baseline1's own (differently-ordered) output into it.
    test_part_ids_canonical = np.array(part_order)[test_indices]
    y_true_canonical = y_part[test_indices]
    y_pred_layer1_canonical = np.argmax(final_probs_all[test_indices], axis=1)
    y_pred_b2_canonical = lgbm_tab.predict(ml._lgbm_df(X_scaled_tab[test_indices]))
    b1_part_ids, _, b1_y_pred = ml.baseline_rule_abc_predictions(part_catalog, train_parts, test_parts)
    y_pred_b1_canonical = (
        pd.Series(b1_y_pred, index=b1_part_ids).reindex(test_part_ids_canonical).to_numpy()
    )
    l1_boot_df = ml.bootstrap_layer1_comparisons(
        test_part_ids_canonical,
        y_true_canonical,
        {
            "Baseline1_rule_price": y_pred_b1_canonical,
            "Baseline2_LGBM_tabular": y_pred_b2_canonical,
            "Layer1_LGBM_GAT": y_pred_layer1_canonical,
        },
        n_boot=int(os.environ.get("L2_BOOTSTRAP_N", "2000")),
    )
    l1_boot_df.to_csv(out_dir / "classification_comparison_bootstrap.csv", index=False)

    layer1_rows = [
        ml.classification_metrics_row("Baseline1_rule_price", b1),
        ml.classification_metrics_row("Baseline2_LGBM_tabular", b2_metrics),
    ]
    if l1_metrics is not None:
        layer1_rows.append(ml.classification_metrics_row("Layer1_LGBM_GAT", l1_metrics))
    elif (out_dir / "layer1_results.json").is_file():
        layer1_rows.append(
            ml.classification_metrics_row(
                "Layer1_LGBM_GAT",
                json.loads((out_dir / "layer1_results.json").read_text(encoding="utf-8")),
            )
        )
    l1_df = pd.DataFrame(layer1_rows)
    l1_df.to_csv(out_dir / "classification_comparison.csv", index=False)
    l1_df.to_csv(out_dir / "full_results_summary_layer1.csv", index=False)

    l2 = ml.run_layer2_evaluation(
        df_train,
        df_test,
        part_catalog,
        train_parts,
        train_crit_mat,
        test_crit_mat,
        layer1_mode,
        out_dir,
        random_state=run_seed + 1,
    )
    summary.update({k: v for k, v in l2.items() if k != "y_test" and k != "scores_test"})
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outputs-dir", type=Path, default=REPO / "outputs")
    ap.add_argument("--out-dir", type=Path, default=REPO / "outputs" / "modeling")
    ap.add_argument("--layer1-features", default=None, choices=("clean", "full"))
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--skip-layer1", action="store_true")
    ap.add_argument("--no-paper-defaults", action="store_true", help="Do not set tuned env defaults.")
    args = ap.parse_args()

    if not args.no_paper_defaults:
        _apply_paper_defaults()

    mode = args.layer1_features or os.environ.get("LAYER1_FEATURES", "clean")
    os.environ["LAYER1_FEATURES"] = mode
    if args.fast:
        os.environ["MODELING_FAST"] = "1"

    summary = run_pipeline(
        args.outputs_dir,
        args.out_dir,
        layer1_mode=mode,
        fast=args.fast,
        skip_layer1=args.skip_layer1,
    )
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
