#!/usr/bin/env python3
"""
Cluster-robust logistic regression + likelihood-ratio test for whether Layer 1's criticality
conditioning (crit_prob_A/B/C) has a statistically significant association with
compliance_failure, accounting for the part-month panel's clustering by part_id.

Why this, alongside the existing LightGBM + bootstrap/multi-seed analysis: that analysis tests
OUT-OF-SAMPLE predictive improvement (does conditioning improve held-out AUC-PR/F1/etc.). This
tests a different, complementary question: IS THERE A SIGNIFICANT ASSOCIATION between crit_prob_*
and the outcome at all, using the full available panel (not sacrificing power to a train/test
split), via a model well-suited to small-cluster-count panels. LightGBM + a random-intercept GLMM
were both considered and rejected/deferred: GLMMs are prone to convergence failure with ~101
clusters and an ~8% event rate; logistic regression with cluster-robust (sandwich) SEs is the
standard, numerically stable alternative for this scale.

Technical notes:
  - crit_prob_A + crit_prob_B + crit_prob_C = 1 by construction (softmax output) -- including all
    three in a regression is perfectly collinear. Following standard practice for compositional
    features, crit_prob_C is dropped as the reference level (only A, B enter the model). Same for
    the *_x_at_risk interaction terms (which sum to supplier_at_risk_flag, already in the model).
  - The likelihood-ratio test compares standard (non-cluster-adjusted) maximum-likelihood fits of
    the nested models -- clustering affects the STANDARD ERRORS used for per-coefficient Wald
    tests, not the likelihood itself. Both are reported.
  - Caveat: cluster-robust SE asymptotics are typically considered reliable with 50+ clusters;
    N=101 real-category-linked parts is above that rule of thumb but not large. Report this
    honestly alongside the result.

Run: uv run --group modeling python scripts/panel_significance_test.py
Writes: outputs/modeling/panel_significance_test.json
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import modeling_lib as ml  # noqa: E402


def main() -> None:
    outputs_dir = REPO / "outputs"
    out_dir = REPO / "outputs" / "modeling"

    df, part_catalog, G, _, _ = ml.load_and_prepare_data(outputs_dir)
    train_parts, test_parts = ml.part_level_train_test_split(part_catalog)
    df_train = ml.filter_layer2_scope(df[df["part_id"].isin(train_parts)].copy(), part_catalog)
    df_test = ml.filter_layer2_scope(df[df["part_id"].isin(test_parts)].copy(), part_catalog)

    bundle = pickle.loads((out_dir / "layer1_bundle.pkl").read_bytes())
    part_order = bundle["part_order"]
    part_to_idx = {p: i for i, p in enumerate(part_order)}
    temp = float(__import__("os").environ.get("CRIT_PROB_SHARPEN", "0.88"))

    pi_train = df_train["part_id"].map(part_to_idx).to_numpy()
    train_crit_mat = ml.sharpen_crit_probs(bundle["oof_probs"][pi_train], temp)
    pi_test = df_test["part_id"].map(part_to_idx).to_numpy()
    test_crit_mat = ml.sharpen_crit_probs(bundle["final_probs_all"][pi_test], temp)

    df_tr = ml.attach_crit_prob_matrix(df_train, train_crit_mat)
    df_te = ml.attach_crit_prob_matrix(df_test, test_crit_mat)
    panel = pd.concat([df_tr, df_te], ignore_index=True)

    tabular_features = ml.get_layer2_tabular_features(mode="clean")
    # Drop one level each of the two collinear compositional blocks (see module docstring).
    crit_features = ["crit_prob_A", "crit_prob_B"]
    interaction_features = ["crit_prob_A_x_at_risk", "crit_prob_B_x_at_risk"]

    y = panel["compliance_failure"].to_numpy(dtype=int)
    groups = panel["part_id"].astype(str).to_numpy()
    n_parts = panel["part_id"].nunique()

    def build_X(cols: list[str]) -> pd.DataFrame:
        X = panel[cols].astype(float).copy()
        scaler = StandardScaler()
        X.loc[:, :] = scaler.fit_transform(X)
        X = sm.add_constant(X, has_constant="add")
        return X

    X_reduced = build_X(tabular_features)  # "uniform"-equivalent: no criticality conditioning
    X_full = build_X(tabular_features + crit_features + interaction_features)  # "conditioned"

    m_reduced = sm.GLM(y, X_reduced, family=sm.families.Binomial()).fit()
    m_full = sm.GLM(y, X_full, family=sm.families.Binomial()).fit()
    m_full_robust = sm.GLM(y, X_full, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": groups}
    )

    lr_stat = float(2 * (m_full.llf - m_reduced.llf))
    df_diff = int(m_full.df_model - m_reduced.df_model)
    lr_p = float(stats.chi2.sf(lr_stat, df_diff))

    crit_coef_rows = []
    for feat in crit_features + interaction_features:
        crit_coef_rows.append(
            {
                "feature": feat,
                "coef": float(m_full_robust.params[feat]),
                "cluster_robust_se": float(m_full_robust.bse[feat]),
                "cluster_robust_p_value": float(m_full_robust.pvalues[feat]),
            }
        )

    result = {
        "n_rows": int(len(panel)),
        "n_parts_clusters": int(n_parts),
        "n_events": int(y.sum()),
        "likelihood_ratio_test": {
            "lr_statistic": lr_stat,
            "df": df_diff,
            "p_value": lr_p,
            "interpretation": (
                "Tests whether adding crit_prob_A, crit_prob_B, and their at-risk interactions "
                "(4 params) significantly improves fit over the tabular-only (uniform-equivalent) "
                "model, using standard (non-cluster-adjusted) maximum likelihood."
            ),
        },
        "cluster_robust_coefficients": crit_coef_rows,
        "caveat": (
            f"Cluster-robust SE asymptotics assume enough independent clusters; n={n_parts} "
            "part-level clusters is above the commonly-cited 50-cluster rule of thumb but not "
            "large. Treat this as a complementary check to the out-of-sample bootstrap/multi-seed "
            "analysis, not a replacement."
        ),
    }

    out_path = out_dir / "panel_significance_test.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
