#!/usr/bin/env python3
"""
Standalone Layer 2 compliance-prediction validation on the Olist dataset (second real domain).
Does NOT modify any existing modeling script, the synthetic pipeline, or bom_graph.py -- only
imports modeling_lib for its already-validated statistical machinery (bootstrap, FDR correction,
business value, metrics), so the methodology matches the synthetic pipeline exactly rather than
risking a parallel reimplementation that silently diverges.

INTENTIONAL DESIGN: Olist has no BOM/assembly structure, so Layer 1 here is tabular LightGBM
ONLY -- no GAT. The "Synthetic" comparison column in Step 8 therefore uses baseline2_lgbm_tabular
(our synthetic pipeline's tabular-only model), not layer1_full_gat, for an apples-to-apples
comparison -- verified against outputs/paper_results.json before writing this script.

DEVIATION FROM THE ORIGINAL REQUEST, disclosed here rather than silently applied: the LGBM
hyperparameters given in the original request (n_estimators=500, learning_rate=0.05,
num_leaves=63, min_child_samples=20) are STALE -- they predate this session's hyperparameter
search. This script uses modeling_lib.LGBM_MULTICLASS_PARAMS / LGBM_BINARY_PARAMS directly (the
actual current post-search values) so "same as synthetic pipeline" is literally true, not
nominally true. Verified: lead_time_cv imputation constant (0.2227) matches
outputs/part_catalog.csv's real mean exactly, kept as given.

Run: uv run --group modeling python run_olist_layer2.py
Writes: outputs/olist/{layer1_results,layer2_results,panel_logistic_results,business_value_results}.json,
        outputs/olist/comparison_table.csv, outputs/olist/narrative_summary.txt
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from lightgbm import LGBMClassifier
from scipy import stats
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import modeling_lib as ml  # noqa: E402

OUT = ROOT / "outputs" / "olist"
OUT.mkdir(parents=True, exist_ok=True)

COST_A_FAILURE = 50000
COST_B_FAILURE = 10000
COST_C_FAILURE = 2000
COST_FALSE_POSITIVE = 500
PLANNING_HORIZON_MONTHS = 23  # Olist's real span; note: cost constants are automotive-calibrated,
# not Brazilian e-commerce retail -- business value numbers here are illustrative of the
# framework's operational-value mechanics, not Olist's actual financial exposure.

SYNTHETIC_LEAD_TIME_CV_MEAN = 0.2227  # verified against outputs/part_catalog.csv before use
AT_RISK_OTD_THRESHOLD = 0.85

COLUMN_MAP = {
    "product_id": "part_id",
    "seller_id": "supplier_id",
    "price": "unit_cost_mean_raw",  # avoid clobbering the already-computed unit_cost_mean column
    "product_category_en": "product_category",  # NOTE: fixed from the request's
    # 'product_category_name_english' -- the actual saved column (verified against
    # outputs/olist_modeling_ready.csv) is 'product_category_en'.
    "otd_rate_month": "otd_oem_measured",
    "otd_roll3": "otd_oem_roll3",
    "otd_roll6": "otd_oem_roll6",
    "n_sellers": "n_qualified_suppliers",
    "demand_cv_monthly": "demand_cv",
    "seller_state": "supplier_region",
}


def r4(x: Any) -> Any:
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return x
    if math.isnan(f) or math.isinf(f):
        return None
    return round(f, 4)


def fmt_p(p: Any) -> Any:
    if p is None:
        return None
    f = float(p)
    if math.isnan(f):
        return None
    return "<0.001" if f < 0.001 else round(f, 4)


def ci(lo: Any, hi: Any) -> Optional[list]:
    if lo is None or hi is None:
        return None
    return [r4(lo), r4(hi)]


def log_step(n: int, desc: str) -> None:
    print(f"\n{'=' * 70}\nStep {n}: {desc}\n{'=' * 70}")


# ---------------------------------------------------------------------------
# Step 1: Load and validate
# ---------------------------------------------------------------------------
def step1_load() -> pd.DataFrame:
    log_step(1, "Load and validate")
    path = ROOT / "outputs" / "olist_modeling_ready.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- run check_olist_dataset.py first")
    df = pd.read_csv(path, parse_dates=["month"])
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Null rates:\n{df.isna().mean().round(3)}")
    print(f"Overall compliance_failure rate: {df['compliance_failure'].mean():.4f}")

    if "criticality_class" not in df.columns:
        print("  criticality_class not found -- computing via composite price/demand-CV rank (fallback path).")
        price_rank = df.groupby("product_id")["price"].mean().rank(pct=True)
        cv_rank = df.groupby("product_id")["demand_cv_monthly"].mean().rank(pct=True)
        composite = (0.7 * price_rank + 0.3 * cv_rank).rename("composite")
        q_a, q_b = composite.quantile([0.80, 0.50])
        cls = np.where(composite >= q_a, "A", np.where(composite >= q_b, "B", "C"))
        cls_map = pd.Series(cls, index=composite.index)
        df["criticality_class"] = df["product_id"].map(cls_map)
    else:
        print("  criticality_class already present in file -- using as-is (fallback path not needed).")
    print(f"Compliance failure rate by class:\n{df.groupby('criticality_class')['compliance_failure'].mean().round(4)}")

    pair_totals = df.groupby(["product_id", "seller_id"])["n_orders"].sum()
    qualifying = pair_totals[pair_totals >= 3].index
    n_before = len(df)
    df = df.set_index(["product_id", "seller_id"]).loc[df.set_index(["product_id", "seller_id"]).index.isin(qualifying)].reset_index()
    print(f"Min-activity filter (>=3 total orders per product-seller pair): {n_before} -> {len(df)} rows")
    print(f"Unique products remaining: {df['product_id'].nunique()}")
    return df


# ---------------------------------------------------------------------------
# Step 2: Feature construction
# ---------------------------------------------------------------------------
def step2_features(df: pd.DataFrame) -> pd.DataFrame:
    log_step(2, "Feature construction")
    df = df.rename(columns=COLUMN_MAP).copy()

    seller_overall_otd = df.groupby("supplier_id")["otd_oem_measured"].mean()
    df["supplier_at_risk_flag"] = (df["supplier_id"].map(seller_overall_otd) < AT_RISK_OTD_THRESHOLD).astype(int)
    print(f"Fraction of supplier-rows flagged at-risk: {df['supplier_at_risk_flag'].mean():.4f}")

    df["lead_time_cv"] = SYNTHETIC_LEAD_TIME_CV_MEAN
    df["reschedule_burden_pp"] = 0.0

    # BOM features EXCLUDED entirely (not zero-imputed) from the Olist feature set used in
    # modeling -- per the original request's explicit instruction: zero-imputing would let a
    # model learn "zero BOM exposure" as a signal, when it actually means "unavailable." We only
    # record which BOM columns would have existed, for the appendix table below.
    bom_feature_names = [
        "bom_in_degree", "bom_out_degree", "bom_longest_downstream_path",
        "bom_n_downstream_a_assemblies", "bom_criticality_propagation_score",
    ]

    feature_table = pd.DataFrame([
        {"feature": "unit_cost_mean", "status": "real", "note": "Olist price field, order-item grain, mean per product"},
        {"feature": "demand_cv", "status": "real", "note": "Olist demand_cv_monthly, computed from real order counts"},
        {"feature": "n_qualified_suppliers", "status": "real (proxy)", "note": "Olist n_sellers -- distinct real sellers per product, not a true 'qualified supplier' count"},
        {"feature": "product_category", "status": "real", "note": "Olist product_category_en"},
        {"feature": "otd_oem_measured", "status": "real", "note": "Olist otd_rate_month, real on-time rate"},
        {"feature": "otd_oem_roll3", "status": "real", "note": "Olist otd_roll3, real rolling rate"},
        {"feature": "otd_oem_roll6", "status": "real", "note": "Olist otd_roll6, real rolling rate"},
        {"feature": "supplier_at_risk_flag", "status": "derived from real data", "note": f"1 if seller's real overall OTD < {AT_RISK_OTD_THRESHOLD}"},
        {"feature": "lead_time_cv", "status": "IMPUTED (constant)", "note": f"No Olist analog; set to synthetic dataset's real mean ({SYNTHETIC_LEAD_TIME_CV_MEAN}) for all rows"},
        {"feature": "reschedule_burden_pp", "status": "IMPUTED (constant)", "note": "No Olist analog; set to 0.0 for all rows"},
    ] + [
        {"feature": f, "status": "EXCLUDED", "note": "No Olist analog (no BOM/assembly structure) -- excluded entirely, not zero-imputed, to avoid the model learning a fake 'zero exposure' signal"}
        for f in bom_feature_names
    ])
    print(feature_table.to_string(index=False))
    feature_table.to_csv(OUT / "feature_provenance_table.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# Step 3: Train/test split
# ---------------------------------------------------------------------------
def step3_split(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    log_step(3, "Train/test split (product-level, stratified)")
    products = df.drop_duplicates("part_id")[["part_id", "criticality_class"]]
    y = products["criticality_class"].map(ml.CRIT_MAP)
    train_parts, test_parts = train_test_split(
        products["part_id"].to_numpy(), test_size=0.2, random_state=42, stratify=y
    )
    df_train = df[df["part_id"].isin(train_parts)]
    df_test = df[df["part_id"].isin(test_parts)]
    print(f"Train: {len(df_train)} rows, {df_train['part_id'].nunique()} products, "
          f"failure rate {df_train['compliance_failure'].mean():.4f}")
    print(f"Test:  {len(df_test)} rows, {df_test['part_id'].nunique()} products, "
          f"failure rate {df_test['compliance_failure'].mean():.4f}")
    print(f"Train ABC shares: {products[products['part_id'].isin(train_parts)]['criticality_class'].value_counts(normalize=True).round(4).to_dict()}")
    print(f"Test ABC shares:  {products[products['part_id'].isin(test_parts)]['criticality_class'].value_counts(normalize=True).round(4).to_dict()}")
    return train_parts, test_parts


# ---------------------------------------------------------------------------
# Step 4: Layer 1 -- tabular LGBM only, no GAT (Olist has no BOM)
# ---------------------------------------------------------------------------
LAYER1_FEATURES = ["unit_cost_mean", "demand_cv", "n_qualified_suppliers"]


def _category_dummies(df: pd.DataFrame, top_categories: Optional[list] = None) -> tuple[pd.DataFrame, list]:
    if top_categories is None:
        top_categories = df["product_category"].value_counts().head(20).index.tolist()
    cat = df["product_category"].where(df["product_category"].isin(top_categories), "other")
    dummies = pd.get_dummies(cat, prefix="cat")
    return dummies, top_categories


def step4_layer1(df: pd.DataFrame, train_parts: np.ndarray, test_parts: np.ndarray) -> dict:
    log_step(4, "Layer 1 -- ABC classification (tabular LightGBM ONLY, no GAT: Olist has no BOM structure)")
    products = df.drop_duplicates("part_id").set_index("part_id")
    train_p = products.loc[products.index.isin(train_parts)]
    test_p = products.loc[products.index.isin(test_parts)]

    cat_dummies_all, top_cats = _category_dummies(products)
    X_all = pd.concat([products[LAYER1_FEATURES].reset_index(drop=True), cat_dummies_all.reset_index(drop=True)], axis=1)
    X_all.index = products.index
    y_all = products["criticality_class"].map(ml.CRIT_MAP)

    X_train, y_train = X_all.loc[train_p.index], y_all.loc[train_p.index]
    X_test, y_test = X_all.loc[test_p.index], y_all.loc[test_p.index]

    clf = LGBMClassifier(objective="multiclass", num_class=3, **ml.LGBM_MULTICLASS_PARAMS)
    clf.fit(X_train, y_train)
    y_pred_lgbm = clf.predict(X_test)
    lgbm_metrics = ml._multiclass_metrics_dict(y_test.to_numpy(), y_pred_lgbm)
    print(f"LGBM tabular macro_f1: {lgbm_metrics['f1_macro']:.4f}")

    price_train = train_p["unit_cost_mean"]
    q20, q50 = price_train.quantile([0.20, 0.50])
    y_pred_rule = np.where(test_p["unit_cost_mean"] <= q20, 0, np.where(test_p["unit_cost_mean"] <= q50, 1, 2))
    rule_metrics = ml._multiclass_metrics_dict(y_test.to_numpy(), y_pred_rule)
    print(f"Rule-based (price quantile) macro_f1: {rule_metrics['f1_macro']:.4f}")

    boot_df = ml.bootstrap_layer1_comparisons(
        test_p.index.to_numpy().astype(str), y_test.to_numpy(),
        {"Baseline1_rule_price": y_pred_rule, "Baseline2_LGBM_tabular": y_pred_lgbm},
        n_boot=1000, random_state=0,
    )
    print(boot_df[["comparison", "point_diff", "p_value_two_sided", "p_value_fdr_bh"]].to_string(index=False))

    proba_all = clf.predict_proba(X_all)
    result = {
        "note": "Layer 1 is tabular-only (no GAT) -- Olist has no BOM/assembly structure.",
        "models": {
            "baseline1_rule_price": {"macro_f1": r4(rule_metrics["f1_macro"]), "weighted_f1": r4(rule_metrics["f1_weighted"]),
                                      "per_class_f1": {c: r4(rule_metrics.get(f"f1_{c}")) for c in "ABC"}},
            "baseline2_lgbm_tabular": {"macro_f1": r4(lgbm_metrics["f1_macro"]), "weighted_f1": r4(lgbm_metrics["f1_weighted"]),
                                        "per_class_f1": {c: r4(lgbm_metrics.get(f"f1_{c}")) for c in "ABC"},
                                        "per_class_precision": {c: r4(lgbm_metrics.get(f"precision_{c}")) for c in "ABC"},
                                        "per_class_recall": {c: r4(lgbm_metrics.get(f"recall_{c}")) for c in "ABC"},
                                        "confusion_matrix": lgbm_metrics["confusion_matrix"]},
        },
        "bootstrap_significance": [
            {"comparison": row["comparison"], "point_diff_macro_f1": r4(row["point_diff"]),
             "p_value_raw": fmt_p(row["p_value_two_sided"]), "p_value_fdr_bh": fmt_p(row["p_value_fdr_bh"])}
            for _, row in boot_df.iterrows()
        ],
    }
    ml.save_json(OUT / "layer1_results.json", result)
    return {"clf": clf, "proba_all": proba_all, "product_ids_all": list(X_all.index), "top_cats": top_cats, "result": result}


# ---------------------------------------------------------------------------
# Step 5: Layer 2 -- three models
# ---------------------------------------------------------------------------
L2_TABULAR_FEATURES = [
    "otd_oem_measured", "otd_oem_roll3", "otd_oem_roll6", "supplier_at_risk_flag",
    "unit_cost_mean", "demand_cv", "n_qualified_suppliers", "lead_time_cv", "reschedule_burden_pp",
]


def step5_layer2(df: pd.DataFrame, train_parts, test_parts, l1: dict) -> dict:
    log_step(5, "Layer 2 -- compliance prediction (prior / conditioned / oracle)")
    part_to_idx = {p: i for i, p in enumerate(l1["product_ids_all"])}
    proba_all = l1["proba_all"]

    cat_dummies, _ = _category_dummies(df, top_categories=l1["top_cats"])
    df = pd.concat([df.reset_index(drop=True), cat_dummies.reset_index(drop=True)], axis=1)
    cat_cols = list(cat_dummies.columns)

    df_train = df[df["part_id"].isin(train_parts)].copy()
    df_test = df[df["part_id"].isin(test_parts)].copy()
    y_tr = df_train["compliance_failure"].to_numpy(dtype=int)
    y_te = df_test["compliance_failure"].to_numpy(dtype=int)

    products_table = df.drop_duplicates("part_id")[["part_id", "criticality_class"]]

    def attach_pred_probs(target_df: pd.DataFrame) -> pd.DataFrame:
        idx = target_df["part_id"].map(part_to_idx).to_numpy()
        return ml.attach_crit_prob_matrix(target_df, proba_all[idx])

    df_tr_prior = ml.attach_marginal_prior_crit_probs(df_train, products_table, train_parts)
    df_te_prior = ml.attach_marginal_prior_crit_probs(df_test, products_table, train_parts)
    df_tr_cond = attach_pred_probs(df_train)
    df_te_cond = attach_pred_probs(df_test)
    df_tr_orac = ml.attach_oracle_crit_probs(df_train)
    df_te_orac = ml.attach_oracle_crit_probs(df_test)

    l2_cols = L2_TABULAR_FEATURES + cat_cols + ["crit_prob_A", "crit_prob_B", "crit_prob_C",
                                                  "crit_prob_A_x_at_risk", "crit_prob_B_x_at_risk", "crit_prob_C_x_at_risk"]

    spw = int((y_tr == 0).sum()) / max(int((y_tr == 1).sum()), 1)
    l2_params = ml._l2_classifier_params(spw)

    parts_present = df_train["part_id"].astype(str).unique()
    y_part_present = products_table.set_index("part_id").loc[parts_present, "criticality_class"].map(ml.CRIT_MAP).to_numpy()
    n_splits = max(2, min(5, int(np.bincount(y_part_present).min())))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=43)
    part_arr = df_train["part_id"].astype(str).to_numpy()

    def fit_predict(df_tr, df_te):
        Xtr, Xte = df_tr[l2_cols].to_numpy(dtype=np.float64), df_te[l2_cols].to_numpy(dtype=np.float64)
        m = LGBMClassifier(objective="binary", **l2_params)
        m.fit(ml._lgbm_df(Xtr), y_tr)
        s_te = m.predict_proba(ml._lgbm_df(Xte))[:, 1]
        s_val = np.full(len(df_tr), np.nan)
        for fit_idx, val_idx in skf.split(parts_present, y_part_present):
            fit_mask = np.isin(part_arr, parts_present[fit_idx])
            val_mask = np.isin(part_arr, parts_present[val_idx])
            mf = LGBMClassifier(objective="binary", **l2_params)
            mf.fit(ml._lgbm_df(Xtr[fit_mask]), y_tr[fit_mask])
            s_val[val_mask] = mf.predict_proba(ml._lgbm_df(Xtr[val_mask]))[:, 1]
        return s_te, s_val

    s_prior_te, s_prior_val = fit_predict(df_tr_prior, df_te_prior)
    s_cond_te, s_cond_val = fit_predict(df_tr_cond, df_te_cond)
    s_orac_te, s_orac_val = fit_predict(df_tr_orac, df_te_orac)

    models = {}
    for name, s_te in [("prior_baseline", s_prior_te), ("conditioned", s_cond_te), ("oracle", s_orac_te)]:
        m = ml.binary_metrics_suite(y_te, s_te, threshold=0.5)
        by_class = ml.metrics_by_criticality(df_test, y_te, s_te, threshold=0.5)
        models[name] = {**{k: r4(v) for k, v in m.items()},
                         "by_class": {c: {kk: r4(vv) for kk, vv in by_class.get(c, {}).items()} for c in "ABC"}}

    for name, y_val, s_val, s_te in [("prior_baseline", y_tr, s_prior_val, s_prior_te),
                                       ("conditioned", y_tr, s_cond_val, s_cond_te),
                                       ("oracle", y_tr, s_orac_val, s_orac_te)]:
        rep = ml.threshold_validation_max_f1_report(name, y_val, s_val, y_te, s_te)
        models[name]["f1_at_validation_threshold"] = r4(rep["f1_test"])
        models[name]["validation_threshold"] = r4(rep["threshold_validation_max_f1"])

    boot_df = ml.bootstrap_layer2_comparisons(
        df_test, y_te, {"prior": s_prior_te, "conditioned": s_cond_te, "oracle": s_orac_te},
        n_boot=1000, random_state=0,
        pairs=[("conditioned", "prior"), ("oracle", "prior")],
    )
    print(boot_df[["comparison", "metric", "point_diff", "p_value_two_sided", "p_value_fdr_bh"]].to_string(index=False))

    sig = [
        {"comparison": row["comparison"], "metric": row["metric"], "point_diff": r4(row["point_diff"]),
         "ci95": ci(row["ci_low"], row["ci_high"]), "p_value_raw": fmt_p(row["p_value_two_sided"]),
         "p_value_fdr_bh": fmt_p(row["p_value_fdr_bh"])}
        for _, row in boot_df.iterrows()
    ]
    result = {"models": models, "pairwise_significance_fdr": sig}
    ml.save_json(OUT / "layer2_results.json", result)
    return {"result": result, "df_train": df_train, "df_test": df_test, "y_tr": y_tr, "y_te": y_te,
            "s_cond_te": s_cond_te, "df_tr_cond": df_tr_cond, "df_te_cond": df_te_cond, "l2_cols": l2_cols}


# ---------------------------------------------------------------------------
# Step 6: Panel logistic regression
# ---------------------------------------------------------------------------
def step6_panel_logistic(l2: dict) -> dict:
    log_step(6, "Panel logistic regression (cluster-robust, NOT random-intercept)")
    df_tr_cond, df_te_cond = l2["df_tr_cond"], l2["df_te_cond"]
    panel = pd.concat([df_tr_cond, df_te_cond], ignore_index=True)
    y = panel["compliance_failure"].to_numpy(dtype=int)
    groups = panel["part_id"].astype(str).to_numpy()

    tabular_cols = [c for c in L2_TABULAR_FEATURES if c in panel.columns]
    crit_features = ["crit_prob_A", "crit_prob_B"]
    interaction_features = ["crit_prob_A_x_at_risk", "crit_prob_B_x_at_risk"]

    # lead_time_cv / reschedule_burden_pp are imputed CONSTANTS for Olist (no per-row analog --
    # see feature_provenance_table.csv). A zero-variance column is linearly dependent with the
    # intercept, which makes X'X exactly singular for GLM -- harmless for LightGBM (never split
    # on), fatal for logistic regression. Drop them here only, not from the LightGBM feature set.
    zero_variance = [c for c in tabular_cols if panel[c].nunique() <= 1]
    if zero_variance:
        print(f"Dropping zero-variance columns from panel regression design matrix: {zero_variance}")
        tabular_cols = [c for c in tabular_cols if c not in zero_variance]

    def build_X(cols):
        X = panel[cols].astype(float).copy()
        X.loc[:, :] = StandardScaler().fit_transform(X)
        return sm.add_constant(X, has_constant="add")

    X_reduced = build_X(tabular_cols)
    X_full = build_X(tabular_cols + crit_features + interaction_features)

    m_reduced = sm.GLM(y, X_reduced, family=sm.families.Binomial()).fit()
    m_full = sm.GLM(y, X_full, family=sm.families.Binomial()).fit()
    m_full_robust = sm.GLM(y, X_full, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": groups})

    lr_stat = float(2 * (m_full.llf - m_reduced.llf))
    df_diff = int(m_full.df_model - m_reduced.df_model)
    lr_p = float(stats.chi2.sf(lr_stat, df_diff))
    print(f"LR test: stat={lr_stat:.3f}, df={df_diff}, p={lr_p:.4f}")

    coefs = {}
    for feat in crit_features:
        est, se = m_full_robust.params[feat], m_full_robust.bse[feat]
        coefs[feat] = {"estimate": r4(est), "cluster_robust_se": r4(se), "z_stat": r4(est / se),
                        "p_value_raw": fmt_p(m_full_robust.pvalues[feat])}

    result = {
        "model_specification": "Logistic regression (statsmodels GLM Binomial) with part-level cluster-robust sandwich SEs, NOT random-intercept GLMM.",
        "criticality_coefficients": coefs,
        "omnibus_likelihood_ratio_test": {"lr_statistic": r4(lr_stat), "df": df_diff, "p_value_raw": fmt_p(lr_p),
                                           "significant_at_0.05": bool(lr_p < 0.05)},
        "n_observations": int(len(panel)), "n_parts_clusters": int(panel["part_id"].nunique()),
        "n_events": int(y.sum()),
    }
    ml.save_json(OUT / "panel_logistic_results.json", result)
    return result


# ---------------------------------------------------------------------------
# Step 7: Business value simulation
# ---------------------------------------------------------------------------
def step7_business_value(l2: dict) -> dict:
    log_step(7, "Business value simulation (conditioned model, illustrative -- see cost-constant caveat)")
    crit_te = l2["df_test"]["criticality_class"].astype(str).to_numpy()
    biz = ml.business_value_summary(
        crit_te, l2["y_te"], l2["s_cond_te"], [0.10, 0.15, 0.20, 0.25],
        {"A": COST_A_FAILURE, "B": COST_B_FAILURE, "C": COST_C_FAILURE}, COST_FALSE_POSITIVE,
    )
    print(biz.round(2).to_string(index=False))
    result = {
        "note": "Cost constants are automotive-manufacturing-calibrated, not Brazilian e-commerce retail -- illustrative of the framework's mechanics, not Olist's actual financial exposure.",
        "cost_assumptions": {"cost_A_failure": COST_A_FAILURE, "cost_B_failure": COST_B_FAILURE,
                              "cost_C_failure": COST_C_FAILURE, "cost_false_positive": COST_FALSE_POSITIVE,
                              "planning_horizon_months": PLANNING_HORIZON_MONTHS},
        "by_threshold": {str(r4(row["threshold"])): {k: (r4(v) if isinstance(v, float) else int(v)) for k, v in row.items()} for _, row in biz.iterrows()},
    }
    ml.save_json(OUT / "business_value_results.json", result)
    return result


# ---------------------------------------------------------------------------
# Step 8: Comparison summary
# ---------------------------------------------------------------------------
def step8_comparison(l1_result: dict, l2_result: dict, panel_result: dict) -> pd.DataFrame:
    log_step(8, "Comparison summary (synthetic vs Olist)")
    synth_path = ROOT / "outputs" / "paper_results.json"
    if not synth_path.exists():
        print(f"  [MISSING] {synth_path} -- run collect_paper_results.py first. Comparison table will have null synthetic values.")
        synth = None
    else:
        synth = json.loads(synth_path.read_text(encoding="utf-8"))

    def synth_get(*path, default=None):
        d = synth
        if d is None:
            return default
        for p in path:
            d = d.get(p, {}) if isinstance(d, dict) else default
        return d if d != {} else default

    rows = [
        ("Layer1 rule_based macro_f1", synth_get("layer1_classification", "models", "baseline1_rule_price", "macro_f1"),
         l1_result["models"]["baseline1_rule_price"]["macro_f1"]),
        ("Layer1 lgbm_tabular macro_f1", synth_get("layer1_classification", "models", "baseline2_lgbm_tabular", "macro_f1"),
         l1_result["models"]["baseline2_lgbm_tabular"]["macro_f1"]),
        ("Layer2 prior_baseline auc_pr", synth_get("layer2_predictive_single_run", "models", "prior_baseline", "auc_pr"),
         l2_result["models"]["prior_baseline"]["auc_pr"]),
        ("Layer2 conditioned auc_pr", synth_get("layer2_predictive_single_run", "models", "conditioned", "auc_pr"),
         l2_result["models"]["conditioned"]["auc_pr"]),
        ("Layer2 oracle auc_pr", synth_get("layer2_predictive_single_run", "models", "oracle", "auc_pr"),
         l2_result["models"]["oracle"]["auc_pr"]),
        ("panel_lr_p_value", synth_get("panel_logistic", "omnibus_likelihood_ratio_test", "p_value_raw"),
         panel_result["omnibus_likelihood_ratio_test"]["p_value_raw"]),
    ]
    out_rows = []
    for metric, synth_val, olist_val in rows:
        try:
            sv, ov = float(str(synth_val).replace("<", "")), float(str(olist_val).replace("<", ""))
            direction_consistent = (sv > 0) == (ov > 0) if "p_value" not in metric else None
        except (TypeError, ValueError):
            direction_consistent = None
        out_rows.append({"metric": metric, "synthetic": synth_val, "olist": olist_val,
                          "direction_consistent": direction_consistent})
    cmp_df = pd.DataFrame(out_rows)
    print(cmp_df.to_string(index=False))
    cmp_df.to_csv(OUT / "comparison_table.csv", index=False)
    return cmp_df


# ---------------------------------------------------------------------------
# Step 9: Narrative summary
# ---------------------------------------------------------------------------
def _p_significant(p_formatted: Any, alpha: float = 0.05) -> bool:
    """p_formatted is fmt_p()'s output: either a float, or the string '<0.001'. Compares the
    underlying numeric value, not the display string (an earlier version of this function
    string-compared '<0.001' against '0.05' lexicographically, which is wrong: mislabels highly
    significant results as non-significant)."""
    if p_formatted is None:
        return False
    if isinstance(p_formatted, str) and p_formatted.startswith("<"):
        return True  # fmt_p only emits "<0.001", which is significant at any alpha used here
    return float(p_formatted) < alpha


def step9_narrative(l1_result: dict, l2_result: dict, panel_result: dict) -> None:
    log_step(9, "Narrative summary")
    from datetime import datetime, timezone

    rule_f1 = l1_result["models"]["baseline1_rule_price"]["macro_f1"]
    lgbm_f1 = l1_result["models"]["baseline2_lgbm_tabular"]["macro_f1"]
    l1_sig = next((s for s in l1_result["bootstrap_significance"] if "Baseline2" in s["comparison"] and "Baseline1" in s["comparison"]), {})

    prior_pr = l2_result["models"]["prior_baseline"]["auc_pr"]
    cond_pr = l2_result["models"]["conditioned"]["auc_pr"]
    orac_pr = l2_result["models"]["oracle"]["auc_pr"]
    l2_sig = [s for s in l2_result["pairwise_significance_fdr"] if s["metric"] == "auc_pr"]
    panel_p = panel_result["omnibus_likelihood_ratio_test"]["p_value_raw"]
    panel_sig = panel_result["omnibus_likelihood_ratio_test"]["significant_at_0.05"]

    lines = [
        "OLIST VALIDATION SUMMARY",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "LAYER 1 FINDING:",
        (
            f"Tabular LightGBM (macro F1={lgbm_f1}) {'beats' if lgbm_f1 > rule_f1 else 'does not beat'} "
            f"the price-quantile rule baseline (macro F1={rule_f1}) on Olist "
            f"({'a significant' if _p_significant(l1_sig.get('p_value_fdr_bh')) else 'a not-clearly-significant'} "
            f"difference, FDR-adjusted p={l1_sig.get('p_value_fdr_bh')}). This "
            f"{'replicates' if (lgbm_f1 > rule_f1) else 'does not replicate'} the synthetic dataset's "
            "significant ML-beats-rule finding (FDR-adjusted p=0.0105 there)."
        ),
        "",
        "LAYER 2 FINDING:",
        (
            f"AUC-PR: prior_baseline={prior_pr}, conditioned={cond_pr}, oracle={orac_pr}. "
            f"Conditioned-vs-prior and oracle-vs-prior FDR-adjusted p-values: "
            f"{[(s['comparison'], s['p_value_fdr_bh']) for s in l2_sig]}. "
            f"The panel logistic regression's omnibus LR test for the criticality block is "
            f"{'significant' if panel_sig else 'not significant'} (p={panel_p}), which is "
            f"{'consistent with' if not panel_sig else 'inconsistent with'} the synthetic dataset's "
            "post-hyperparameter-search null result (p=0.921) on the same test."
        ),
        "",
        "CROSS-DOMAIN CONSISTENCY:",
        (
            "See comparison_table.csv for the full metric-by-metric comparison and direction-consistency "
            "flags. Where directions agree across both a synthetic dataset (literature-parameterized, "
            "DGP explicitly searched to produce a conditioning effect during development) and an "
            "independent real retail dataset (Olist, no such search applied), that agreement is more "
            "informative than either result alone -- it is not explainable by DGP tuning, since Olist's "
            "generating process was not tuned by this project at all."
        ),
        "",
        "LIMITATIONS OF THIS VALIDATION:",
        "- Domain difference: Brazilian e-commerce retail vs. automotive/industrial manufacturing -- different criticality drivers entirely.",
        "- lead_time_cv and reschedule_burden_pp are IMPUTED constants (no Olist analog), not real per-product/seller values -- see feature_provenance_table.csv.",
        "- BOM features excluded entirely (not imputed) -- Layer 1 is tabular-only, no GAT comparison possible on Olist.",
        "- compliance_failure here means 'delivered later than Olist's own estimated delivery date' (a promise to the customer), a different construct than a fixed production schedule.",
        "- License of the specific mirror used was not independently confirmed against the official Kaggle listing (see olist_feasibility_report.txt).",
        "- Business value dollar figures use automotive-calibrated cost constants; not interpretable as Olist's real financial exposure.",
        "",
        "RECOMMENDED PAPER FRAMING:",
        (
            "Frame this as a cross-domain generalizability check, not a replication in the strict sense "
            "(the domains, outcome semantics, and available features differ meaningfully) and not merely "
            "a robustness check (Olist is an independent real dataset, not a resampling of the same "
            "data). Report both the Layer 1 and Layer 2 findings with their significance levels exactly "
            "as computed, and lead with whichever result (agreement or disagreement) actually occurred, "
            "rather than the one that would look better -- the value of this check is in what it shows, "
            "not in confirming the synthetic result."
        ),
    ]
    (OUT / "narrative_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    t0 = time.time()
    df = step1_load()
    df = step2_features(df)
    train_parts, test_parts = step3_split(df)
    l1 = step4_layer1(df, train_parts, test_parts)
    l2 = step5_layer2(df, train_parts, test_parts, l1)
    panel_result = step6_panel_logistic(l2)
    step7_business_value(l2)
    step8_comparison(l1["result"], l2["result"], panel_result)
    step9_narrative(l1["result"], l2["result"], panel_result)
    print(f"\nTotal runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
