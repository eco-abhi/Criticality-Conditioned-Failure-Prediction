"""
Helpers for modeling.ipynb — two-layer criticality + compliance pipeline.

Inductive GAT: training forward uses only edges whose endpoints are both in the
current training part set (no message paths through held-out parts during backprop).

Layer 2: criticality probabilities on training rows come from K-fold OOF Layer 1;
test rows use probabilities from Layer 1 refit on all training parts.
"""

from __future__ import annotations

import json
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from lightgbm import LGBMClassifier
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import GATConv

# ---------------------------------------------------------------------------
# Feature definitions (aligned with synthetic generator exports)
# ---------------------------------------------------------------------------

CLASSIFICATION_FEATURES: List[str] = [
    "lead_time_mean_weeks",
    "lead_time_cv",
    "lead_time_sigma_weeks",
    "n_qualified_suppliers",
    "substitutable_flag",
    "stockout_events_per_year",
    "bom_in_degree",
    "bom_out_degree",
    "bom_longest_downstream_path",
    "bom_n_downstream_A_assemblies",
    "bom_criticality_propagation_score",
]

COMPLIANCE_EXTRA_FEATURES: List[str] = [
    "supplier_at_risk_flag",
    "otd_oem_measured",
    "otd_oem_roll3",
    "otd_oem_roll6",
    "reschedule_burden_pp",
    "reschedule_burden_roll3",
]

# Always available to Layer 2 even when LAYER1_FEATURES=clean (used in compliance DGP).
LAYER2_BOM_EXTRA_FEATURES: List[str] = ["bom_criticality_propagation_score"]

CRIT_PROB_FEATURES: List[str] = ["crit_prob_A", "crit_prob_B", "crit_prob_C"]

CRIT_PROB_AT_RISK_INTERACTION_FEATURES: List[str] = [
    "crit_prob_A_x_at_risk",
    "crit_prob_B_x_at_risk",
    "crit_prob_C_x_at_risk",
]

LGBM_MULTICLASS_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=20,
    class_weight="balanced",
    random_state=42,
    verbosity=-1,
    n_jobs=1,
)

LGBM_BINARY_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=20,
    random_state=42,
    verbosity=-1,
    n_jobs=1,
)

CRIT_MAP = {"A": 0, "B": 1, "C": 2}
INV_CRIT = {0: "A", 1: "B", 2: "C"}

# Excluded when LAYER1_FEATURES=clean (see get_layer1_classification_features).
# Includes: (1) BOM summaries built using true criticality_class in bom_graph; (2) generator ABC
# score inputs — labels are assigned from these in the synthetic pipeline, but they are not
# observable in deployment the way the quantile score is; they are omitted from CLASSIFICATION_FEATURES
# so supervised models cannot refit the scoring function from features.
LABEL_LEAKING_LAYER1_FEATURES: frozenset[str] = frozenset(
    {
        "abc_price_proxy",
        "abc_demand_cv_proxy",
        "bom_n_downstream_A_assemblies",
        "bom_criticality_propagation_score",
    }
)


def get_layer2_tabular_features(mode: Optional[str] = None) -> List[str]:
    """
    Tabular columns for Layer 2 (excludes crit_prob_* and interaction terms).

    Includes Layer 1 tabular features plus BOM cascade when Layer 1 is ``clean``.
    """
    l1 = get_layer1_classification_features(mode)
    extra = [c for c in LAYER2_BOM_EXTRA_FEATURES if c not in l1]
    return l1 + extra + list(COMPLIANCE_EXTRA_FEATURES)


def get_layer2_model_feature_columns(
    *,
    include_crit_probs: bool = True,
    include_interactions: bool = True,
    layer1_mode: Optional[str] = None,
) -> List[str]:
    """Full Layer 2 design matrix column list."""
    cols = get_layer2_tabular_features(layer1_mode)
    if include_crit_probs:
        cols = cols + list(CRIT_PROB_FEATURES)
    if include_interactions:
        cols = cols + list(CRIT_PROB_AT_RISK_INTERACTION_FEATURES)
    return cols


def sharpen_crit_probs(probs: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """``temperature < 1`` sharpens rows toward argmax (useful for Layer 2 conditioning)."""
    t = float(temperature)
    if t >= 1.0 - 1e-9:
        return np.asarray(probs, dtype=float)
    p = np.power(np.clip(np.asarray(probs, dtype=float), 1e-12, 1.0), 1.0 / t)
    return p / p.sum(axis=1, keepdims=True)


def add_crit_prob_at_risk_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """``crit_prob_* × supplier_at_risk_flag`` (uses whatever crit probs are already on ``df``)."""
    out = df.copy()
    ar = out["supplier_at_risk_flag"].astype(float)
    for c in ["A", "B", "C"]:
        out[f"crit_prob_{c}_x_at_risk"] = out[f"crit_prob_{c}"].astype(float) * ar
    return out


def attach_uniform_crit_probs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[["crit_prob_A", "crit_prob_B", "crit_prob_C"]] = 1.0 / 3.0
    return add_crit_prob_at_risk_interactions(out)


def attach_crit_prob_matrix(df: pd.DataFrame, probs: np.ndarray) -> pd.DataFrame:
    """``probs`` shape (n_rows, 3) aligned with ``df`` row order."""
    out = df.copy()
    if len(probs) != len(out):
        raise ValueError(f"probs rows {len(probs)} != df rows {len(out)}")
    out[["crit_prob_A", "crit_prob_B", "crit_prob_C"]] = probs.astype(float)
    return add_crit_prob_at_risk_interactions(out)


def attach_oracle_crit_probs(df: pd.DataFrame) -> pd.DataFrame:
    """Oracle ceiling: true-class one-hot as crit_prob_* plus interactions."""
    cc = df["criticality_class"].astype(str)
    probs = np.column_stack(
        [
            (cc == "A").astype(float),
            (cc == "B").astype(float),
            (cc == "C").astype(float),
        ]
    )
    return attach_crit_prob_matrix(df, probs)


def get_layer1_classification_features(mode: Optional[str] = None) -> List[str]:
    """
    Return the feature list used for Layer 1 (and the tabular slice of Layer 2 part-month rows).

    mode / env LAYER1_FEATURES:
      - ``full`` — all columns in CLASSIFICATION_FEATURES (default). Oracle ABC score proxies
        (`abc_price_proxy`, `abc_demand_cv_proxy`) are not in this list; they remain in
        ``part_catalog.csv`` for rule baselines only.
      - ``clean`` — drop columns in ``LABEL_LEAKING_LAYER1_FEATURES`` that appear in
        ``CLASSIFICATION_FEATURES`` (label-derived BOM summaries), for honest ABC-from-observables
        experiments.
    """
    m = (mode if mode is not None else os.environ.get("LAYER1_FEATURES", "full")).strip().lower()
    if m == "full":
        return list(CLASSIFICATION_FEATURES)
    if m == "clean":
        return [c for c in CLASSIFICATION_FEATURES if c not in LABEL_LEAKING_LAYER1_FEATURES]
    raise ValueError(
        f"Unknown LAYER1_FEATURES mode {m!r}; use 'full' or 'clean' (or pass mode= explicitly)."
    )


def optimal_f1_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    t_min: Optional[float] = None,
    t_max: Optional[float] = None,
    n_thresholds: int = 91,
) -> Tuple[float, float]:
    """
    Grid-search threshold on (y_true, y_score); return (best_threshold, f1 at that threshold).

    Search range defaults to ``THRESHOLD_SEARCH_MIN``–``THRESHOLD_SEARCH_MAX`` (0.05–0.50)
    so rare-positive panels do not pick near-zero thresholds that flag everything.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    lo = float(os.environ.get("THRESHOLD_SEARCH_MIN", "0.05")) if t_min is None else float(t_min)
    hi = float(os.environ.get("THRESHOLD_SEARCH_MAX", "0.50")) if t_max is None else float(t_max)
    if lo >= hi:
        raise ValueError(f"threshold search requires t_min < t_max, got {lo} >= {hi}")
    best_t, best_f = 0.5, -1.0
    for t in np.linspace(lo, hi, n_thresholds):
        f = float(f1_score(y_true, (y_score >= t).astype(int), zero_division=0))
        if f > best_f:
            best_f, best_t = f, float(t)
    return best_t, best_f


def classification_metrics_row(model_name: str, metrics: Mapping[str, Any]) -> Dict[str, Any]:
    """Flatten multiclass metrics (skip confusion matrix) for CSV export."""
    row: Dict[str, Any] = {"model": model_name}
    for k, v in metrics.items():
        if k == "confusion_matrix":
            continue
        if isinstance(v, (int, float, np.floating, np.integer)):
            row[k] = float(v)
    return row


def metrics_at_fixed_thresholds(
    model_name: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: Sequence[float],
) -> List[Dict[str, Any]]:
    """Precision/recall/F1 on test at each fixed threshold (business tradeoff curve)."""
    y_te = np.asarray(y_true).astype(int)
    s = np.asarray(y_score, dtype=float)
    rows: List[Dict[str, Any]] = []
    for thr in thresholds:
        y_hat = (s >= thr).astype(int)
        rows.append(
            {
                "model": model_name,
                "threshold": float(thr),
                "f1": float(f1_score(y_te, y_hat, zero_division=0)),
                "precision": float(precision_score(y_te, y_hat, zero_division=0)),
                "recall": float(recall_score(y_te, y_hat, zero_division=0)),
            }
        )
    return rows


def threshold_validation_max_f1_report(
    model_name: str,
    y_val: np.ndarray,
    s_val: np.ndarray,
    y_test: np.ndarray,
    s_test: np.ndarray,
) -> Dict[str, Any]:
    """Tune threshold on a held-out validation panel; report test metrics at that frozen threshold."""
    thr, f1_val = optimal_f1_threshold(y_val, s_val)
    s_te = np.asarray(s_test, dtype=float)
    y_hat_te = (s_te >= thr).astype(int)
    y_te = np.asarray(y_test).astype(int)
    return {
        "model": model_name,
        "threshold_validation_max_f1": thr,
        "f1_validation_at_threshold": f1_val,
        "f1_test": float(f1_score(y_te, y_hat_te, zero_division=0)),
        "precision_test": float(precision_score(y_te, y_hat_te, zero_division=0)),
        "recall_test": float(recall_score(y_te, y_hat_te, zero_division=0)),
        "brier_test": float(brier_score_loss(y_te, s_te)),
    }


def part_level_train_val_split(
    part_catalog: pd.DataFrame,
    train_parts: Sequence[str],
    val_fraction: float = 0.2,
    random_state: int = 43,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split training parts into fit vs validation (stratified on ABC)."""
    pc = part_catalog[part_catalog["part_id"].astype(str).isin(list(train_parts))].copy()
    parts = pc["part_id"].astype(str).values
    y = pc["criticality_class"].map(CRIT_MAP).astype(int).values
    fit_p, val_p = train_test_split(
        parts,
        test_size=val_fraction,
        random_state=random_state,
        stratify=y,
    )
    return fit_p, val_p


def _lgbm_df(X: np.ndarray) -> pd.DataFrame:
    """Consistent column names for LightGBM sklearn API (avoids feature-name warnings)."""
    X = np.asarray(X, dtype=np.float64)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def project_root() -> Path:
    return Path(__file__).resolve().parent


def load_bom_graph(path: Path) -> nx.DiGraph:
    with open(path, "rb") as f:
        G = pickle.load(f)
    if not isinstance(G, nx.DiGraph):
        raise TypeError("bom_graph.gpickle must be a networkx.DiGraph")
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("BOM graph must be a DAG")
    return G


def nx_to_edge_index(G: nx.DiGraph, part_order: Sequence[str]) -> torch.Tensor:
    idx = {p: i for i, p in enumerate(part_order)}
    src: List[int] = []
    dst: List[int] = []
    for u, v in G.edges():
        src.append(idx[u])
        dst.append(idx[v])
    if not src:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.tensor([src, dst], dtype=torch.long)


def filter_edges_both_endpoints_in(
    edge_index: torch.Tensor, allowed_indices: np.ndarray
) -> torch.Tensor:
    """Keep only edges (u,v) where u and v are both in allowed_indices."""
    if edge_index.numel() == 0:
        return edge_index
    allowed = set(int(x) for x in allowed_indices.tolist())
    ei = edge_index.cpu().numpy()
    mask = np.array([ei[0, j] in allowed and ei[1, j] in allowed for j in range(ei.shape[1])])
    if not mask.any():
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.tensor(ei[:, mask], dtype=torch.long)


def get_compliance_grain() -> str:
    """
    Layer 2 row unit: ``part`` (default, one row per part) or ``part_month`` (full panel).
    Set ``COMPLIANCE_GRAIN`` env before loading data.
    """
    g = os.environ.get("COMPLIANCE_GRAIN", "part_month").strip().lower()
    if g not in ("part", "part_month"):
        raise ValueError(f"COMPLIANCE_GRAIN must be 'part' or 'part_month', got {g!r}")
    return g


def get_layer2_scope() -> str:
    """
    Layer 2 part scope: ``all`` (default) or ``real_category_only``.

    UCI Online Retail and DataCo do not share a category vocabulary (see
    ``generate_synthetic_datasets.build_unified_part_catalog`` — a TF-IDF similarity check found
    >50% of UCI products have zero real overlap with any DataCo category). Most UCI-sourced parts'
    supplier-proxy link is therefore a random within-category fallback, not a genuine category
    match. ``real_category_only`` restricts Layer 2 (supplier/compliance) rows to parts flagged
    ``real_category_link=True`` in part_catalog.csv (~100-120 parts per run) for a fully-grounded,
    smaller-N result. Set env ``LAYER2_SCOPE``.
    """
    s = os.environ.get("LAYER2_SCOPE", "all").strip().lower()
    if s not in ("all", "real_category_only"):
        raise ValueError(f"LAYER2_SCOPE must be 'all' or 'real_category_only', got {s!r}")
    return s


def filter_layer2_scope(df: pd.DataFrame, part_catalog: pd.DataFrame) -> pd.DataFrame:
    """Apply ``get_layer2_scope()`` to a compliance panel keyed by ``part_id``."""
    scope = get_layer2_scope()
    if scope == "all":
        return df
    if "real_category_link" not in part_catalog.columns:
        raise ValueError(
            "LAYER2_SCOPE=real_category_only requires a 'real_category_link' column in "
            "part_catalog.csv; regenerate data with the current generate_synthetic_datasets.py."
        )
    linked = set(
        part_catalog.loc[part_catalog["real_category_link"].astype(bool), "part_id"].astype(str)
    )
    out = df[df["part_id"].astype(str).isin(linked)].copy()
    if out.empty:
        raise ValueError("LAYER2_SCOPE=real_category_only left zero rows; check part_catalog.csv.")
    return out


def aggregate_compliance_panel_to_part_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse the compliance panel to one row per part.

    Label: **last month** in the horizon (matches ~8% calibrated monthly rate).
    Using ``max`` over months would inflate the positive rate (~1 - (1-p)^24).

    Features: last month's supplier snapshot; static part/BOM fields unchanged.
    """
    work = df.sort_values(["part_id", "month"])
    last = work.groupby("part_id", as_index=False).last()
    if "compliance_failure_prob" in last.columns:
        # Optional sanity: part-level empirical rate should stay near generator target.
        pass
    return last.reset_index(drop=True)


def load_and_prepare_data(
    outputs_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, nx.DiGraph, List[str], List[str]]:
    """Load compliance panel; optionally aggregate to one row per part (see ``get_compliance_grain``)."""
    part_catalog = pd.read_csv(outputs_dir / "part_catalog.csv")
    supplier_history = pd.read_csv(outputs_dir / "supplier_history.csv")
    compliance = pd.read_csv(outputs_dir / "compliance_outcomes.csv")
    G = load_bom_graph(outputs_dir / "bom_graph.gpickle")

    overlap = [c for c in CLASSIFICATION_FEATURES if c in compliance.columns]
    co = compliance.drop(columns=overlap, errors="ignore")
    pc_cols = ["part_id"] + CLASSIFICATION_FEATURES + ["criticality_class"]
    missing_pc = [c for c in pc_cols if c not in part_catalog.columns]
    if missing_pc:
        raise ValueError(f"part_catalog missing columns: {missing_pc}")
    df = co.merge(part_catalog[pc_cols], on="part_id", how="left", suffixes=("", "_pc"))

    roll_cols = ["part_id", "month", "otd_oem_roll3", "otd_oem_roll6", "reschedule_burden_roll3"]
    sh = supplier_history[roll_cols].drop_duplicates(subset=["part_id", "month"])
    df = df.merge(sh, on=["part_id", "month"], how="left", validate="one_to_one")

    for c in CLASSIFICATION_FEATURES:
        if c not in df.columns:
            raise ValueError(f"Missing classification column after merge: {c}")

    assert df.duplicated(subset=["part_id", "month"]).sum() == 0, "Duplicate part-month rows"
    assert len(df) == len(compliance), "Row count mismatch after merges"

    grain = get_compliance_grain()
    if grain == "part":
        df = aggregate_compliance_panel_to_part_level(df)

    compliance_features = get_layer2_model_feature_columns(
        include_crit_probs=True,
        include_interactions=True,
        layer1_mode="full",
    )
    return df, part_catalog, G, CLASSIFICATION_FEATURES, compliance_features


def part_level_train_test_split(
    part_catalog: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    parts = part_catalog["part_id"].astype(str).values
    y = part_catalog["criticality_class"].map(CRIT_MAP).astype(int).values
    train_p, test_p = train_test_split(
        parts,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return train_p, test_p


def baseline_rule_abc_quantile(
    part_catalog: pd.DataFrame,
    train_parts: np.ndarray,
    test_parts: np.ndarray,
    price_col: str = "abc_price_proxy",
) -> Dict[str, Any]:
    """20/30/50 quantile cuts on price proxy (quantiles fit on train parts only)."""
    train_mask = part_catalog["part_id"].isin(train_parts)
    prices_train = part_catalog.loc[train_mask, price_col].astype(float)
    q20 = float(prices_train.quantile(0.20))
    q50 = float(prices_train.quantile(0.50))

    def assign_abc(p: float) -> int:
        if p <= q20:
            return 0  # A — lowest 20% by price proxy
        if p <= q50:
            return 1  # B — next 30%
        return 2  # C — top 50%

    y_true = part_catalog["criticality_class"].map(CRIT_MAP).astype(int).values
    y_pred = part_catalog[price_col].astype(float).map(assign_abc).astype(int).values
    test_mask = part_catalog["part_id"].isin(test_parts)
    yt = y_true[test_mask.values]
    yp = y_pred[test_mask.values]
    return _multiclass_metrics_dict(yt, yp)


def _multiclass_metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }
    for c in [0, 1, 2]:
        m = y_true == c
        if m.any():
            out[f"precision_{INV_CRIT[c]}"] = float(
                precision_score(y_true == c, y_pred == c, zero_division=0)
            )
            out[f"recall_{INV_CRIT[c]}"] = float(
                recall_score(y_true == c, y_pred == c, zero_division=0)
            )
            out[f"f1_{INV_CRIT[c]}"] = float(
                f1_score(y_true == c, y_pred == c, zero_division=0)
            )
    return out


multiclass_metrics_dict = _multiclass_metrics_dict


def train_lgbm_multiclass(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[LGBMClassifier, Dict[str, Any]]:
    clf = LGBMClassifier(objective="multiclass", num_class=3, **LGBM_MULTICLASS_PARAMS)
    clf.fit(_lgbm_df(X_train), y_train)
    y_pred = clf.predict(_lgbm_df(X_test))
    metrics = _multiclass_metrics_dict(y_test, y_pred)
    return clf, metrics


def predict_part_criticality_proba(
    clf: LGBMClassifier, X_parts: np.ndarray, part_ids: Sequence[str]
) -> pd.DataFrame:
    proba = clf.predict_proba(_lgbm_df(X_parts))
    return pd.DataFrame(
        proba,
        index=list(part_ids),
        columns=["crit_prob_A", "crit_prob_B", "crit_prob_C"],
    )


class GATClassifier(torch.nn.Module):
    """Two-layer GAT with classification head for training; embeddings are pre-head."""

    def __init__(
        self,
        in_dim: int,
        hidden: int = 64,
        embed_dim: int = 32,
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.dropout_p = dropout
        self.conv1 = GATConv(in_dim, hidden, heads=4, dropout=dropout, add_self_loops=True)
        self.conv2 = GATConv(4 * hidden, embed_dim, heads=1, concat=False, dropout=dropout, add_self_loops=True)
        self.lin = torch.nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        emb = self.conv2(x, edge_index)
        logits = self.lin(emb)
        return emb, logits


def class_weights_from_labels(y: np.ndarray, num_classes: int = 3) -> torch.Tensor:
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (num_classes * counts)
    return torch.tensor(w, dtype=torch.float32)


def train_gat_classifier(
    x: torch.Tensor,
    edge_index_train: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    epochs: int = 200,
    lr: float = 0.005,
    weight_decay: float = 1e-4,
    device: Optional[torch.device] = None,
) -> Tuple[GATClassifier, List[float]]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GATClassifier(x.shape[1]).to(device)
    y_train = y[train_mask].cpu().numpy()
    cw = class_weights_from_labels(y_train).to(device)
    crit = torch.nn.CrossEntropyLoss(weight=cw)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    x = x.to(device)
    edge_index_train = edge_index_train.to(device)
    y = y.to(device)
    train_mask = train_mask.to(device)
    losses: List[float] = []
    model.train()
    for ep in range(1, epochs + 1):
        opt.zero_grad()
        _, logits = model(x, edge_index_train)
        loss = crit(logits[train_mask], y[train_mask])
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
        if ep % 20 == 0 or ep == 1:
            print(f"  GAT epoch {ep}/{epochs} loss={losses[-1]:.4f}")
    return model, losses


@torch.no_grad()
def gat_embeddings_and_logits(
    model: GATClassifier,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    device = device or next(model.parameters()).device
    model.eval()
    emb, logits = model(x.to(device), edge_index.to(device))
    return emb.cpu().numpy(), logits.cpu().numpy()


def build_part_arrays(
    part_catalog: pd.DataFrame,
    part_order: Sequence[str],
    features: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    pc = part_catalog.set_index("part_id").loc[list(part_order)]
    X = pc[list(features)].to_numpy(dtype=np.float64)
    y = pc["criticality_class"].map(CRIT_MAP).astype(int).to_numpy()
    return X, y, list(pc.index)


def layer1_stack_train_predict(
    X_tab_train: np.ndarray,
    y_train: np.ndarray,
    X_tab_test: np.ndarray,
    y_test: np.ndarray,
    emb_train: np.ndarray,
    emb_test: np.ndarray,
) -> Tuple[LGBMClassifier, Dict[str, Any], np.ndarray, np.ndarray]:
    Xtr = np.hstack([X_tab_train, emb_train])
    Xte = np.hstack([X_tab_test, emb_test])
    clf = LGBMClassifier(objective="multiclass", num_class=3, **LGBM_MULTICLASS_PARAMS)
    clf.fit(_lgbm_df(Xtr), y_train)
    y_pred = clf.predict(_lgbm_df(Xte))
    metrics = _multiclass_metrics_dict(y_test, y_pred)
    proba_test = clf.predict_proba(_lgbm_df(Xte))
    return clf, metrics, y_pred, proba_test


def binary_metrics_suite(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    y_hat = (y_score >= threshold).astype(int)
    out = {
        "auc_roc": float(roc_auc_score(y_true, y_score)),
        "auc_pr": float(average_precision_score(y_true, y_score)),
        "f1": float(f1_score(y_true, y_hat, zero_division=0)),
        "precision": float(precision_score(y_true, y_hat, zero_division=0)),
        "recall": float(recall_score(y_true, y_hat, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_score)),
    }
    return out


def metrics_by_criticality(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    by: Dict[str, Any] = {}
    crit = df["criticality_class"].astype(str).values
    for c in ["A", "B", "C"]:
        m = crit == c
        if m.sum() == 0:
            continue
        by[c] = binary_metrics_suite(y_true[m], y_score[m], threshold=threshold)
    return by


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def plot_loss_curve(losses: List[float], path: Path, title: str = "GAT training loss") -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(losses, color="#2c7fb8")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc_pr_pair(
    y_true: np.ndarray,
    s_a: np.ndarray,
    s_b: np.ndarray,
    label_a: str,
    label_b: str,
    roc_path: Path,
    pr_path: Path,
) -> None:
    fpr1, tpr1, _ = roc_curve(y_true, s_a)
    fpr2, tpr2, _ = roc_curve(y_true, s_b)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr1, tpr1, label=label_a)
    plt.plot(fpr2, tpr2, label=label_b)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.title("ROC — compliance")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close()

    p1, r1, _ = precision_recall_curve(y_true, s_a)
    p2, r2, _ = precision_recall_curve(y_true, s_b)
    plt.figure(figsize=(6, 5))
    plt.plot(r1, p1, label=label_a)
    plt.plot(r2, p2, label=label_b)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.title("Precision–recall — compliance")
    plt.tight_layout()
    plt.savefig(pr_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_score_by_crit(df: pd.DataFrame, score_col: str, path: Path) -> None:
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df, x="criticality_class", y=score_col, order=["A", "B", "C"])
    plt.title("Compliance risk score by criticality class (test)")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_calibration(y_true: np.ndarray, y_score: np.ndarray, path: Path, n_bins: int = 10) -> None:
    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=n_bins, strategy="uniform")
    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect")
    plt.xlabel("Mean predicted risk")
    plt.ylabel("Fraction positives")
    plt.legend()
    plt.title("Calibration (reliability)")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_calibration_uniform_vs_full_stratum(
    criticality: np.ndarray,
    y_true: np.ndarray,
    score_uniform: np.ndarray,
    score_full: np.ndarray,
    stratum: str,
    path: Path,
    n_bins: int = 8,
) -> bool:
    """
    Reliability diagram for one true-criticality stratum (test rows): uniform vs conditioned scores.
    Returns False if too few rows/events to draw a stable curve (no file written).
    """
    m = np.asarray(criticality).astype(str) == str(stratum)
    y = np.asarray(y_true).astype(int)[m]
    su = np.asarray(score_uniform, dtype=float)[m]
    sf = np.asarray(score_full, dtype=float)[m]
    if y.size < 50 or int(y.sum()) < 5:
        return False
    n_bins = int(max(3, min(n_bins, y.sum(), len(y) // 5)))
    plt.figure(figsize=(5.5, 4.5))
    for scores, label, color in [
        (su, "Uniform crit", "#1f77b4"),
        (sf, "Layer1-conditioned", "#ff7f0e"),
    ]:
        try:
            prob_true, prob_pred = calibration_curve(
                y, scores, n_bins=n_bins, strategy="uniform"
            )
        except ValueError:
            plt.close()
            return False
        plt.plot(prob_pred, prob_true, marker="o", label=label, color=color)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.35, label="Perfect")
    plt.xlabel("Mean predicted risk")
    plt.ylabel("Fraction positives")
    plt.legend(loc="lower right")
    plt.title(f"Calibration — test, true criticality {stratum} (n={len(y)}, positives={int(y.sum())})")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return True


def shap_bar_top_multiclass(
    model: LGBMClassifier,
    X: np.ndarray,
    feature_names: Sequence[str],
    path: Path,
    max_display: int = 10,
    sample_size: int = 1500,
) -> None:
    import shap

    rng = np.random.default_rng(42)
    n = min(sample_size, len(X))
    idx = rng.choice(len(X), size=n, replace=False)
    Xs_df = _lgbm_df(X[idx])
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(Xs_df)
    if isinstance(sv, list):
        imp = np.mean([np.abs(s) for s in sv], axis=0).mean(0)
    elif isinstance(sv, np.ndarray) and sv.ndim == 3:
        # LightGBM multiclass: (n_samples, n_features, n_classes)
        imp = np.abs(sv).mean(axis=(0, 2))
    else:
        imp = np.abs(np.asarray(sv)).mean(axis=0)
        if imp.ndim > 1:
            imp = imp.mean(axis=-1)
    order = np.argsort(-imp)[:max_display]
    plt.figure(figsize=(8, 5))
    plt.barh(range(len(order)), imp[order][::-1])
    plt.yticks(range(len(order)), [feature_names[i] for i in order[::-1]])
    plt.xlabel("Mean |SHAP|")
    plt.title(f"Top {max_display} features (multiclass mean |SHAP|)")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def shap_summary_binary_top(
    model: LGBMClassifier,
    X: np.ndarray,
    feature_names: Sequence[str],
    path: Path,
    max_display: int = 15,
    sample_size: int = 2000,
) -> None:
    import shap

    rng = np.random.default_rng(43)
    n = min(sample_size, len(X))
    idx = rng.choice(len(X), size=n, replace=False)
    Xs_df = _lgbm_df(X[idx])
    if len(feature_names) == Xs_df.shape[1]:
        Xs_df.columns = list(feature_names)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(Xs_df)
    if isinstance(sv, list):
        sv_use = sv[1] if len(sv) == 2 else sv[0]
    else:
        sv_use = sv
    shap.summary_plot(sv_use, Xs_df, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def oof_layer1_gat_lgbm(
    part_order: Sequence[str],
    X_tab: np.ndarray,
    y_all: np.ndarray,
    train_indices: np.ndarray,
    edge_index_full: torch.Tensor,
    n_splits: int = 5,
    gat_epochs_oof: int = 80,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, List[List[float]]]:
    """
    For each training part, out-of-fold multiclass probabilities from GAT+LGBM stack.
    oof_probs rows align with part_order (zeros for non-train rows).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = len(part_order)
    oof = np.zeros((N, 3), dtype=np.float64)
    y_tr = y_all[train_indices]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_losses: List[List[float]] = []
    for fold_id, (rel_tr, rel_va) in enumerate(skf.split(np.zeros(len(train_indices)), y_tr)):
        print(f"Layer1 OOF fold {fold_id + 1}/{n_splits}")
        tr_idx = train_indices[rel_tr]
        va_idx = train_indices[rel_va]
        scaler = StandardScaler()
        scaler.fit(X_tab[tr_idx])
        X_scaled = scaler.transform(X_tab)
        x_t = torch.tensor(X_scaled, dtype=torch.float32)
        edge_train = filter_edges_both_endpoints_in(edge_index_full, tr_idx)
        train_mask = torch.zeros(N, dtype=torch.bool)
        train_mask[tr_idx] = True
        y_t = torch.tensor(y_all, dtype=torch.long)
        model, losses = train_gat_classifier(
            x_t,
            edge_train,
            y_t,
            train_mask,
            epochs=gat_epochs_oof,
            device=device,
        )
        all_losses.append(losses)
        emb_full, _ = gat_embeddings_and_logits(model, x_t, edge_index_full, device=device)
        X_lgb_tr = np.hstack([X_scaled[tr_idx], emb_full[tr_idx]])
        X_lgb_va = np.hstack([X_scaled[va_idx], emb_full[va_idx]])
        clf = LGBMClassifier(objective="multiclass", num_class=3, **LGBM_MULTICLASS_PARAMS)
        clf.fit(_lgbm_df(X_lgb_tr), y_all[tr_idx])
        oof[va_idx] = clf.predict_proba(_lgbm_df(X_lgb_va))
        del model, clf
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    assert np.isfinite(oof[train_indices]).all(), "OOF probabilities incomplete"
    return oof, all_losses


def fit_final_layer1_gat_lgbm(
    part_order: Sequence[str],
    X_tab: np.ndarray,
    y_all: np.ndarray,
    train_indices: np.ndarray,
    edge_index_full: torch.Tensor,
    gat_epochs_final: int = 200,
    device: Optional[torch.device] = None,
) -> Tuple[StandardScaler, GATClassifier, LGBMClassifier, List[float], np.ndarray]:
    """Train scaler + GAT + stacking LGBM on all training parts; return test-ready artifacts."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = len(part_order)
    scaler = StandardScaler()
    scaler.fit(X_tab[train_indices])
    X_scaled = scaler.transform(X_tab)
    x_t = torch.tensor(X_scaled, dtype=torch.float32)
    edge_train = filter_edges_both_endpoints_in(edge_index_full, train_indices)
    train_mask = torch.zeros(N, dtype=torch.bool)
    train_mask[train_indices] = True
    y_t = torch.tensor(y_all, dtype=torch.long)
    model, losses = train_gat_classifier(
        x_t,
        edge_train,
        y_t,
        train_mask,
        epochs=gat_epochs_final,
        device=device,
    )
    emb_full, _ = gat_embeddings_and_logits(model, x_t, edge_index_full, device=device)
    X_lgb_tr = np.hstack([X_scaled[train_indices], emb_full[train_indices]])
    clf = LGBMClassifier(objective="multiclass", num_class=3, **LGBM_MULTICLASS_PARAMS)
    clf.fit(_lgbm_df(X_lgb_tr), y_all[train_indices])
    return scaler, model, clf, losses, emb_full


def layer1_predict_tabular_gat(
    scaler: StandardScaler,
    gat: GATClassifier,
    lgbm: LGBMClassifier,
    X_tab: np.ndarray,
    edge_index_full: torch.Tensor,
    part_indices: np.ndarray,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    X_scaled = scaler.transform(X_tab)
    x_t = torch.tensor(X_scaled, dtype=torch.float32)
    emb_full, _ = gat_embeddings_and_logits(gat, x_t, edge_index_full, device=device)
    X_stack = np.hstack([X_scaled[part_indices], emb_full[part_indices]])
    return lgbm.predict_proba(_lgbm_df(X_stack))


def attach_crit_probs_to_panel(
    df: pd.DataFrame,
    part_order: Sequence[str],
    probs_per_part_index: np.ndarray,
) -> pd.DataFrame:
    """Map (N,3) part-index-aligned probs to each row by part_id."""
    idx_map = {p: i for i, p in enumerate(part_order)}
    A, B, C = [], [], []
    for pid in df["part_id"].astype(str).values:
        j = idx_map[pid]
        A.append(probs_per_part_index[j, 0])
        B.append(probs_per_part_index[j, 1])
        C.append(probs_per_part_index[j, 2])
    out = df.copy()
    out["crit_prob_A"] = A
    out["crit_prob_B"] = B
    out["crit_prob_C"] = C
    return out


def business_value_summary(
    crit: np.ndarray,
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: Sequence[float],
    cost_by_class: Mapping[str, float],
    fp_cost: float,
) -> pd.DataFrame:
    """
    Baseline assumes no intervention: every compliance failure pays undetected class cost.
    With model at threshold: FN still pay class cost; FP pay fp_cost; TP avoid failure cost.
    Net value = baseline_cost - (FN_cost + FP_cost).
    """
    rows = []
    crit = np.asarray(crit).astype(str)
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    baseline = sum(float(cost_by_class[crit[i]]) for i in range(len(y_true)) if y_true[i] == 1)
    for thr in thresholds:
        pred = y_score >= thr
        fn_cost = 0.0
        fp_cost_tot = 0.0
        tp_by = {"A": 0, "B": 0, "C": 0}
        fp_by = {"A": 0, "B": 0, "C": 0}
        fn_by = {"A": 0, "B": 0, "C": 0}
        for i in range(len(y_true)):
            c = crit[i]
            if pred[i] and y_true[i] == 0:
                fp_cost_tot += fp_cost
                fp_by[c] = fp_by.get(c, 0) + 1
            elif (not pred[i]) and y_true[i] == 1:
                fn_cost += float(cost_by_class[c])
                fn_by[c] = fn_by.get(c, 0) + 1
            elif pred[i] and y_true[i] == 1:
                tp_by[c] = tp_by.get(c, 0) + 1
        model_cost = fn_cost + fp_cost_tot
        rows.append(
            dict(
                threshold=thr,
                baseline_failure_cost=baseline,
                fn_cost=fn_cost,
                fp_cost=fp_cost_tot,
                model_total_cost=model_cost,
                net_value=baseline - model_cost,
                tp_A=tp_by["A"],
                tp_B=tp_by["B"],
                tp_C=tp_by["C"],
                fp_A=fp_by["A"],
                fp_B=fp_by["B"],
                fp_C=fp_by["C"],
                fn_A=fn_by["A"],
                fn_B=fn_by["B"],
                fn_C=fn_by["C"],
            )
        )
    return pd.DataFrame(rows)


def _l2_classifier_params(scale_pos_weight: float) -> dict:
    """Layer 2 LightGBM params (env overrides for leaves / lr / class weight)."""
    nl = int(os.environ.get("L2_NUM_LEAVES", "127"))
    lr = float(os.environ.get("L2_LEARNING_RATE", "0.05"))
    cw = os.environ.get("L2_CLASS_WEIGHT", "").strip().lower()
    params = {**LGBM_BINARY_PARAMS, "num_leaves": nl, "learning_rate": lr}
    if cw == "balanced":
        params["class_weight"] = "balanced"
    else:
        params["scale_pos_weight"] = scale_pos_weight
    return params


def run_layer2_evaluation(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    part_catalog: pd.DataFrame,
    train_parts: np.ndarray,
    train_crit_mat: np.ndarray,
    test_crit_mat: np.ndarray,
    layer1_mode: str,
    out_dir: Path,
) -> Dict[str, Any]:
    """
    Train uniform / conditioned / oracle Layer 2 models; write CSV + JSON artifacts.

    Uses validation-part threshold tuning (headline) and fixed 0.5 threshold metrics.
    """
    temp = float(os.environ.get("CRIT_PROB_SHARPEN", "0.88"))
    train_crit_mat = sharpen_crit_probs(train_crit_mat, temp)
    test_crit_mat = sharpen_crit_probs(test_crit_mat, temp)

    df_tr_u = attach_uniform_crit_probs(df_train)
    df_te_u = attach_uniform_crit_probs(df_test)
    df_tr_f = attach_crit_prob_matrix(df_train, train_crit_mat)
    df_te_f = attach_crit_prob_matrix(df_test, test_crit_mat)
    df_tr_o = attach_oracle_crit_probs(df_train)
    df_te_o = attach_oracle_crit_probs(df_test)

    l2_cols = get_layer2_model_feature_columns(layer1_mode=layer1_mode)
    y_tr = df_train["compliance_failure"].to_numpy(dtype=int)
    y_te = df_test["compliance_failure"].to_numpy(dtype=int)

    # Genuine held-out fit/val split *within* train (fit_parts and val_parts are disjoint),
    # via part_level_train_val_split -- also works when df_train has been scoped down (e.g.
    # LAYER2_SCOPE=real_category_only) since it's derived from parts actually present in df_train,
    # not the raw `train_parts` arg.
    #
    # NOTE: this replaces a prior bug where s_val was produced by scoring the SAME model used for
    # s_te (fit on all of df_train) on a subset of its own training rows -- in-sample, not held
    # out. That inflated threshold_validation_max_f1_report's validation F1 (it hit the ~1.0
    # ceiling at LAYER2_SCOPE=real_category_only's smaller N, where a 127-leaf LightGBM can nearly
    # memorize ~1.9k training rows). Now s_val comes from a model fit only on fit_mask rows.
    train_parts_present = df_train["part_id"].astype(str).unique()
    fit_parts, val_parts = part_level_train_val_split(
        part_catalog, train_parts_present, val_fraction=0.2, random_state=43
    )
    fit_mask = df_train["part_id"].isin(fit_parts).to_numpy()
    val_mask = df_train["part_id"].isin(val_parts).to_numpy()
    y_val = y_tr[val_mask]

    spw = int((y_tr == 0).sum()) / max(int((y_tr == 1).sum()), 1)
    l2_params = _l2_classifier_params(spw)

    def _fit_predict(df_tr: pd.DataFrame, df_te: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """s_te: production model, fit on all of df_tr, scored on held-out test parts (df_te).
        s_val: a separate model, fit only on fit_mask rows (excluding val_mask rows), scored on
        val_mask rows -- genuinely held out, unlike scoring the production model on its own train
        data."""
        Xtr = df_tr[l2_cols].to_numpy(dtype=np.float64)
        Xte = df_te[l2_cols].to_numpy(dtype=np.float64)
        m = LGBMClassifier(objective="binary", **l2_params)
        m.fit(_lgbm_df(Xtr), y_tr)
        s_te = m.predict_proba(_lgbm_df(Xte))[:, 1]

        m_val = LGBMClassifier(objective="binary", **l2_params)
        m_val.fit(_lgbm_df(Xtr[fit_mask]), y_tr[fit_mask])
        s_val = m_val.predict_proba(_lgbm_df(Xtr[val_mask]))[:, 1]
        return s_te, s_val

    s_uni_te, s_uni_val = _fit_predict(df_tr_u, df_te_u)
    s_cond_te, s_cond_val = _fit_predict(df_tr_f, df_te_f)
    s_orac_te, s_orac_val = _fit_predict(df_tr_o, df_te_o)

    thr_def = 0.5
    m_uni = binary_metrics_suite(y_te, s_uni_te, threshold=thr_def)
    m_cond = binary_metrics_suite(y_te, s_cond_te, threshold=thr_def)
    m_orac = binary_metrics_suite(y_te, s_orac_te, threshold=thr_def)
    by_uni = metrics_by_criticality(df_test, y_te, s_uni_te, threshold=thr_def)
    by_cond = metrics_by_criticality(df_test, y_te, s_cond_te, threshold=thr_def)
    by_orac = metrics_by_criticality(df_test, y_te, s_orac_te, threshold=thr_def)

    rep_val_u = threshold_validation_max_f1_report("uniform", y_val, s_uni_val, y_te, s_uni_te)
    rep_val_c = threshold_validation_max_f1_report("conditioned", y_val, s_cond_val, y_te, s_cond_te)
    rep_val_o = threshold_validation_max_f1_report("oracle", y_val, s_orac_val, y_te, s_orac_te)

    biz_thrs = [float(x) for x in os.environ.get("BUSINESS_THRESHOLDS", "0.10,0.15,0.20,0.25").split(",")]
    biz_rows: List[Dict[str, Any]] = []
    for name, scores in [
        ("uniform", s_uni_te),
        ("conditioned", s_cond_te),
        ("oracle", s_orac_te),
    ]:
        biz_rows.extend(metrics_at_fixed_thresholds(name, y_te, scores, biz_thrs))

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"model": "uniform", **m_uni},
            {"model": "conditioned", **m_cond},
            {"model": "oracle", **m_orac},
        ]
    ).to_csv(out_dir / "compliance_comparison.csv", index=False)
    pd.DataFrame([rep_val_u, rep_val_c, rep_val_o]).to_csv(
        out_dir / "compliance_comparison_val_threshold.csv", index=False
    )
    pd.DataFrame(biz_rows).to_csv(out_dir / "compliance_comparison_business_thresholds.csv", index=False)

    cost_by = {
        "A": float(os.environ.get("COST_A", "50000")),
        "B": float(os.environ.get("COST_B", "10000")),
        "C": float(os.environ.get("COST_C", "2000")),
    }
    fp_cost = float(os.environ.get("FP_COST", "500"))
    crit_te = df_test["criticality_class"].astype(str).to_numpy()
    business_value_summary(crit_te, y_te, s_cond_te, biz_thrs, cost_by, fp_cost).to_csv(
        out_dir / "business_value_simulation.csv", index=False
    )

    save_json(
        out_dir / "layer2_baseline_uniform.json",
        {"overall": m_uni, "by_criticality_true": by_uni, "validation_threshold": rep_val_u},
    )
    save_json(
        out_dir / "layer2_full_conditioned.json",
        {"overall": m_cond, "by_criticality_true": by_cond, "validation_threshold": rep_val_c},
    )
    save_json(
        out_dir / "layer2_oracle_true_class.json",
        {"overall": m_orac, "by_criticality_true": by_orac, "validation_threshold": rep_val_o},
    )

    manifest = {
        "compliance_grain": get_compliance_grain(),
        "layer1_features": layer1_mode,
        "crit_prob_sharpen": temp,
        "l2_num_leaves": int(os.environ.get("L2_NUM_LEAVES", "127")),
        "train_rows": int(len(df_train)),
        "test_rows": int(len(df_test)),
        "train_failure_rate": float(y_tr.mean()),
        "test_failure_rate": float(y_te.mean()),
    }
    save_json(out_dir / "modeling_manifest.json", manifest)

    return {
        "uniform_auc_pr": m_uni["auc_pr"],
        "conditioned_auc_pr": m_cond["auc_pr"],
        "oracle_auc_pr": m_orac["auc_pr"],
        "delta_auc_pr_cond_minus_uniform": m_cond["auc_pr"] - m_uni["auc_pr"],
        "uniform_brier": m_uni["brier"],
        "conditioned_brier": m_cond["brier"],
        "y_test": y_te,
        "scores_test": {
            "uniform": s_uni_te,
            "conditioned": s_cond_te,
            "oracle": s_orac_te,
        },
        "metrics": {"uniform": m_uni, "conditioned": m_cond, "oracle": m_orac},
        "by_criticality": {"uniform": by_uni, "conditioned": by_cond, "oracle": by_orac},
        **manifest,
    }
