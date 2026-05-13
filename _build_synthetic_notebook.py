#!/usr/bin/env python3
"""One-off builder for synthetic_data_generator.ipynb (kept for reproducibility)."""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

cells: list[dict] = []


def md(s: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(s).strip("\n").splitlines(True)})


def code(s: str) -> None:
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": textwrap.dedent(s).strip("\n").splitlines(True),
        }
    )


md(
    r"""
# Synthetic hybrid dataset generator (criticality-aware inventory)

This notebook builds a **hybrid** research dataset for a two-layer framework: (1) ABC **criticality classification** and (2) **schedule compliance / failure risk** conditioned on criticality.

**Hybrid strategy (methodology-facing).** Public datasets supply the **structural backbone** (realistic order timing, shipment performance, SKU demand/price dispersion, and product taxonomy). Literature-informed **synthetic augmentation** fills gaps that public retail / generic supply-chain exports do not cover for **automotive** and **criticality-specific** modeling (multi-tier BOM topology, supplier qualification counts, substitutability priors, cascade exposure scores, and calibrated compliance labels).

> **Reproducibility:** all stochastic steps use `RANDOM_SEED`. **Sensitivity analysis:** adjust constants in the first code cell only.
"""
)

code(
    r"""
# =============================================================================
# 1) Setup and imports
# =============================================================================
from __future__ import annotations

import hashlib
import io
import os
import pickle
import re
import warnings
from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from scipy import stats
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------------------------------------------------------
# Global constants (edit for sensitivity analysis)
# -----------------------------------------------------------------------------
N_PARTS = 5000
RANDOM_SEED = 42

# ABC nominal shares (rule-based proxy classifier targets)
A_SHARE, B_SHARE, C_SHARE = 0.20, 0.30, 0.50

# Lead time priors by criticality class (weeks) — synthetic augmentation
LT_MEAN_WEEKS = {"A": 5.0, "B": 7.5, "C": 11.0}
LT_CV = {"A": 0.20, "B": 0.28, "C": 0.32}

# Supplier base (qualified suppliers) — synthetic augmentation
N_SUPPLIERS_BY_CLASS = {"A": (1, 2), "B": (2, 4), "C": (3, 6)}

# Substitutability Bernoulli probabilities by class — synthetic augmentation
P_SUBSTITUTABLE = {"A": 0.08, "B": 0.35, "C": 0.72}

# Stockout events per year (Poisson means) — synthetic augmentation
STOCKOUT_RATE_BY_CLASS = {"A": 0.35, "B": 1.1, "C": 2.4}

# BOM generator — automotive-style fan-out (CIRP Annals automotive BOM literature)
BOM_DEPTH_MIN, BOM_DEPTH_MAX = 3, 5
BOM_FANOUT_MIN, BOM_FANOUT_MAX = 3, 8

# Supplier risk augmentation — synthetic layer on top of observed OTD
AT_RISK_SUPPLIER_SHARE = 0.15
OTD_TARGET_HIGH = 0.985
OTD_ESCALATION_THRESHOLD = 0.97
OTD_AT_RISK_MEAN = 0.955

# OEM vs supplier-reported OTD gap (“reschedule burden”), percentage points
RESCHEDULE_BURDEN_PP_MIN, RESCHEDULE_BURDEN_PP_MAX = 4.0, 12.0

# Monthly panel horizon
N_MONTHS = 24

# Compliance label generator — calibrated logistic layer (synthetic outcome model)
COMPLIANCE_FAILURE_RATE_TARGET = 0.08
COMPLIANCE_FAILURE_RATE_MAX = 0.20

# Paths
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATACO_CSV_PATH = Path(os.environ.get("DATACO_CSV_PATH", "data/DataCoSupplyChainDataset.csv"))
UCI_ONLINE_RETAIL_URLS = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx",
    "https://github.com/gagolews/teaching-data/raw/master/marek/OnlineRetail.xlsx",
)

DATACO_FALLBACK_URL = os.environ.get("DATACO_FALLBACK_URL", "").strip()

np.random.seed(RANDOM_SEED)
rng = np.random.default_rng(RANDOM_SEED)

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.figsize"] = (10, 5)

_ = (sklearn, StandardScaler, LogisticRegression)  # required imports for downstream modeling notebooks
print("Constants loaded. OUT_DIR =", OUT_DIR.resolve())
"""
)

md(
    r"""
## 2) Load public datasets (structural backbone)

### DataCo Global Supply Chain (Kaggle / Mendeley Data)

**What it provides for this paper.** Realistic order records with **scheduled vs realized shipping delays**, **delivery status**, **category taxonomy**, **quantities**, **discounts**, and **sales** — a strong backbone for **on-time delivery (OTD)** behavior and **category-level performance baselines**.

**Limitations.** The dataset is **not automotive OEM-specific**; it reflects a consumer-goods supply chain. Customer/partner identifiers are **not trustworthy automotive supplier IDs**, so supplier entities in later steps are **derived clusters** (e.g., market × category), with augmentation for automotive risk patterns.

**Citation.** Constante, Fabian; Silva, Fernando; Pereira, António (2019), “DataCo SMART SUPPLY CHAIN FOR BIG DATA ANALYSIS”, Mendeley Data, V5, doi:10.17632/8gx2fvg2k6.5; Kaggle mirror: `shashwatwork/dataco-smart-supply-chain-for-big-data-analysis`.

### UCI Online Retail

**What it provides.** Transaction-level **SKU demand** and **unit price** dispersion suitable for demand variability features and price-sensitive criticality proxies.

**Limitations.** It is **e-commerce retail**, not production BOMs; SKUs are not automotive part numbers.

**Citation.** Daqing Chen, Sai Liang Sain, Kun Guo, (2012) Online Retail, UCI ML Repository, https://doi.org/10.24432/C5BW3K.
"""
)

code(
    r"""
# =============================================================================
# 2) Load public datasets
# =============================================================================

PII_DROP_SUBSTR = ("email", "password", "fname", "lname", "street", "customer zip", "zipcode")


def _snake(s: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z]+", "_", str(s).strip().lower()).strip("_")
    return re.sub(r"_+", "_", s)


def load_dataco_supply_chain(
    csv_path: Optional[Path] = None,
    fallback_url: Optional[str] = None,
) -> pd.DataFrame:
    # Load DataCo from a local CSV. Remote fetch is opt-in (DATACO_FALLBACK_URL).
    use_path = (csv_path or DATACO_CSV_PATH).expanduser()
    if use_path.exists():
        df = pd.read_csv(use_path, low_memory=False, encoding="latin1")
        source = f"local_file:{use_path}"
    else:
        url = (fallback_url or DATACO_FALLBACK_URL or "").strip()
        if not url:
            raise FileNotFoundError(
                f"DataCo CSV not found at {use_path.resolve()}.\n"
                "Download `DataCoSupplyChainDataset.csv` from Kaggle "
                "(shashwatwork/dataco-smart-supply-chain-for-big-data-analysis) or Mendeley Data "
                "(doi:10.17632/8gx2fvg2k6.5), place it under `data/DataCoSupplyChainDataset.csv`, "
                "or set DATACO_CSV_PATH.\n"
                "Optional: set DATACO_FALLBACK_URL to an HTTPS URL if you must fetch remotely."
            )
        r = requests.get(url, timeout=180)
        r.raise_for_status()
        df = pd.read_csv(io.BytesIO(r.content), low_memory=False, encoding="latin1")
        source = f"url:{url}"

    df.columns = [_snake(c) for c in df.columns]

    keep_cols = [c for c in df.columns if not any(p in c for p in PII_DROP_SUBSTR)]
    df = df[keep_cols]

    rename_map = {
        "order_date_dateorders": "order_date",
        "shipping_date_dateorders": "ship_date",
        "days_for_shipping_real": "days_shipping_real",
        "days_for_shipment_scheduled": "days_shipment_scheduled",
        "category_name": "product_category",
        "order_item_discount_rate": "order_item_discount_rate",
        "order_item_quantity": "order_quantity",
        "sales": "sales",
        "delivery_status": "delivery_status",
        "product_name": "product_name",
        "market": "market",
        "order_region": "order_region",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    required = [
        "order_date",
        "ship_date",
        "days_shipping_real",
        "days_shipment_scheduled",
        "delivery_status",
        "product_category",
        "order_quantity",
        "order_item_discount_rate",
        "sales",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"DataCo schema missing columns {missing} after normalization (source={source}).")

    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df["ship_date"] = pd.to_datetime(df["ship_date"], errors="coerce")

    df["days_late_ship"] = (
        pd.to_numeric(df["days_shipping_real"], errors="coerce")
        - pd.to_numeric(df["days_shipment_scheduled"], errors="coerce")
    )
    df["on_time_delivery"] = (df["days_late_ship"] <= 0).astype("Int64")

    df["lead_time_days_calendar"] = (df["ship_date"] - df["order_date"]).dt.days
    df["on_time_delivery_calendar"] = (
        df["lead_time_days_calendar"] <= pd.to_numeric(df["days_shipment_scheduled"], errors="coerce")
    ).astype("Int64")

    df["dataset"] = "dataco"
    df.attrs["load_source"] = source
    return df


def load_uci_online_retail(urls: Sequence[str] = UCI_ONLINE_RETAIL_URLS) -> pd.DataFrame:
    # Load UCI Online Retail from a URL list (xlsx).
    last_err: Optional[Exception] = None
    df = None
    for url in urls:
        try:
            r = requests.get(url, timeout=180)
            r.raise_for_status()
            df = pd.read_excel(io.BytesIO(r.content), engine="openpyxl")
            df.attrs["load_source"] = f"url:{url}"
            break
        except Exception as e:
            last_err = e
            df = None
    if df is None:
        raise RuntimeError(f"Failed to download UCI Online Retail from all URLs. Last error: {last_err}")

    df.columns = [_snake(c) for c in df.columns]
    need = {"stockcode", "description", "quantity", "unitprice", "invoicedate", "customerid"}
    missing = sorted(need - set(df.columns))
    if missing:
        raise ValueError(f"UCI schema missing {missing} after normalization.")

    df = df.rename(
        columns={
            "stockcode": "stock_code",
            "description": "description",
            "quantity": "quantity",
            "unitprice": "unit_price",
            "invoicedate": "invoice_date",
            "customerid": "customer_id",
        }
    )

    df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")

    df = df.dropna(subset=["stock_code", "invoice_date"])
    df = df[df["quantity"] > 0]
    df = df[df["unit_price"] >= 0]

    df["dataset"] = "uci_online_retail"
    return df


dataco_raw = load_dataco_supply_chain()
uci_raw = load_uci_online_retail()

assert len(dataco_raw) > 0
assert len(uci_raw) > 0
assert dataco_raw["on_time_delivery"].notna().mean() > 0.5

print("DataCo:", dataco_raw.attrs.get("load_source"), "rows=", len(dataco_raw))
print("UCI:", uci_raw.attrs.get("load_source"), "rows=", len(uci_raw))
"""
)

md(
    r"""
## 3) Part catalog construction (public SKU backbone + literature augmentation)

**Public inputs.** Unique product identities and category labels from **DataCo** (`product_name`, `product_category`) and **UCI** (`stock_code`, `description`). Demand variability and price moments come from **UCI** transactions aggregated to SKU; **DataCo** contributes category anchors and line-sales / quantity signals.

**Synthetic augmentation (gap filling).** Multi-echelon **lead time mean/CV**, **qualified supplier counts**, **substitutability**, and **stockout** priors are not present in these public sources for automotive programs; we draw **class-conditional** parameters consistent with the literature ranges noted in the paper prompt (documented again in `data_dictionary.md`).

**Criticality labels (proxy).** ABC labels are generated with a **transparent rule-based** score combining **unit price** and **demand coefficient of variation**, then **quantile-cut** to match target ABC shares. This stands in for the paper’s eventual ML classifier.
"""
)

code(
    r"""
# =============================================================================
# 3) Part catalog construction
# =============================================================================


def _base_part_id(namespace: str, dedup_key: str) -> str:
    digest = hashlib.sha256(f"{namespace}|{dedup_key}".encode("utf-8")).hexdigest()[:24]
    return f"{namespace[:3].upper()}_{digest}"


def _assign_part_ids_with_collision_suffix(namespace_keys: pd.Series, dedup_keys: pd.Series) -> list[str]:
    bases = [_base_part_id(str(ns), str(dk)) for ns, dk in zip(namespace_keys, dedup_keys)]
    counts: Dict[str, int] = {}
    out: list[str] = []
    for b in bases:
        k = counts.get(b, 0)
        if k == 0:
            out.append(b)
        else:
            out.append(f"{b}_dup{k}")
        counts[b] = k + 1
    return out


def build_uci_sku_stats(uci: pd.DataFrame) -> pd.DataFrame:
    def _mode_nonempty(s: pd.Series) -> str:
        s2 = s.dropna().astype(str)
        if s2.empty:
            return ""
        return str(s2.mode().iloc[0])

    g = (
        uci.groupby("stock_code", observed=True)
        .agg(
            uci_description=("description", _mode_nonempty),
            uci_median_unit_price=("unit_price", "median"),
            uci_mean_unit_price=("unit_price", "mean"),
            uci_demand_mean_monthly=("quantity", "mean"),
            uci_demand_cv_monthly=(
                "quantity",
                lambda s: float(np.std(s, ddof=1) / np.mean(s)) if float(np.mean(s)) > 0 else np.nan,
            ),
            uci_n_invoices=("quantity", "size"),
        )
        .reset_index()
    )
    return g


def build_dataco_product_stats(dataco: pd.DataFrame) -> pd.DataFrame:
    g = (
        dataco.dropna(subset=["product_name"])
        .groupby(["product_name", "product_category"], observed=True)
        .agg(
            dataco_median_line_sales=("sales", "median"),
            dataco_mean_order_qty=("order_quantity", "mean"),
            dataco_on_time_rate=("on_time_delivery", "mean"),
        )
        .reset_index()
    )
    return g


def assign_abc_by_price_demand_cv(
    df: pd.DataFrame,
    price_col: str,
    cv_col: str,
    rng: np.random.Generator,
    a_share: float,
    b_share: float,
    c_share: float,
) -> pd.DataFrame:
    x = np.log1p(df[price_col].clip(lower=1e-6))
    zcv = df[cv_col].replace([np.inf, -np.inf], np.nan)
    zcv = zcv.fillna(zcv.median())
    zcv = (zcv - zcv.mean()) / (zcv.std(ddof=1) + 1e-6)
    score = x + 0.75 * zcv + 0.05 * rng.normal(size=len(df))
    q1, q2 = np.quantile(score, [a_share, a_share + b_share])
    cls = np.where(score <= q1, "A", np.where(score <= q2, "B", "C"))
    out = df.copy()
    out["criticality_class"] = cls
    return out


def augment_literature_part_features(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    df = df.copy()

    def draw_supplier_count(row) -> int:
        lo, hi = N_SUPPLIERS_BY_CLASS[row["criticality_class"]]
        return int(rng.integers(lo, hi + 1))

    df["n_qualified_suppliers"] = df.apply(draw_supplier_count, axis=1)
    df["substitutable_flag"] = df["criticality_class"].map(lambda k: int(rng.random() < P_SUBSTITUTABLE[k]))
    df["stockout_events_per_year"] = df["criticality_class"].map(
        lambda k: float(max(0.0, rng.poisson(STOCKOUT_RATE_BY_CLASS[k])))
    )

    def draw_lt(row):
        k = row["criticality_class"]
        mean_w = float(rng.normal(LT_MEAN_WEEKS[k], 0.35))
        mean_w = float(np.clip(mean_w, 4.0 if k == "A" else 6.0, 6.0 if k == "A" else 14.0))
        cv = float(rng.normal(LT_CV[k], 0.03))
        cv = float(np.clip(cv, 0.15, 0.35))
        sigma_w = max(1e-3, cv * mean_w)
        return mean_w, cv, sigma_w

    lt = df.apply(draw_lt, axis=1, result_type="expand")
    df["lead_time_mean_weeks"] = lt[0]
    df["lead_time_cv"] = lt[1]
    df["lead_time_sigma_weeks"] = lt[2]

    df["lead_time_source"] = "synthetic_literature_augmentation"
    df["supplier_count_source"] = "synthetic_literature_augmentation"
    df["substitutability_source"] = "synthetic_literature_augmentation"
    df["stockout_source"] = "synthetic_literature_augmentation"
    return df


def build_unified_part_catalog(
    dataco: pd.DataFrame,
    uci: pd.DataFrame,
    n_target: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    uci_stats = build_uci_sku_stats(uci)
    dataco_stats = build_dataco_product_stats(dataco)

    uci_parts = uci_stats.copy()
    uci_parts["namespace"] = "UCI"
    uci_parts["display_name"] = uci_parts["stock_code"].astype(str) + " | " + uci_parts["uci_description"].astype(str).str.slice(
        0, 80
    )
    uci_parts["product_category"] = "UCI_ONLINE_RETAIL"
    uci_parts["public_backbone"] = "UCI Online Retail (SKU stats)"

    dc_parts = dataco_stats.rename(columns={"product_name": "display_name"}).copy()
    dc_parts["stock_code"] = np.nan
    dc_parts["namespace"] = "DCO"
    dc_parts["public_backbone"] = "DataCo (product/category grain)"

    common_cols = [
        "namespace",
        "display_name",
        "product_category",
        "uci_median_unit_price",
        "uci_mean_unit_price",
        "uci_demand_mean_monthly",
        "uci_demand_cv_monthly",
        "uci_n_invoices",
        "dataco_median_line_sales",
        "dataco_mean_order_qty",
        "dataco_on_time_rate",
        "public_backbone",
        "stock_code",
    ]

    for c in common_cols:
        if c not in uci_parts.columns:
            uci_parts[c] = np.nan
        if c not in dc_parts.columns:
            dc_parts[c] = np.nan

    both = pd.concat([uci_parts[common_cols], dc_parts[common_cols]], ignore_index=True)

    both["dedup_key"] = both["display_name"].astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    both = both.drop_duplicates(subset=["dedup_key"], keep="first").reset_index(drop=True)

    price = both["uci_median_unit_price"].fillna(
        both["dataco_median_line_sales"].fillna(0.0) / (both["dataco_mean_order_qty"].fillna(1.0) + 1.0)
    )
    cv = both["uci_demand_cv_monthly"].fillna(0.35)

    both["abc_price_proxy"] = price.clip(lower=1e-6)
    both["abc_demand_cv_proxy"] = cv

    both = assign_abc_by_price_demand_cv(both, "abc_price_proxy", "abc_demand_cv_proxy", rng, A_SHARE, B_SHARE, C_SHARE)

    if len(both) >= n_target:
        both = both.sample(n=n_target, random_state=int(rng.integers(1_000_000))).reset_index(drop=True)
    else:
        need = n_target - len(both)
        synth_rows = []
        for i in range(need):
            row = both.sample(1, random_state=int(rng.integers(1_000_000))).iloc[0].to_dict()
            row["namespace"] = "SYN"
            row["display_name"] = f"SYNTHETIC_VARIANT_{i:05d} | {row['display_name']}"
            row["dedup_key"] = str(row["display_name"]).lower()
            row["public_backbone"] = "Synthetic SKU variant (cloned moments from public catalog)"
            synth_rows.append(row)
        both = pd.concat([both, pd.DataFrame(synth_rows)], ignore_index=True)
        both = both.drop_duplicates(subset=["dedup_key"], keep="first").reset_index(drop=True)

        if len(both) < n_target:
            extra = n_target - len(both)
            pad_rows = []
            for i in range(extra):
                row = both.sample(1, random_state=int(rng.integers(1_000_000))).iloc[0].to_dict()
                row["namespace"] = "SYN"
                row["display_name"] = f"SYNTHETIC_PAD_{i:05d} | {row['display_name']}"
                row["dedup_key"] = str(row["display_name"]).lower()
                row["public_backbone"] = "Synthetic padding SKU"
                pad_rows.append(row)
            both = pd.concat([both, pd.DataFrame(pad_rows)], ignore_index=True).reset_index(drop=True)

    both = both.reset_index(drop=True)
    both["part_id"] = _assign_part_ids_with_collision_suffix(both["namespace"], both["dedup_key"])

    both = augment_literature_part_features(both, rng)

    if not both["part_id"].is_unique:
        raise AssertionError("Internal error: part_id must be unique after catalog build.")

    shares = both["criticality_class"].value_counts(normalize=True)
    assert abs(float(shares.get("A", 0.0)) - A_SHARE) < 0.03
    assert abs(float(shares.get("B", 0.0)) - B_SHARE) < 0.04
    assert abs(float(shares.get("C", 0.0)) - C_SHARE) < 0.06

    assert both["lead_time_cv"].between(0.15, 0.35).mean() > 0.90
    assert both["n_qualified_suppliers"].between(1, 6).all()

    return both


part_catalog = build_unified_part_catalog(dataco_raw, uci_raw, N_PARTS, rng)
print(part_catalog[["criticality_class", "public_backbone"]].value_counts().head(10))
print(part_catalog.head(3).T)
"""
)

md(
    r"""
## 4) BOM graph generator (NetworkX DAG + topology features)

**Implementation.** Logic lives in **`bom_graph.py`** at the project root. The notebook loads it with **`runpy.run_path`** so Jupyter always runs the file on disk (no stale inline `def generate_bom_dag`).

**What it builds.** A **directed acyclic graph** over the `N_PARTS` catalog with **tiered** assembly structure, **automotive-style fan-out** between tiers, and **BOM position** features.

**Duplicate parts in real BOMs vs this notebook.** Multi-use of the same SKU is **multiple edges** from one node. **Duplicate `part_id` rows** in `part_catalog` are invalid here (NetworkX node = label).

**Data sources.** Topology is **synthetic**; depth/fan-out are literature-motivated priors (see `data_dictionary.md`).

**Edge direction.** `component -> assembly`.
"""
)

code(
    r"""
# =============================================================================
# 4) BOM graph generator
# =============================================================================
# Implementation is in **bom_graph.py** (loaded with runpy) so Jupyter cannot keep a stale
# copy of `generate_bom_dag` inside this cell after edits — a common source of confusion.
from pathlib import Path
import runpy

_repo = Path.cwd()
if not (_repo / "bom_graph.py").is_file():
    _repo = Path.cwd().parent
if not (_repo / "bom_graph.py").is_file():
    raise FileNotFoundError(
        "Could not find bom_graph.py next to the notebook (current working directory) or in the parent folder. "
        "Open Jupyter from the repository root (directory containing bom_graph.py), or chdir there, then restart the kernel."
    )
_bom_path = (_repo / "bom_graph.py").resolve()
_bom = runpy.run_path(str(_bom_path))
generate_bom_dag = _bom["generate_bom_dag"]
compute_bom_position_features = _bom["compute_bom_position_features"]
print("[BOM] loaded from", str(_bom_path.resolve()), flush=True)


crit_series = part_catalog.set_index("part_id")["criticality_class"]
bom_graph = generate_bom_dag(
    part_ids=part_catalog["part_id"].tolist(),
    depth_min=BOM_DEPTH_MIN,
    depth_max=BOM_DEPTH_MAX,
    fanout_min=BOM_FANOUT_MIN,
    fanout_max=BOM_FANOUT_MAX,
    criticality=crit_series,
    rng=rng,
)

bom_features = compute_bom_position_features(bom_graph, crit_series.to_dict())
part_catalog = part_catalog.set_index("part_id").join(bom_features, how="left").reset_index()

assert part_catalog["bom_in_degree"].between(0, 500).all()
assert (part_catalog["bom_longest_downstream_path"] >= 0).all()
assert float(part_catalog["bom_in_degree"].mean()) > 1.0
_cascade = part_catalog["bom_criticality_propagation_score"].astype(float)
assert float(_cascade.notna().mean()) >= 0.99
assert np.isfinite(_cascade.fillna(0.0)).all()

print(bom_graph)
print(part_catalog[["bom_in_degree", "bom_out_degree", "bom_longest_downstream_path"]].describe())
"""
)

md(
    r"""
## 5) Supplier performance history (DataCo-anchored monthly panel + automotive augmentation)

**Public backbone.** We aggregate **observed** DataCo line-level OTD to a **category × month** baseline.

**Synthetic augmentation (gap filling).** Public rows do not include **automotive supplier quality programs**, **Tier-1 OTD targets**, or **OEM–supplier reporting gaps**. We therefore:
- instantiate a **supplier_id** as a stable hash of **(market, category)** (derived entity, not a real supplier DUNS),
- mark **15%** of suppliers as **at-risk** with degraded OTD draws anchored near industry escalation narratives,
- add **reschedule burden** (percentage points) as a gap between supplier-reported and OEM-side OTD.

**Literature / industry anchors.** APQC / automotive supplier quality discourse motivates very high OTD targets for Tier-1 programs; symestic-style industry commentary motivates **single-digit to low-double-digit percentage-point** gaps between supplier-reported and OEM-measured performance. (Citations are consolidated in `data_dictionary.md`.)
"""
)

code(
    r"""
# =============================================================================
# 5) Supplier performance history
# =============================================================================


def _month_period(s: pd.Series) -> pd.Series:
    return s.dt.to_period("M").dt.to_timestamp()


def build_category_month_otd_baseline(dataco: pd.DataFrame) -> pd.DataFrame:
    df = dataco.dropna(subset=["order_date", "product_category"]).copy()
    df["month"] = _month_period(df["order_date"])
    g = (
        df.groupby(["product_category", "month"], observed=True)
        .agg(
            otd_public=("on_time_delivery", "mean"),
            n_lines=("on_time_delivery", "size"),
        )
        .reset_index()
    )
    g["otd_public"] = g["otd_public"].astype(float)
    g = g.groupby(["product_category", "month"], as_index=False).agg({"otd_public": "mean", "n_lines": "sum"})
    return g


def derive_supplier_id(dataco: pd.DataFrame) -> pd.DataFrame:
    df = dataco.copy()
    df["supplier_id"] = pd.util.hash_pandas_object(df[["market", "product_category"]].astype(str), index=False).astype(str)
    return df


def build_monthly_supplier_panel(
    dataco: pd.DataFrame,
    part_catalog: pd.DataFrame,
    n_months: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    dataco_w = derive_supplier_id(dataco)
    baseline = build_category_month_otd_baseline(dataco_w)

    suppliers = dataco_w[["supplier_id", "market", "product_category"]].drop_duplicates().reset_index(drop=True)
    m = int(max(1, round(AT_RISK_SUPPLIER_SHARE * len(suppliers))))
    at_risk = set(suppliers.sample(m, random_state=int(rng.integers(1_000_000)))["supplier_id"].tolist())
    suppliers["supplier_at_risk_flag"] = suppliers["supplier_id"].map(lambda s: int(s in at_risk))

    t0 = pd.Timestamp(dataco_w["order_date"].min()).to_period("M").to_timestamp()
    months = pd.date_range(t0, periods=n_months, freq="MS")

    pc = part_catalog[["part_id", "product_category"]].copy()
    pc["product_category"] = pc["product_category"].astype(str)

    j = pc.merge(suppliers, on="product_category", how="left")
    j["_rk"] = rng.random(len(j))
    j = j.sort_values(["part_id", "_rk"], kind="mergesort").drop_duplicates(subset=["part_id"], keep="first")
    j = j.drop(columns=["_rk"])

    miss = j["supplier_id"].isna()
    if miss.any():
        h = pd.util.hash_pandas_object(j.loc[miss, "part_id"], index=False).astype(np.int64).to_numpy() % len(suppliers)
        h = h.astype(int)
        fb = suppliers.iloc[h].reset_index(drop=True)
        j.loc[miss, "supplier_id"] = fb["supplier_id"].astype(str).values
        j.loc[miss, "supplier_at_risk_flag"] = fb["supplier_at_risk_flag"].astype(int).values

    j = j[["part_id", "product_category", "supplier_id", "supplier_at_risk_flag"]]

    months_df = pd.DataFrame({"month": months})
    panel = j.merge(months_df, how="cross")

    base = baseline.rename(columns={"otd_public": "otd_public_raw"})
    panel = panel.merge(base, on=["product_category", "month"], how="left")
    month_avg = baseline.groupby("month", observed=True)["otd_public"].mean()
    panel["otd_public_dataco"] = panel["otd_public_raw"].astype(float)
    panel["otd_public_dataco"] = panel["otd_public_dataco"].fillna(panel["month"].map(month_avg))
    panel = panel.drop(columns=["otd_public_raw"], errors="ignore")

    n = len(panel)
    risk = panel["supplier_at_risk_flag"].astype(bool).to_numpy()
    stable_otd = np.clip(rng.normal(0.985, 0.008, size=n), 0.97, 0.999)
    at_risk_otd = np.clip(rng.normal(0.955, 0.012, size=n), 0.70, 0.97)
    otd_supplier = np.where(risk, at_risk_otd, stable_otd)
    burden_pp = rng.uniform(RESCHEDULE_BURDEN_PP_MIN, RESCHEDULE_BURDEN_PP_MAX, size=n)
    otd_oem = np.clip(otd_supplier - burden_pp / 100.0, 0.65, 0.999)

    panel["otd_supplier_reported"] = otd_supplier
    panel["otd_oem_measured"] = otd_oem
    panel["reschedule_burden_pp"] = burden_pp

    assert panel["month"].nunique() == n_months
    assert int(panel.groupby("supplier_id")["supplier_at_risk_flag"].nunique().max()) == 1
    assert abs(float(panel["supplier_at_risk_flag"].mean()) - AT_RISK_SUPPLIER_SHARE) < 0.05

    return panel


supplier_panel = build_monthly_supplier_panel(dataco_raw, part_catalog, N_MONTHS, rng)
print(supplier_panel.head())
print(supplier_panel.groupby("supplier_at_risk_flag")["otd_oem_measured"].mean())
"""
)

md(
    r"""
## 6) Compliance outcome generator (calibrated logistic layer)

**Why synthetic.** Public exports do not include **program milestone compliance** aligned to automotive PPAP/APQP-style schedules for these SKUs. We therefore generate a **binary compliance outcome** using a **transparent logistic data-generating process** whose coefficients favor rarer failures for **A** parts while keeping the **overall** failure rate near the **lower end** of the 5–20% automotive-motivated band.

**Calibration approach.** We **bisection-calibrate** an intercept on a linear predictor so the realized Bernoulli mean matches `COMPLIANCE_FAILURE_RATE_TARGET`.
"""
)

code(
    r"""
# =============================================================================
# 6) Compliance outcome generator
# =============================================================================


def _cls_onehot(classes: pd.Series) -> pd.DataFrame:
    d = pd.get_dummies(classes.astype(str), prefix="crit")
    for col in ["crit_A", "crit_B", "crit_C"]:
        if col not in d.columns:
            d[col] = 0
    return d[["crit_A", "crit_B", "crit_C"]]


def generate_compliance_outcomes(
    part_catalog: pd.DataFrame,
    supplier_panel: pd.DataFrame,
    rng: np.random.Generator,
    target_rate: float,
) -> pd.DataFrame:
    df = supplier_panel.merge(
        part_catalog[
            [
                "part_id",
                "criticality_class",
                "lead_time_cv",
                "bom_criticality_propagation_score",
            ]
        ],
        on="part_id",
        how="left",
    )

    df["criticality_class"] = df["criticality_class"].fillna("C")
    lt_cv = df["lead_time_cv"].astype(float)
    lt_cv = lt_cv.fillna(float(lt_cv.median()))
    cascade = df["bom_criticality_propagation_score"].astype(float)
    cascade = cascade.fillna(float(cascade.median()))
    cascade = (cascade - cascade.mean()) / (float(cascade.std(ddof=1)) + 1e-6)

    z_crit = _cls_onehot(df["criticality_class"])
    z = pd.concat(
        [
            z_crit,
            pd.Series(df["supplier_at_risk_flag"].astype(float), name="at_risk"),
            lt_cv.rename("lt_cv"),
            cascade.rename("cascade"),
        ],
        axis=1,
    )

    beta = np.array([-0.55, 0.10, 0.35, 0.65, 0.90, 0.25])
    xb = z.to_numpy(dtype=float) @ beta
    if not np.isfinite(xb).all():
        raise ValueError("Non-finite linear predictor in compliance model; check merged features.")

    def mean_prob(b0: float) -> float:
        t = xb + b0
        p = np.where(t >= 0.0, 1.0 / (1.0 + np.exp(-t)), np.exp(t) / (1.0 + np.exp(t)))
        return float(np.mean(p))

    lo, hi = -30.0, 30.0
    if mean_prob(lo) > target_rate or mean_prob(hi) < target_rate:
        raise ValueError(
            "Compliance calibration bracket failed (cannot hit target_rate with b0 in [-30, 30]). "
            "Inspect feature scales / beta."
        )

    mid = 0.0
    for _ in range(56):
        mid = (lo + hi) / 2.0
        if mean_prob(mid) > target_rate:
            hi = mid
        else:
            lo = mid

    b0 = float(mid)
    t = xb + b0
    p = np.where(t >= 0.0, 1.0 / (1.0 + np.exp(-t)), np.exp(t) / (1.0 + np.exp(t)))
    y = rng.binomial(1, p)

    out = df.assign(compliance_failure=y.astype(int), compliance_failure_prob=p, compliance_model_intercept=b0)

    mean_p = float(np.mean(p))
    assert mean_p <= COMPLIANCE_FAILURE_RATE_MAX + 1e-6, (mean_p, b0, float(np.max(xb)), float(np.min(xb)))
    assert abs(mean_p - target_rate) < 0.01

    sample_mean = float(out["compliance_failure"].mean())
    se = float(np.sqrt(mean_p * (1.0 - mean_p) / max(1, len(out))))
    assert sample_mean <= COMPLIANCE_FAILURE_RATE_MAX + 4.0 * se + 0.005

    prob_by = out.groupby("criticality_class", observed=True)["compliance_failure_prob"].mean()
    assert float(prob_by.get("A", 1.0)) < float(prob_by.get("C", 0.0))

    return out


compliance = generate_compliance_outcomes(part_catalog, supplier_panel, rng, COMPLIANCE_FAILURE_RATE_TARGET)
print(compliance.groupby("criticality_class")["compliance_failure"].mean())
"""
)

md(
    r"""
## 7) Feature engineering (rolling supplier stats + interactions)

**Inputs.** Merges **part static** features, **BOM topology** features, and the **monthly supplier panel**; computes **3- and 6-month rolling** means of OEM-measured OTD; adds **criticality × at-risk** interactions for modeling experiments.
"""
)

code(
    r"""
# =============================================================================
# 7) Feature engineering
# =============================================================================


def add_rolling_supplier_features(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.sort_values(["part_id", "month"]).copy()
    g = df.groupby("part_id", group_keys=False)
    df["otd_oem_roll3"] = g["otd_oem_measured"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    df["otd_oem_roll6"] = g["otd_oem_measured"].transform(lambda s: s.rolling(6, min_periods=1).mean())
    df["reschedule_burden_roll3"] = g["reschedule_burden_pp"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    return df


def engineer_final_matrix(part_catalog: pd.DataFrame, panel_roll: pd.DataFrame) -> pd.DataFrame:
    df = panel_roll.merge(part_catalog, on="part_id", how="left", suffixes=("", "_part"))

    for c in ["A", "B", "C"]:
        df[f"is_{c}"] = (df["criticality_class"] == c).astype(int)
        df[f"is_{c}_x_at_risk"] = df[f"is_{c}"] * df["supplier_at_risk_flag"].astype(int)

    return df


supplier_panel_fe = add_rolling_supplier_features(supplier_panel)
feature_matrix = engineer_final_matrix(part_catalog, supplier_panel_fe)

assert feature_matrix["otd_oem_roll3"].notna().all()
print(feature_matrix.filter(like="roll").head())
"""
)

md(
    r"""
## 8) Exploratory data analysis (validation plots)

These plots support **face validity** checks: ABC separation on engineered proxies, **failure rates**, **graph degree** behavior, **OTD shifts** under the at-risk flag, **feature correlations**, and a **benchmark comparison** panel for augmented parameters.
"""
)

code(
    r"""
# =============================================================================
# 8) Exploratory data analysis
# =============================================================================


def plot_feature_distributions_by_class(df: pd.DataFrame, cols: list[str], title: str) -> None:
    melt = df.melt(id_vars=["criticality_class"], value_vars=cols, var_name="feature", value_name="value")
    plt.figure(figsize=(14, 7))
    sns.violinplot(data=melt, x="feature", y="value", hue="criticality_class", cut=0)
    plt.title(title)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.show()


plot_feature_distributions_by_class(
    part_catalog,
    ["abc_price_proxy", "abc_demand_cv_proxy", "lead_time_mean_weeks", "lead_time_cv", "n_qualified_suppliers"],
    "Part-level feature distributions by criticality class",
)

plt.figure(figsize=(8, 5))
rate = compliance.groupby("criticality_class")["compliance_failure"].mean().reindex(["A", "B", "C"])
sns.barplot(x=rate.index.astype(str), y=rate.values, color="#4C72B0")
plt.ylabel("Compliance failure rate")
plt.title("Compliance failure rate by criticality class")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(part_catalog["bom_in_degree"], bins=40, color="#55A868")
plt.title("BOM in-degree distribution (count of direct consuming assemblies)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 5))
sns.kdeplot(data=supplier_panel, x="otd_oem_measured", hue="supplier_at_risk_flag", common_norm=False)
plt.title("OEM-measured OTD distribution by supplier at-risk flag")
plt.tight_layout()
plt.show()

num_cols = [c for c in part_catalog.select_dtypes(include=[np.number]).columns if part_catalog[c].notna().sum() > 10]
plt.figure(figsize=(12, 10))
corr = part_catalog[num_cols].corr(numeric_only=True)
sns.heatmap(corr, cmap="vlag", center=0)
plt.title("Correlation heatmap (part-catalog numeric features)")
plt.tight_layout()
plt.show()

bench = pd.DataFrame(
    [
        ("Lead time CV (pooled mean)", float(part_catalog["lead_time_cv"].mean()), 0.15, 0.35),
        ("Tier-1 OTD target anchor", float(OTD_TARGET_HIGH), 0.98, 0.99),
        ("Escalation threshold anchor", float(OTD_ESCALATION_THRESHOLD), 0.96, 0.98),
        ("Reschedule burden (pp mean)", float(supplier_panel["reschedule_burden_pp"].mean()), RESCHEDULE_BURDEN_PP_MIN, RESCHEDULE_BURDEN_PP_MAX),
    ],
    columns=["metric", "simulated", "bench_low", "bench_high"],
)

fig, ax = plt.subplots(figsize=(10, 4.5))
y = np.arange(len(bench))
for i, (_, row) in enumerate(bench.iterrows()):
    lo, hi, sim = float(row["bench_low"]), float(row["bench_high"]), float(row["simulated"])
    ax.barh(i, hi - lo, left=lo, height=0.38, color="#E8E8E8", edgecolor="#888888", linewidth=0.8, zorder=1)
    ax.scatter([sim], [i], color="#C44E52", s=60, zorder=3, marker="D", label="simulated" if i == 0 else "")
ax.set_yticks(y)
ax.set_yticklabels(bench["metric"].tolist())
ax.set_xlabel("Value (gray bar = benchmark range; red diamond = simulated)")
ax.set_title("Augmented parameters vs benchmark ranges (illustrative)")
ax.legend(loc="lower right")
plt.tight_layout()
plt.show()

print(bench)
"""
)

md(
    r"""
## 9) Export artifacts + data dictionary

Writes CSVs, serialized BOM graph, and a column-by-column **data dictionary** separating **public** vs **synthetic** fields with citations.
"""
)

code(
    r"""
# =============================================================================
# 9) Export
# =============================================================================


def write_data_dictionary(path: Path) -> None:
    lines: list[str] = []
    lines.append("# Hybrid dataset dictionary\n")
    lines.append("This file documents each exported column and whether it originates from **public datasets** or **literature-informed synthetic augmentation**.\n")

    lines.append("\n## Citations (non-exhaustive)\n")
    lines.append("- DataCo SMART Supply Chain: Constante et al. (2019), Mendeley Data, doi:10.17632/8gx2fvg2k6.5; Kaggle: `shashwatwork/dataco-smart-supply-chain-for-big-data-analysis`.\n")
    lines.append("- UCI Online Retail: Chen et al. (2012), UCI ML Repository, https://doi.org/10.24432/C5BW3K.\n")
    lines.append("- Lead time uncertainty / CV bands: stochastic lead-time / supply uncertainty literature (Omega-style empirical studies; **synthetic** mapping in this notebook).\n")
    lines.append("- Automotive BOM depth / fan-out: CIRP Annals automotive assembly / BOM complexity literature (**synthetic graph**).\n")
    lines.append("- Supplier OTD targets / escalation: APQC performance management + industry quality reporting (**anchors**); symestic-style OEM–supplier measured gaps (**synthetic reschedule burden**).\n")
    lines.append("- Compliance rate band: operations management / quality performance benchmarking discourse (**calibrated synthetic outcome**).\n")

    def dump_df(name: str, df: pd.DataFrame) -> None:
        lines.append(f"\n## `{name}`\n")
        public_cols = {
            "display_name",
            "product_category",
            "uci_median_unit_price",
            "uci_mean_unit_price",
            "uci_demand_mean_monthly",
            "uci_demand_cv_monthly",
            "uci_n_invoices",
            "dataco_median_line_sales",
            "dataco_mean_order_qty",
            "dataco_on_time_rate",
            "public_backbone",
            "stock_code",
            "otd_public_dataco",
        }
        for c in df.columns:
            src = "public_or_dataco_derived" if c in public_cols else "synthetic_or_derived"
            lines.append(f"- **{c}**: {src}\n")

    dump_df("part_catalog.csv", part_catalog.drop(columns=[c for c in ["dedup_key"] if c in part_catalog.columns], errors="ignore"))
    dump_df("supplier_history.csv", supplier_panel_fe)
    dump_df("compliance_outcomes.csv", compliance)

    path.write_text("".join(lines), encoding="utf-8")


part_out = part_catalog.drop(columns=[c for c in ["dedup_key"] if c in part_catalog.columns], errors="ignore")
part_out.to_csv(OUT_DIR / "part_catalog.csv", index=False)
supplier_panel_fe.to_csv(OUT_DIR / "supplier_history.csv", index=False)
compliance.to_csv(OUT_DIR / "compliance_outcomes.csv", index=False)

with open(OUT_DIR / "bom_graph.gpickle", "wb") as f:
    pickle.dump(bom_graph, f, protocol=pickle.HIGHEST_PROTOCOL)

write_data_dictionary(OUT_DIR / "data_dictionary.md")

print("Wrote:", sorted(p.name for p in OUT_DIR.iterdir()))
"""
)

code(
    r"""
# =============================================================================
# Final summary table (public vs synthetic provenance)
# =============================================================================

g = part_catalog.groupby("criticality_class", observed=True)
summary = pd.DataFrame(
    {
        "n_parts": g.size(),
        "mean_price_proxy": g["abc_price_proxy"].mean(),
        "mean_demand_cv": g["abc_demand_cv_proxy"].mean(),
        "mean_lead_time_weeks": g["lead_time_mean_weeks"].mean(),
        "mean_lead_time_cv": g["lead_time_cv"].mean(),
        "mean_bom_in_degree": g["bom_in_degree"].mean(),
        "mean_dataco_otd": g["dataco_on_time_rate"].mean(),
    }
).reset_index()

provenance = {
    "n_parts": "derived",
    "mean_price_proxy": "public",
    "mean_demand_cv": "public",
    "mean_lead_time_weeks": "synthetic",
    "mean_lead_time_cv": "synthetic",
    "mean_bom_in_degree": "synthetic",
    "mean_dataco_otd": "public",
}

for col, src in provenance.items():
    if col in summary.columns:
        summary[f"{col}__source"] = src

print(summary.to_string(index=False))
"""
)

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = Path(__file__).resolve().parent / "synthetic_data_generator.ipynb"
out_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("Wrote", out_path)
