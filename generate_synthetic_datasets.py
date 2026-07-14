#!/usr/bin/env python3
"""Hybrid synthetic dataset generator (CLI). See repository README / paper methods.

Usage:
  cd /path/to/part-class-variability
  uv run python generate_synthetic_datasets.py
  uv run python generate_synthetic_datasets.py --plots
  uv run python generate_synthetic_datasets.py --demo   # no DataCo CSV; stub backbone only

  ABC label de-confounding: **ABC_SCORE_NOISE** scales Gaussian jitter on the ABC *score* before
  quantile cuts (`score = log(price) + 0.75*zcv + ABC_SCORE_NOISE * N(0,1)`). Increase it to thicken
  A/B and B/C boundaries (default 0.05 reproduces the original nearly-deterministic score). Env
  `ABC_SCORE_NOISE` or `--abc-score-noise`.

  Optional **ABC_LABEL_NOISE** (post-cut uniform relabel) remains for audits; it does not fix the
  score path the way pre-cut jitter does.

  Class-conditional feature overlap: ABC_FEATURE_OVERLAP in [0, 1] linearly blends literature
  augmentation (suppliers, lead time, stockouts, substitutability) from class-specific priors
  toward pooled priors so A/B/C distributions overlap more (reduces label leakage through
  augment_literature_part_features). Env ABC_FEATURE_OVERLAP or --abc-feature-overlap.

  **Latent criticality mode** (`SYNTHETIC_GENERATOR_MODE=latent`): each part gets a continuous
  `latent_criticality_score`; ABC labels and observables are **independent noisy views** of that
  latent (no price→label→feature chain). Knobs: `LATENT_TO_LABEL_NOISE`, `LATENT_TO_FEATURE_NOISE`.
  BOM tiers use latent tertiles; cascade BOM summaries use latent tertiles, not ABC labels.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore", category=UserWarning)



# =============================================================================
# 1) Setup and imports
# =============================================================================
import hashlib
import io
import json
import re
from datetime import datetime, timezone
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import requests

# Matplotlib / seaborn imported lazily in main() and run_eda() so `--help` stays fast.

warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------------------------------------------------------
# Global constants (edit for sensitivity analysis)
# -----------------------------------------------------------------------------
N_PARTS = int(os.environ.get("N_PARTS", "3500"))
RANDOM_SEED = 42

# ABC nominal shares (rule-based proxy classifier targets)
A_SHARE, B_SHARE, C_SHARE = 0.20, 0.30, 0.50

# Post-quantile random ABC relabel (uniform A/B/C) — breaks invertible price/CV → class mapping.
ABC_LABEL_NOISE: float = float(os.environ.get("ABC_LABEL_NOISE", "0.0"))

# Blend class-conditional literature augmentation toward pooled distributions (see module docstring).
ABC_FEATURE_OVERLAP: float = float(os.environ.get("ABC_FEATURE_OVERLAP", "0.0"))

# Scale of N(0,1) jitter on ABC score *before* quantile cuts (thickens class boundaries).
ABC_SCORE_NOISE: float = float(os.environ.get("ABC_SCORE_NOISE", "0.05"))

# --- Latent criticality generator (SYNTHETIC_GENERATOR_MODE=latent) ---
SYNTHETIC_GENERATOR_MODE: str = os.environ.get("SYNTHETIC_GENERATOR_MODE", "latent").strip().lower()
# Std of Gaussian noise added to latent before ABC quantile assignment (higher = blurrier A/B/C).
LATENT_TO_LABEL_NOISE: float = float(os.environ.get("LATENT_TO_LABEL_NOISE", "0.45"))
# Scales independent observation noise on each feature channel vs latent (higher = weaker signal).
LATENT_TO_FEATURE_NOISE: float = float(os.environ.get("LATENT_TO_FEATURE_NOISE", "1.05"))

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

# Supplier-proxy at-risk share: bottom quantile by real DataCo OTD is flagged at-risk
# (see build_monthly_supplier_panel — data-derived threshold, not a random draw).
AT_RISK_SUPPLIER_SHARE = 0.15
OTD_TARGET_HIGH = 0.985
OTD_ESCALATION_THRESHOLD = 0.97

# Literature-anchored band for the OEM-vs-reported OTD gap ("reschedule burden"), percentage
# points. build_supplier_otd_from_dataco computes this gap from two independently real DataCo
# fields; these bounds are used only as a post-hoc plausibility check (run_eda benchmark chart),
# not to generate the value.
RESCHEDULE_BURDEN_PP_MIN, RESCHEDULE_BURDEN_PP_MAX = 4.0, 12.0

# Monthly panel horizon
N_MONTHS = 24

# Compliance label generator — calibrated logistic layer (synthetic outcome model)
COMPLIANCE_FAILURE_RATE_TARGET = 0.08
COMPLIANCE_FAILURE_RATE_MAX = 0.20

# Paths (OUT_DIR finalized in main())
OUT_DIR = ROOT / "outputs"

DATACO_CSV_PATH = Path(os.environ.get("DATACO_CSV_PATH", str(ROOT / "data/DataCoSupplyChainDataset.csv")))
UCI_ONLINE_RETAIL_URLS = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx",
    "https://github.com/gagolews/teaching-data/raw/master/marek/OnlineRetail.xlsx",
)

# Optional explicit URL only if user opts in (mirrors are unreliable for research reproducibility).
DATACO_FALLBACK_URL = os.environ.get("DATACO_FALLBACK_URL", "").strip()

np.random.seed(RANDOM_SEED)
rng = np.random.default_rng(RANDOM_SEED)



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
    """Load DataCo from a local CSV. Network mirrors are opt-in only (see DATACO_FALLBACK_URL)."""
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
                "(doi:10.17632/8gx2fvg2k6.5), place it at:\n"
                f"  {ROOT / 'data' / 'DataCoSupplyChainDataset.csv'}\n"
                "or set environment variable DATACO_CSV_PATH to the file location.\n"
                "Optional: set DATACO_FALLBACK_URL to a direct HTTPS URL if you must fetch remotely "
                "(not recommended for reproducibility)."
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


def make_demo_dataco_stub(rng: np.random.Generator, n_rows: int = 18_000) -> pd.DataFrame:
    """Synthetic order lines matching post-`load_dataco_supply_chain` schema (no real DataCo file).

    Use only for smoke tests / CI when `data/DataCoSupplyChainDataset.csv` is unavailable.
    """
    categories = [f"DEMO_CAT_{i:02d}" for i in range(16)]
    markets = ["NA", "EMEA", "APAC", "LATAM"]
    t0 = pd.Timestamp("2018-06-01")
    order_dates = t0 + pd.to_timedelta(rng.integers(0, 800, size=n_rows), unit="D")
    sched = rng.integers(4, 11, size=n_rows)
    jitter = rng.integers(-1, 2, size=n_rows)
    real_days = np.clip(sched.astype(float) + jitter.astype(float), 1.0, None)
    ship_dates = order_dates + pd.to_timedelta(real_days.astype(int), unit="D")

    df = pd.DataFrame(
        {
            "order_date": order_dates,
            "ship_date": ship_dates,
            "days_shipping_real": real_days,
            "days_shipment_scheduled": sched.astype(float),
            "delivery_status": np.where(real_days <= sched, "Shipping On Time", "Shipping Late"),
            "product_category": rng.choice(categories, size=n_rows),
            "order_quantity": rng.integers(1, 80, size=n_rows),
            "order_item_discount_rate": rng.uniform(0.0, 0.25, size=n_rows).round(4),
            "sales": rng.lognormal(3.0, 0.9, size=n_rows).round(2),
            "product_name": np.array([f"DEMO_SKU_{i % 320:04d}" for i in range(n_rows)]),
            "market": rng.choice(markets, size=n_rows),
        }
    )
    df["days_late_ship"] = df["days_shipping_real"] - df["days_shipment_scheduled"]
    df["on_time_delivery"] = (df["days_late_ship"] <= 0).astype("Int64")
    df["lead_time_days_calendar"] = (df["ship_date"] - df["order_date"]).dt.days
    df["on_time_delivery_calendar"] = (
        df["lead_time_days_calendar"] <= pd.to_numeric(df["days_shipment_scheduled"], errors="coerce")
    ).astype("Int64")
    df["dataset"] = "dataco_demo_stub"
    df.attrs["load_source"] = "synthetic:demo_stub_no_real_dataco_csv"
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


# =============================================================================
# 3) Part catalog construction
# =============================================================================


def _base_part_id(namespace: str, dedup_key: str) -> str:
    """Content-derived id: stable across partial reruns for the same (namespace, dedup_key)."""
    digest = hashlib.sha256(f"{namespace}|{dedup_key}".encode("utf-8")).hexdigest()[:24]
    return f"{namespace[:3].upper()}_{digest}"


def _assign_part_ids_with_collision_suffix(namespace_keys: pd.Series, dedup_keys: pd.Series) -> list[str]:
    """Assign part_id from content hash; append _dup{n} only on true collisions."""
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
    label_noise: float | None = None,
    score_noise: float | None = None,
) -> pd.DataFrame:
    ln = float(ABC_LABEL_NOISE if label_noise is None else label_noise)
    if not (0.0 <= ln <= 1.0):
        raise ValueError(f"label_noise must be in [0, 1], got {ln!r}")
    sn = float(ABC_SCORE_NOISE if score_noise is None else score_noise)
    if sn < 0.0:
        raise ValueError(f"score_noise must be >= 0, got {sn!r}")

    x = np.log1p(df[price_col].clip(lower=1e-6))
    zcv = df[cv_col].replace([np.inf, -np.inf], np.nan)
    zcv = zcv.fillna(zcv.median())
    zcv = (zcv - zcv.mean()) / (zcv.std(ddof=1) + 1e-6)
    score = x + 0.75 * zcv + sn * rng.normal(size=len(df))
    q1, q2 = np.quantile(score, [a_share, a_share + b_share])
    cls = np.asarray(np.where(score <= q1, "A", np.where(score <= q2, "B", "C")), dtype=object)
    if ln > 0.0:
        flip_mask = rng.random(size=len(cls)) < ln
        if np.any(flip_mask):
            cls = cls.copy()
            cls[flip_mask] = rng.choice(np.array(["A", "B", "C"], dtype=object), size=int(flip_mask.sum()))

    out = df.copy()
    out["criticality_class"] = cls.astype(str)
    return out


def _blended_literature_tables(alpha: float) -> tuple[
    dict[str, tuple[int, int]],
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, float],
]:
    """Linear blend (alpha in [0,1]) from class-specific constants toward pooled values."""
    classes = ("A", "B", "C")
    lt_mean_p = sum(LT_MEAN_WEEKS[c] for c in classes) / 3.0
    lt_cv_p = sum(LT_CV[c] for c in classes) / 3.0
    psub_p = sum(P_SUBSTITUTABLE[c] for c in classes) / 3.0
    stock_p = sum(STOCKOUT_RATE_BY_CLASS[c] for c in classes) / 3.0
    ns_lo_p, ns_hi_p = 1, 6

    n_sup: dict[str, tuple[int, int]] = {}
    p_sub: dict[str, float] = {}
    stock_r: dict[str, float] = {}
    lt_m: dict[str, float] = {}
    lt_c: dict[str, float] = {}
    for c in classes:
        lo0, hi0 = N_SUPPLIERS_BY_CLASS[c]
        lo1 = int(max(1, round((1.0 - alpha) * lo0 + alpha * ns_lo_p)))
        hi1 = int(max(lo1, round((1.0 - alpha) * hi0 + alpha * ns_hi_p)))
        n_sup[c] = (lo1, hi1)
        p_sub[c] = (1.0 - alpha) * P_SUBSTITUTABLE[c] + alpha * psub_p
        stock_r[c] = max(0.05, (1.0 - alpha) * STOCKOUT_RATE_BY_CLASS[c] + alpha * stock_p)
        lt_m[c] = (1.0 - alpha) * LT_MEAN_WEEKS[c] + alpha * lt_mean_p
        lt_c[c] = (1.0 - alpha) * LT_CV[c] + alpha * lt_cv_p
    return n_sup, p_sub, stock_r, lt_m, lt_c


def _lt_mean_clip_bounds(class_key: str, alpha: float) -> tuple[float, float]:
    """Clip range for sampled lead-time mean (weeks); blend toward pooled (4, 14)."""
    if class_key == "A":
        lo_s, hi_s = 4.0, 6.0
    else:
        lo_s, hi_s = 6.0, 14.0
    lo_p, hi_p = 4.0, 14.0
    return (1.0 - alpha) * lo_s + alpha * lo_p, (1.0 - alpha) * hi_s + alpha * hi_p


def augment_literature_part_features(
    df: pd.DataFrame,
    rng: np.random.Generator,
    feature_overlap: float | None = None,
) -> pd.DataFrame:
    alpha = float(ABC_FEATURE_OVERLAP if feature_overlap is None else feature_overlap)
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"feature_overlap must be in [0, 1], got {alpha!r}")

    n_sup, p_sub, stock_r, lt_m, lt_c = _blended_literature_tables(alpha)
    df = df.copy()

    def draw_supplier_count(row) -> int:
        lo, hi = n_sup[str(row["criticality_class"])]
        return int(rng.integers(lo, hi + 1))

    df["n_qualified_suppliers"] = df.apply(draw_supplier_count, axis=1)
    df["substitutable_flag"] = df["criticality_class"].map(
        lambda k: int(rng.random() < p_sub[str(k)])
    )
    df["stockout_events_per_year"] = df["criticality_class"].map(
        lambda k: float(max(0.0, rng.poisson(stock_r[str(k)])))
    )

    def draw_lt(row):
        k = str(row["criticality_class"])
        mean_w = float(rng.normal(lt_m[k], 0.35))
        lo_clip, hi_clip = _lt_mean_clip_bounds(k, alpha)
        mean_w = float(np.clip(mean_w, lo_clip, hi_clip))
        cv = float(rng.normal(lt_c[k], 0.03))
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


def sample_latent_criticality(n: int, rng: np.random.Generator) -> np.ndarray:
    """Standard normal latent scores (one per part row)."""
    return rng.normal(0.0, 1.0, size=int(n))


def assign_abc_labels_from_latent(
    latent: np.ndarray,
    label_noise_std: float,
    rng: np.random.Generator,
    a_share: float,
    b_share: float,
    c_share: float,
) -> np.ndarray:
    """Noisy quantile cut on latent: ABC is a coarsened, noisy function of the same latent."""
    noisy = latent + float(label_noise_std) * rng.normal(size=len(latent))
    q1, q2 = np.quantile(noisy, [a_share, a_share + b_share])
    cls = np.asarray(np.where(noisy <= q1, "A", np.where(noisy <= q2, "B", "C")), dtype=str)
    return cls


def apply_latent_observed_features(
    df: pd.DataFrame,
    latent: np.ndarray,
    feature_noise_scale: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Independent noisy observables of latent (same column names as legacy augmentation)."""
    out = df.copy()
    fe = float(feature_noise_scale)
    n = len(out)
    lat = np.asarray(latent, dtype=float)
    out["latent_criticality_score"] = lat.astype(float)
    out["abc_price_proxy"] = np.clip(np.exp(0.6 * lat + fe * rng.normal(size=n)), 1e-6, None).astype(float)
    out["abc_demand_cv_proxy"] = np.clip(
        0.35 + 0.45 * lat + fe * 0.35 * rng.normal(size=n),
        0.12,
        3.0,
    ).astype(float)
    lt_mean = np.clip(np.exp(-0.35 * lat + fe * rng.normal(size=n)), 4.0, 14.0).astype(float)
    out["lead_time_mean_weeks"] = lt_mean
    out["lead_time_cv"] = np.clip(
        0.22 + 0.05 * np.tanh(lat) + fe * 0.035 * rng.normal(size=n),
        0.15,
        0.35,
    ).astype(float)
    out["lead_time_sigma_weeks"] = np.clip(out["lead_time_cv"] * out["lead_time_mean_weeks"], 1e-3, None).astype(
        float
    )
    lam = np.clip(3.4 - 0.75 * lat + fe * 0.5 * rng.normal(size=n), 0.6, 10.0)
    ns = rng.poisson(lam).astype(int)
    out["n_qualified_suppliers"] = np.clip(ns, 1, 6).astype(int)
    logit = np.clip(-0.5 + 0.55 * lat + fe * 0.4 * rng.normal(size=n), -6.0, 6.0)
    p_sub = 1.0 / (1.0 + np.exp(-logit))
    out["substitutable_flag"] = rng.binomial(1, p_sub).astype(int)
    span = float(np.ptp(lat)) + 1e-9
    stock_mu = np.clip(
        0.4 + 0.45 * (lat - lat.min()) / span + fe * 0.25 * rng.normal(size=n),
        0.08,
        5.0,
    )
    out["stockout_events_per_year"] = rng.poisson(np.maximum(stock_mu, 1e-3)).astype(float)

    out["lead_time_source"] = "synthetic_latent_generator"
    out["supplier_count_source"] = "synthetic_latent_generator"
    out["substitutability_source"] = "synthetic_latent_generator"
    out["stockout_source"] = "synthetic_latent_generator"
    return out


def build_unified_part_catalog(
    dataco: pd.DataFrame,
    uci: pd.DataFrame,
    n_target: int,
    rng: np.random.Generator,
    abc_label_noise: float | None = None,
    abc_feature_overlap: float | None = None,
    abc_score_noise: float | None = None,
    generator_mode: str | None = None,
) -> pd.DataFrame:
    mode = (generator_mode or SYNTHETIC_GENERATOR_MODE).strip().lower()
    if mode not in ("legacy", "latent"):
        raise ValueError(f"Unknown generator mode {mode!r}; use 'legacy' or 'latent' (env SYNTHETIC_GENERATOR_MODE).")

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

    eff_noise = float(ABC_LABEL_NOISE if abc_label_noise is None else abc_label_noise)
    if mode == "legacy":
        price = both["uci_median_unit_price"].fillna(
            both["dataco_median_line_sales"].fillna(0.0) / (both["dataco_mean_order_qty"].fillna(1.0) + 1.0)
        )
        cv = both["uci_demand_cv_monthly"].fillna(0.35)
        both["abc_price_proxy"] = price.clip(lower=1e-6)
        both["abc_demand_cv_proxy"] = cv
        both = assign_abc_by_price_demand_cv(
            both,
            "abc_price_proxy",
            "abc_demand_cv_proxy",
            rng,
            A_SHARE,
            B_SHARE,
            C_SHARE,
            label_noise=abc_label_noise,
            score_noise=abc_score_noise,
        )

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

    # Assign content-stable part_id only after all sampling / synthetic padding.
    both = both.reset_index(drop=True)
    both["part_id"] = _assign_part_ids_with_collision_suffix(both["namespace"], both["dedup_key"])

    # True iff product_category is a genuine real DataCo category (not the UCI_ONLINE_RETAIL
    # placeholder or a category that only exists because a synthetic-padding row cloned one).
    # UCI Online Retail (household/gift SKUs) and DataCo (sporting goods/apparel/electronics
    # retailer) do not share a category vocabulary — a TF-IDF similarity check between UCI
    # product descriptions and DataCo's real category product-name corpus found >50% of UCI
    # products have zero real overlap with any DataCo category, and even top-scoring matches were
    # frequently keyword coincidences, not genuine category matches. Supplier-panel /
    # compliance-outcome analyses that need a *real* category link (not the random within-category
    # fallback used for unmatched parts — see build_monthly_supplier_panel) should filter on this
    # column; see modeling_lib.get_layer2_scope / filter_layer2_scope.
    real_dataco_categories = set(dataco["product_category"].dropna().astype(str).unique())
    both["real_category_link"] = both["product_category"].astype(str).isin(real_dataco_categories)

    if mode == "legacy":
        both = augment_literature_part_features(both, rng, feature_overlap=abc_feature_overlap)
    else:
        latent_v = sample_latent_criticality(len(both), rng)
        cls = assign_abc_labels_from_latent(
            latent_v,
            LATENT_TO_LABEL_NOISE,
            rng,
            A_SHARE,
            B_SHARE,
            C_SHARE,
        )
        both["criticality_class"] = cls
        both = apply_latent_observed_features(both, latent_v, LATENT_TO_FEATURE_NOISE, rng)

    if not both["part_id"].is_unique:
        raise AssertionError("Internal error: part_id must be unique after catalog build.")

    shares = both["criticality_class"].value_counts(normalize=True)
    if mode == "latent":
        assert set(both["criticality_class"].astype(str).unique()) == {"A", "B", "C"}
        assert (both["criticality_class"].value_counts() >= 1).all()
        for lab, target, tol in (("A", A_SHARE, 0.12), ("B", B_SHARE, 0.12), ("C", C_SHARE, 0.15)):
            assert abs(float(shares.get(lab, 0.0)) - target) < tol, (shares.to_dict(), lab)
    elif eff_noise < 1e-12:
        assert abs(float(shares.get("A", 0.0)) - A_SHARE) < 0.03
        assert abs(float(shares.get("B", 0.0)) - B_SHARE) < 0.04
        assert abs(float(shares.get("C", 0.0)) - C_SHARE) < 0.06
    else:
        assert set(both["criticality_class"].astype(str).unique()) == {"A", "B", "C"}
        assert (both["criticality_class"].value_counts() >= 1).all()

    assert both["lead_time_cv"].between(0.15, 0.35).mean() > 0.90
    assert both["n_qualified_suppliers"].between(1, 6).all()

    return both




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
    """Supplier-proxy entity from real DataCo operational fields.

    DataCo has no literal supplier identifier. (market, product_category, shipping_mode) is the
    finest real operational grouping available (639 distinct combinations, median 47 real order
    lines each over the full 2015-01..2018-01 span) and is used as a documented proxy for
    "supplier". Only the *identity* is a proxy; every OTD / at-risk / reschedule metric attached
    to it downstream (see build_supplier_otd_from_dataco) is computed from real order lines.
    """
    df = dataco.copy()
    df["supplier_id"] = pd.util.hash_pandas_object(
        df[["market", "product_category", "shipping_mode"]].astype(str), index=False
    ).astype(str)
    return df


def build_supplier_otd_from_dataco(
    dataco_w: pd.DataFrame,
    shrink_k: float = 8.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Real, DataCo-derived monthly on-time-delivery rate for the supplier-proxy entity (see
    ``derive_supplier_id``): ``otd_supplier_reported`` = mean(``on_time_delivery``), DataCo's own
    real ``days_shipping_real`` vs ``days_shipment_scheduled`` columns.

    Monthly cells with few real order lines are shrunk toward the supplier-proxy's full-period
    real average via a precision-weighted blend (``n / (n + shrink_k)``) — this is smoothing over
    real observations, not injected noise, so single-digit-line months don't dominate.

    Note on a design choice we reverted: we initially tried treating DataCo's ``late_delivery_risk``
    field (and a calendar-date recomputation of OTD) as an independent second real "OEM-measured"
    lens, to derive ``reschedule_burden_pp`` as a genuine real gap. Both agree with
    ``on_time_delivery`` on >97% of rows — ``late_delivery_risk`` is essentially a restatement of
    DataCo's own ``Delivery Status`` column, and the calendar recomputation differs only by
    date-truncation rounding. DataCo does not contain two genuinely independent OTD measurement
    systems, so that gap would have been fake precision, not a real signal. See
    ``build_monthly_supplier_panel`` for how the reschedule-burden gap is instead handled
    (literature-anchored synthetic overlay, gated by the real at-risk flag).

    Returns (monthly_panel, overall_by_supplier).
    """
    df = dataco_w.dropna(subset=["order_date", "market", "product_category", "shipping_mode"]).copy()
    df["month"] = _month_period(df["order_date"])

    overall = (
        df.groupby("supplier_id", observed=True)
        .agg(
            otd_supplier_reported_all=("on_time_delivery", "mean"),
            n_lines_all=("on_time_delivery", "size"),
        )
        .reset_index()
    )

    cell = (
        df.groupby(["supplier_id", "month"], observed=True)
        .agg(
            otd_supplier_reported_cell=("on_time_delivery", "mean"),
            n_lines=("on_time_delivery", "size"),
        )
        .reset_index()
    )
    cell = cell.merge(overall[["supplier_id", "otd_supplier_reported_all"]], on="supplier_id", how="left")

    w = cell["n_lines"] / (cell["n_lines"] + shrink_k)
    cell["otd_supplier_reported"] = w * cell["otd_supplier_reported_cell"] + (1.0 - w) * cell["otd_supplier_reported_all"]

    monthly = cell[["supplier_id", "month", "otd_supplier_reported"]]
    return monthly, overall


def build_monthly_supplier_panel(
    dataco: pd.DataFrame,
    part_catalog: pd.DataFrame,
    n_months: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    dataco_w = derive_supplier_id(dataco)
    category_baseline = build_category_month_otd_baseline(dataco_w)
    monthly_otd, overall = build_supplier_otd_from_dataco(dataco_w)

    suppliers = dataco_w[["supplier_id", "market", "product_category"]].drop_duplicates().reset_index(drop=True)
    suppliers = suppliers.merge(overall[["supplier_id", "otd_supplier_reported_all"]], on="supplier_id", how="left")

    # At-risk = bottom AT_RISK_SUPPLIER_SHARE of supplier-proxies by real full-period reported
    # OTD — a data-derived threshold on real DataCo performance, not a random draw. Rank-based
    # (not quantile+<=) because many thin supplier-proxy cells tie at 0.0/1.0 exactly, which would
    # otherwise blow past the target share; ties at the cutoff are broken with rng, not by row order.
    n_sup = len(suppliers)
    k_at_risk = int(round(AT_RISK_SUPPLIER_SHARE * n_sup))
    tiebreak = rng.random(n_sup)
    order = np.lexsort((tiebreak, suppliers["otd_supplier_reported_all"].to_numpy()))
    at_risk_idx = set(order[:k_at_risk].tolist())
    suppliers["supplier_at_risk_flag"] = [int(i in at_risk_idx) for i in range(n_sup)]

    t0 = pd.Timestamp(dataco_w["order_date"].min()).to_period("M").to_timestamp()
    months = pd.date_range(t0, periods=n_months, freq="MS")

    pc = part_catalog[["part_id", "product_category"]].copy()
    pc["product_category"] = pc["product_category"].astype(str)

    # Necessary synthetic linkage: parts (from UCI/DataCo product-level stats) and supplier-proxies
    # (from DataCo order-line operational groups) are not linked in the raw data, so one
    # matching-category supplier-proxy is assigned per part at random. Everything about the
    # assigned supplier-proxy's *behavior* below is real, not the assignment itself.
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

    # Coarser real reference (category x month), distinct from the finer supplier-proxy-grain
    # metrics below.
    base = category_baseline.rename(columns={"otd_public": "otd_public_raw"})
    panel = panel.merge(base, on=["product_category", "month"], how="left")
    month_avg = category_baseline.groupby("month", observed=True)["otd_public"].mean()
    panel["otd_public_dataco"] = panel["otd_public_raw"].astype(float)
    panel["otd_public_dataco"] = panel["otd_public_dataco"].fillna(panel["month"].map(month_avg))
    panel = panel.drop(columns=["otd_public_raw"], errors="ignore")

    # Supplier-proxy-grain real OTD (shrunk monthly rate; see build_supplier_otd_from_dataco).
    panel = panel.merge(monthly_otd, on=["supplier_id", "month"], how="left")
    # Fallback for supplier-proxy x month cells outside the real DataCo horizon actually observed
    # for that proxy: the supplier-proxy's own full-period real average (never injected noise).
    ov = overall.set_index("supplier_id")
    panel["otd_supplier_reported"] = panel["otd_supplier_reported"].fillna(
        panel["supplier_id"].map(ov["otd_supplier_reported_all"])
    )

    # reschedule_burden_pp / otd_oem_measured: DataCo has no second, genuinely independent OTD
    # measurement system (see build_supplier_otd_from_dataco docstring) to derive this "OEM audit
    # vs supplier-reported" gap from real data. Kept as a literature-anchored synthetic overlay
    # (RESCHEDULE_BURDEN_PP_MIN/MAX), but — unlike the original design — it is now *gated by the
    # real at-risk flag* rather than independent random noise: at-risk supplier-proxies (real,
    # bottom-quantile OTD) draw from the upper half of the literature band, others from the lower
    # half, so the synthetic gap is at least conditioned on real supplier-proxy behavior.
    n = len(panel)
    risk = panel["supplier_at_risk_flag"].astype(bool).to_numpy()
    band_mid = RESCHEDULE_BURDEN_PP_MIN + 0.5 * (RESCHEDULE_BURDEN_PP_MAX - RESCHEDULE_BURDEN_PP_MIN)
    low_band = rng.uniform(RESCHEDULE_BURDEN_PP_MIN, band_mid, size=n)
    high_band = rng.uniform(band_mid, RESCHEDULE_BURDEN_PP_MAX, size=n)
    panel["reschedule_burden_pp"] = np.where(risk, high_band, low_band)
    panel["otd_oem_measured"] = np.clip(
        panel["otd_supplier_reported"].astype(float) - panel["reschedule_burden_pp"] / 100.0, 0.0, 1.0
    )

    assert panel["month"].nunique() == n_months
    assert int(panel.groupby("supplier_id")["supplier_at_risk_flag"].nunique().max()) == 1
    assert abs(float(panel["supplier_at_risk_flag"].mean()) - AT_RISK_SUPPLIER_SHARE) < 0.05
    assert panel[["otd_supplier_reported", "otd_oem_measured", "reschedule_burden_pp"]].notna().all().all()
    assert panel["otd_oem_measured"].between(0.0, 1.0).mean() > 0.98
    assert (panel["reschedule_burden_pp"] >= 0.0).all()

    return panel




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

    otd = df["otd_oem_measured"].astype(float)
    otd = otd.fillna(float(otd.median()))
    otd = (otd - otd.mean()) / (float(otd.std(ddof=1)) + 1e-6)

    resch = df["reschedule_burden_pp"].astype(float)
    resch = resch.fillna(float(resch.median()))
    resch = (resch - resch.mean()) / (float(resch.std(ddof=1)) + 1e-6)

    z_crit = _cls_onehot(df["criticality_class"])
    z = pd.concat(
        [
            z_crit,
            pd.Series(df["supplier_at_risk_flag"].astype(float), name="at_risk"),
            lt_cv.rename("lt_cv"),
            cascade.rename("cascade"),
            otd.rename("otd"),
            resch.rename("resch"),
        ],
        axis=1,
    )

    # crit_A,B,C | at_risk | lt_cv | cascade | otd (lower OTD -> higher risk) | reschedule burden
    beta = np.array([-0.55, 0.10, 0.35, 0.65, 0.90, 0.25, -0.55, 0.45])
    xb = z.to_numpy(dtype=float) @ beta
    if not np.isfinite(xb).all():
        raise ValueError("Non-finite linear predictor in compliance model; check merged features.")

    def mean_prob(b0: float) -> float:
        t = xb + b0
        # numerically stable sigmoid
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
    """Merge part catalog onto the supplier panel (no true-class interaction columns)."""
    return panel_roll.merge(part_catalog, on="part_id", how="left", suffixes=("", "_part"))


# =============================================================================
# 9) Export
# =============================================================================


def write_data_dictionary(
    path: Path,
    part_catalog: pd.DataFrame,
    supplier_panel_fe: pd.DataFrame,
    compliance: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("# Hybrid dataset dictionary\n")
    lines.append("This file documents each exported column and whether it originates from **public datasets** or **literature-informed synthetic augmentation**.\n")

    lines.append("\n## Citations (non-exhaustive)\n")
    lines.append("- DataCo SMART Supply Chain: Constante et al. (2019), Mendeley Data, doi:10.17632/8gx2fvg2k6.5; Kaggle: `shashwatwork/dataco-smart-supply-chain-for-big-data-analysis`.\n")
    lines.append("- UCI Online Retail: Chen et al. (2012), UCI ML Repository, https://doi.org/10.24432/C5BW3K.\n")
    lines.append(
        "- ABC class-size convention (A_SHARE/B_SHARE/C_SHARE = 20/30/50): no public dataset gives "
        "real part-level criticality ground truth (proprietary business judgments; only "
        "methodology papers are public — investigated, see project history). The class-SIZE "
        "convention (not the label-assignment mechanism) is checked against real revenue "
        "concentration in the DataCo/UCI backbones already in hand via "
        "`scripts/check_abc_share_benchmark.py`: real UCI per-SKU revenue is top-20%-of-SKUs → "
        "79.5% of revenue (Gini 0.763); real DataCo per-product sales is even more concentrated "
        "(95.1%, Gini 0.883, n=118). Both confirm the real backbones follow a Pareto-style skew "
        "consistent with a 20/30/50 split being a reasonable convention. **Does not validate** "
        "`SYNTHETIC_GENERATOR_MODE=latent`'s label-assignment mechanism, which deliberately "
        "decouples labels from price/revenue (to avoid a trivial price→label→feature chain) — see "
        "`outputs/abc_share_benchmark.md`.\n"
    )
    lines.append("- Lead time uncertainty / CV bands: stochastic lead-time / supply uncertainty literature (Omega-style empirical studies; **synthetic** mapping in this notebook).\n")
    lines.append(
        "- BOM depth / fan-out: CIRP Annals automotive assembly / BOM complexity literature "
        "(**synthetic graph**). Also cross-checked against real disassembly data (Babbitt et al. "
        "2020, Scientific Data, doi:10.1038/s41597-020-0573-9; CC0 raw data, "
        "doi:10.6084/m9.figshare.11306792) via `scripts/check_bom_benchmark.py`: 108 real, "
        "lab-disassembled consumer electronics products decompose into a mean of 4.82 (median 4, "
        "range 1-13) major assemblies. **Not a depth/fan-out calibration** — verified directly "
        "against the raw workbook that it records mass composition per named assembly, not "
        "component counts or hierarchy depth, so it measures a different structural concept than "
        "`BOM_FANOUT_MIN/MAX` (components per assembly in a multi-tier supply chain, vs. named "
        "assemblies within one finished product) — offered as face-validity context (same order of "
        "magnitude, single-digit-to-low-teens) only. See `outputs/bom_benchmark_disassembly.md`.\n"
    )
    lines.append("- Supplier OTD targets / escalation: APQC performance management + industry quality reporting (**anchors**).\n")
    lines.append(
        "- Supplier-proxy identity and OTD / at-risk flag: computed directly from real DataCo order-line "
        "`on_time_delivery` (days_shipping_real vs days_shipment_scheduled), grouped by (market, category, "
        "shipping mode) as a documented supplier-proxy entity — see `build_supplier_otd_from_dataco` "
        "(**dataco_derived_proxy**: real behavior, proxy identity; DataCo has no literal supplier field, and "
        "the part<->supplier-proxy assignment is a random within-category match). We evaluated using "
        "DataCo's `late_delivery_risk` field as an independent second real OTD measure but found it >97% "
        "redundant with `on_time_delivery` (a near-restatement of `Delivery Status`), so it is **not** used "
        "as an independent signal.\n"
    )
    lines.append(
        "- Reschedule burden / OEM-measured OTD gap: DataCo contains no second, genuinely independent OTD "
        "measurement system, so this remains a **literature-anchored synthetic overlay** "
        "(RESCHEDULE_BURDEN_PP_MIN/MAX) — but is now gated by the real at-risk flag above (at-risk "
        "supplier-proxies draw from the upper half of the literature band) rather than independent random "
        "noise.\n"
    )
    lines.append(
        "- Compliance rate band: operations management / quality performance benchmarking discourse "
        "(**calibrated synthetic outcome**). We investigated CPSC/NHTSA recalls as a real substitute "
        "for `compliance_failure` and found this is not possible: recall data gives counts, not rates "
        "(no real population denominator links a specific recalled product to this dataset's parts or "
        "to DataCo's/UCI's sold volumes), so any conversion to a rate would itself be invented. Real "
        "CPSC data (saferproducts.gov, 2015-2025) is instead used as a **face-validity scale check** "
        "(not a row-level label) via `scripts/check_recall_benchmark.py`: ~294 recalls/year across the "
        "entire US consumer market vs. this dataset's 8%/part-month calibrated rate confirms "
        "`compliance_failure` is intentionally modeling a different, higher-frequency, lower-severity "
        "phenomenon (routine operational/quality/delivery compliance) than product-safety recalls, not "
        "attempting to replicate the recall rate itself. See `outputs/compliance_benchmark_cpsc.md`.\n"
    )
    lines.append(
        "- `real_category_link`: UCI Online Retail (household/gift SKUs) and DataCo (sporting "
        "goods/apparel/electronics retailer) do not share a category vocabulary. A TF-IDF cosine "
        "similarity check between UCI product descriptions and DataCo's real per-category product-name "
        "corpus found 50% of UCI products have zero real overlap with any DataCo category, and only "
        "5.1% clear a similarity >= 0.20 — with even top-scoring matches frequently reflecting keyword "
        "coincidence rather than genuine category match (e.g. \"WHEELBARROW FOR CHILDREN\" top-matched "
        "\"Children's Clothing\" at a perfect 1.000 score). We therefore did not build a similarity-based "
        "category mapper. `real_category_link` (**public_or_dataco_derived**) flags the ~100-120 parts "
        "per run whose `product_category` is a genuine real DataCo category; Layer 2 (supplier/compliance) "
        "analyses can be scoped to this subset via `LAYER2_SCOPE=real_category_only` "
        "(see `modeling_lib.get_layer2_scope`) for a fully-grounded, smaller-N result, distinct from the "
        "full-catalog run which includes the random within-category supplier-proxy fallback for "
        "unmatched (mostly UCI-sourced) parts.\n"
    )

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
            "real_category_link",
        }
        # Real DataCo order-line behavior attached to a documented supplier-proxy entity (market x
        # category x shipping mode) — see build_supplier_otd_from_dataco. Identity/part-linkage is a
        # proxy; otd_supplier_reported and the at-risk flag are computed from real DataCo columns, not
        # injected noise. reschedule_burden_pp / otd_oem_measured remain a literature-anchored synthetic
        # overlay gated by the real at-risk flag (see citations above) and are NOT in this set.
        dataco_derived_proxy_cols = {
            "supplier_id",
            "supplier_at_risk_flag",
            "otd_supplier_reported",
        }
        for c in df.columns:
            if c in public_cols:
                src = "public_or_dataco_derived"
            elif c in dataco_derived_proxy_cols:
                src = "dataco_derived_proxy"
            else:
                src = "synthetic_or_derived"
            lines.append(f"- **{c}**: {src}\n")

    dump_df("part_catalog.csv", part_catalog.drop(columns=[c for c in ["dedup_key"] if c in part_catalog.columns], errors="ignore"))
    dump_df("supplier_history.csv", supplier_panel_fe)
    dump_df("compliance_outcomes.csv", compliance)

    path.write_text("".join(lines), encoding="utf-8")

def run_eda(part_catalog, compliance, supplier_panel, fig_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")
    _n = 0

    def save():
        nonlocal _n
        plt.savefig(fig_dir / f"eda_{_n:02d}.png", dpi=150, bbox_inches="tight")
        plt.close()
        _n += 1

    melt = part_catalog.melt(
        id_vars=["criticality_class"],
        value_vars=["abc_price_proxy", "abc_demand_cv_proxy", "lead_time_mean_weeks", "lead_time_cv", "n_qualified_suppliers"],
        var_name="feature",
        value_name="value",
    )
    plt.figure(figsize=(14, 7))
    sns.violinplot(data=melt, x="feature", y="value", hue="criticality_class", cut=0)
    plt.title("Part-level feature distributions by criticality class")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    save()

    plt.figure(figsize=(8, 5))
    rate = compliance.groupby("criticality_class")["compliance_failure"].mean().reindex(["A", "B", "C"])
    sns.barplot(x=rate.index.astype(str), y=rate.values, color="#4C72B0")
    plt.ylabel("Compliance failure rate")
    plt.title("Compliance failure rate by criticality class")
    plt.tight_layout()
    save()

    plt.figure(figsize=(8, 5))
    sns.histplot(part_catalog["bom_in_degree"], bins=40, color="#55A868")
    plt.title("BOM in-degree distribution")
    plt.tight_layout()
    save()

    plt.figure(figsize=(9, 5))
    sns.kdeplot(data=supplier_panel, x="otd_oem_measured", hue="supplier_at_risk_flag", common_norm=False)
    plt.title("OEM-measured OTD by supplier at-risk flag")
    plt.tight_layout()
    save()

    num_cols = [c for c in part_catalog.select_dtypes(include=["number"]).columns if part_catalog[c].notna().sum() > 10]
    plt.figure(figsize=(12, 10))
    sns.heatmap(part_catalog[num_cols].corr(numeric_only=True), cmap="vlag", center=0)
    plt.title("Correlation heatmap (part-catalog numeric features)")
    plt.tight_layout()
    save()

    bench = pd.DataFrame(
        [
            ("Lead time CV (pooled mean)", float(part_catalog["lead_time_cv"].mean()), 0.15, 0.35),
            ("Tier-1 OTD target anchor", float(OTD_TARGET_HIGH), 0.98, 0.99),
            ("Escalation threshold anchor", float(OTD_ESCALATION_THRESHOLD), 0.96, 0.98),
            (
                "Reschedule burden (pp mean)",
                float(supplier_panel["reschedule_burden_pp"].mean()),
                RESCHEDULE_BURDEN_PP_MIN,
                RESCHEDULE_BURDEN_PP_MAX,
            ),
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
    save()


def write_run_manifest(out_dir: Path) -> None:
    """Write generator configuration for reproducibility (always; overwrites each run)."""
    payload: Dict[str, object] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": int(RANDOM_SEED),
        "n_parts": int(N_PARTS),
        "synthetic_generator_mode": SYNTHETIC_GENERATOR_MODE,
        "abc_label_noise": float(ABC_LABEL_NOISE),
        "abc_feature_overlap": float(ABC_FEATURE_OVERLAP),
        "abc_score_noise": float(ABC_SCORE_NOISE),
    }
    if SYNTHETIC_GENERATOR_MODE == "latent":
        payload["latent_to_label_noise"] = float(LATENT_TO_LABEL_NOISE)
        payload["latent_to_feature_noise"] = float(LATENT_TO_FEATURE_NOISE)
    (out_dir / "run_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def export_artifacts(part_catalog, supplier_panel_fe, compliance, bom_graph, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    part_out = part_catalog.drop(columns=[c for c in ["dedup_key"] if c in part_catalog.columns], errors="ignore")
    part_out.to_csv(out_dir / "part_catalog.csv", index=False)
    supplier_panel_fe.to_csv(out_dir / "supplier_history.csv", index=False)
    compliance.to_csv(out_dir / "compliance_outcomes.csv", index=False)
    with open(out_dir / "bom_graph.gpickle", "wb") as f:
        pickle.dump(bom_graph, f, protocol=pickle.HIGHEST_PROTOCOL)
    write_data_dictionary(out_dir / "data_dictionary.md", part_catalog, supplier_panel_fe, compliance)
    print("Wrote:", sorted(p.name for p in out_dir.iterdir()))


def print_summary(part_catalog) -> None:
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


def main() -> None:
    global OUT_DIR, rng, ABC_LABEL_NOISE, ABC_FEATURE_OVERLAP, ABC_SCORE_NOISE
    global SYNTHETIC_GENERATOR_MODE, LATENT_TO_LABEL_NOISE, LATENT_TO_FEATURE_NOISE
    parser = argparse.ArgumentParser(description="Generate hybrid synthetic dataset (DataCo + UCI backbone).")
    parser.add_argument("--output-dir", type=Path, default=None, help="Export directory (default: <repo>/outputs)")
    parser.add_argument("--plots", action="store_true", help="Write EDA PNGs to <output-dir>/figures")
    parser.add_argument("--n-parts", type=int, default=None, help="Override N_PARTS")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use in-memory DataCo-shaped stub instead of data/DataCoSupplyChainDataset.csv (smoke test only).",
    )
    parser.add_argument(
        "--abc-label-noise",
        type=float,
        default=None,
        metavar="P",
        help="Probability each part's ABC label is replaced with uniform A/B/C after quantile ABC (0–1). "
        "Overrides ABC_LABEL_NOISE env for this run.",
    )
    parser.add_argument(
        "--abc-feature-overlap",
        type=float,
        default=None,
        metavar="P",
        help="Blend literature augmentation (suppliers, lead times, stockouts, substitutability) from "
        "class-specific priors toward pooled priors (0–1). Overrides ABC_FEATURE_OVERLAP env.",
    )
    parser.add_argument(
        "--abc-score-noise",
        type=float,
        default=None,
        metavar="S",
        help="Multiplier on N(0,1) jitter in ABC score before quantile assignment (>=0). "
        "Overrides ABC_SCORE_NOISE env.",
    )
    parser.add_argument(
        "--generator-mode",
        type=str,
        choices=["legacy", "latent"],
        default=None,
        help="Synthetic catalog: 'legacy' (price/CV ABC + class-conditional augmentation) or "
        "'latent' (latent score drives labels and observables). Overrides SYNTHETIC_GENERATOR_MODE env.",
    )
    parser.add_argument(
        "--latent-label-noise",
        type=float,
        default=None,
        metavar="S",
        help="Std of Gaussian noise on latent before ABC quantile cut (latent mode only). "
        "Overrides LATENT_TO_LABEL_NOISE env.",
    )
    parser.add_argument(
        "--latent-feature-noise",
        type=float,
        default=None,
        metavar="S",
        help="Scales observation noise on latent-derived part features (latent mode only). "
        "Overrides LATENT_TO_FEATURE_NOISE env.",
    )
    args = parser.parse_args()

    os.chdir(ROOT)
    out_dir = (args.output_dir or (ROOT / "outputs")).resolve()
    globals()["OUT_DIR"] = out_dir
    if args.n_parts is not None:
        globals()["N_PARTS"] = args.n_parts
    if args.abc_label_noise is not None:
        ABC_LABEL_NOISE = float(args.abc_label_noise)
    if args.abc_feature_overlap is not None:
        ABC_FEATURE_OVERLAP = float(args.abc_feature_overlap)
    if args.abc_score_noise is not None:
        ABC_SCORE_NOISE = float(args.abc_score_noise)
    if args.generator_mode is not None:
        SYNTHETIC_GENERATOR_MODE = str(args.generator_mode).strip().lower()
    if args.latent_label_noise is not None:
        LATENT_TO_LABEL_NOISE = float(args.latent_label_noise)
    if args.latent_feature_noise is not None:
        LATENT_TO_FEATURE_NOISE = float(args.latent_feature_noise)

    np.random.seed(RANDOM_SEED)
    globals()["rng"] = np.random.default_rng(RANDOM_SEED)

    import importlib.util
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.figsize"] = (10, 5)

    print("OUT_DIR =", OUT_DIR)
    print("SYNTHETIC_GENERATOR_MODE =", SYNTHETIC_GENERATOR_MODE)
    print("ABC_LABEL_NOISE =", ABC_LABEL_NOISE)
    print("ABC_FEATURE_OVERLAP =", ABC_FEATURE_OVERLAP)
    print("ABC_SCORE_NOISE =", ABC_SCORE_NOISE)
    if SYNTHETIC_GENERATOR_MODE == "latent":
        print("LATENT_TO_LABEL_NOISE =", LATENT_TO_LABEL_NOISE)
        print("LATENT_TO_FEATURE_NOISE =", LATENT_TO_FEATURE_NOISE)
    if args.demo:
        print("WARNING: --demo uses a synthetic DataCo stub, not the public DataCo dataset.")
        dataco_raw = make_demo_dataco_stub(rng)
    else:
        dataco_raw = load_dataco_supply_chain()
    uci_raw = load_uci_online_retail()
    assert len(dataco_raw) > 0 and len(uci_raw) > 0
    assert dataco_raw["on_time_delivery"].notna().mean() > 0.5
    print("DataCo rows:", len(dataco_raw), "UCI rows:", len(uci_raw))

    part_catalog = build_unified_part_catalog(dataco_raw, uci_raw, N_PARTS, rng)
    print(part_catalog[["criticality_class", "public_backbone"]].value_counts().head(10))

    spec = importlib.util.spec_from_file_location("bom_graph", ROOT / "bom_graph.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    generate_bom_dag = mod.generate_bom_dag
    compute_bom_position_features = mod.compute_bom_position_features
    print("[BOM] loaded from", ROOT / "bom_graph.py")

    crit_series = part_catalog.set_index("part_id")["criticality_class"]
    latent_series = None
    latent_dict = None
    if SYNTHETIC_GENERATOR_MODE == "latent" and "latent_criticality_score" in part_catalog.columns:
        latent_series = part_catalog.set_index("part_id")["latent_criticality_score"].astype(float)
        latent_dict = latent_series.to_dict()
    bom_graph = generate_bom_dag(
        part_ids=part_catalog["part_id"].tolist(),
        depth_min=BOM_DEPTH_MIN,
        depth_max=BOM_DEPTH_MAX,
        fanout_min=BOM_FANOUT_MIN,
        fanout_max=BOM_FANOUT_MAX,
        criticality=crit_series,
        rng=rng,
        latent=latent_series,
    )
    bom_features = compute_bom_position_features(bom_graph, crit_series.to_dict(), latent_by_part=latent_dict)
    part_catalog = part_catalog.set_index("part_id").join(bom_features, how="left").reset_index()
    assert part_catalog["bom_in_degree"].between(0, 500).all()
    assert float(part_catalog["bom_in_degree"].mean()) > 1.0
    _cascade = part_catalog["bom_criticality_propagation_score"].astype(float)
    assert float(_cascade.notna().mean()) >= 0.99, "BOM cascade score missing for too many parts; compliance calibration would be misleading."
    assert np.isfinite(_cascade.fillna(0.0)).all(), "Non-finite bom_criticality_propagation_score after BOM join."
    print(bom_graph)

    supplier_panel = build_monthly_supplier_panel(dataco_raw, part_catalog, N_MONTHS, rng)
    compliance = generate_compliance_outcomes(part_catalog, supplier_panel, rng, COMPLIANCE_FAILURE_RATE_TARGET)
    supplier_panel_fe = add_rolling_supplier_features(supplier_panel)
    feature_matrix = engineer_final_matrix(part_catalog, supplier_panel_fe)
    assert feature_matrix["otd_oem_roll3"].notna().all()
    _ = feature_matrix

    if args.plots:
        run_eda(part_catalog, compliance, supplier_panel, OUT_DIR / "figures")

    export_artifacts(part_catalog, supplier_panel_fe, compliance, bom_graph, OUT_DIR)
    write_run_manifest(OUT_DIR)
    print_summary(part_catalog)


if __name__ == "__main__":
    main()
