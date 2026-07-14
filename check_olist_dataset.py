#!/usr/bin/env python3
"""
Feasibility check + modeling-ready panel build for the Olist Brazilian E-Commerce dataset as a
second real validation domain. Read-only investigation -- no modeling/generator code touched.

PROVENANCE NOTE (verified before writing this script): the Kaggle API is not configured in this
environment (no `kaggle` package, no ~/.kaggle/kaggle.json, no KAGGLE_USERNAME/KAGGLE_KEY) and the
Kaggle dataset page itself returns 404 to a non-browser client (Kaggle's site requires JS
rendering; this is not evidence the dataset doesn't exist -- it's a well-known, real, widely-used
dataset). The specific GitHub mirror given in the original request
(katetotka/brazilian_ecommerce) returned 404 on every file -- verified via direct HTTP checks, not
assumed. Used GitHub's API to find and verify a working alternative mirror instead of guessing
further URLs: spdrio/Brazilian-E-Commerce-Public-Dataset-by-Olist (files/ subdirectory), confirmed
to serve all 5 needed real files with the exact claimed schema (order_delivered_customer_date,
order_estimated_delivery_date, product_id, seller_id, price, product_category_name, seller_state
all verified present via direct header inspection before this script was written).

Run: uv run python check_olist_dataset.py
Writes: data/olist/*.csv, outputs/olist_feasibility_report.txt, outputs/olist_modeling_ready.csv (if viable)
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DATA_DIR = ROOT / "data" / "olist"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; research-data-fetch/1.0)"}

MIRROR_BASE = "https://raw.githubusercontent.com/spdrio/Brazilian-E-Commerce-Public-Dataset-by-Olist/master/files"
FILES = [
    "olist_orders_dataset",
    "olist_order_items_dataset",
    "olist_products_dataset",
    "olist_sellers_dataset",
    "product_category_name_translation",
]


# ---------------------------------------------------------------------------
# Step 1: Download
# ---------------------------------------------------------------------------
def step1_download() -> dict:
    print("=== Step 1: Download ===")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    log = {"kaggle_api_available": False, "source_used": MIRROR_BASE, "files": {}}
    try:
        import kaggle  # noqa: F401
        log["kaggle_api_available"] = True
    except ImportError:
        print("  Kaggle API not installed/configured -- using verified GitHub mirror instead.")

    for f in FILES:
        url = f"{MIRROR_BASE}/{f}.csv"
        dest = DATA_DIR / f"{f}.csv"
        r = requests.get(url, headers=HEADERS, timeout=60)
        r.raise_for_status()
        dest.write_bytes(r.content)
        df_check = pd.read_csv(dest)
        print(f"  {f}.csv: {len(df_check)} rows, {dest.stat().st_size / 1024:.0f} KB")
        log["files"][f] = {"rows": len(df_check), "size_kb": round(dest.stat().st_size / 1024, 1)}
    return log


# ---------------------------------------------------------------------------
# Step 2: Load and join
# ---------------------------------------------------------------------------
def step2_load_and_join() -> tuple[pd.DataFrame, dict]:
    print("\n=== Step 2: Load and join ===")
    orders = pd.read_csv(DATA_DIR / "olist_orders_dataset.csv")
    items = pd.read_csv(DATA_DIR / "olist_order_items_dataset.csv")
    products = pd.read_csv(DATA_DIR / "olist_products_dataset.csv")
    cat_trans = pd.read_csv(DATA_DIR / "product_category_name_translation.csv")
    sellers = pd.read_csv(DATA_DIR / "olist_sellers_dataset.csv")

    drop_log = {"start_order_items_rows": len(items)}

    df = items.merge(orders, on="order_id", how="left")
    df = df.merge(products, on="product_id", how="left")
    df = df.merge(cat_trans, on="product_category_name", how="left")
    df = df.merge(sellers, on="seller_id", how="left")
    drop_log["after_joins_rows"] = len(df)

    n_before = len(df)
    df = df[df["order_status"] == "delivered"]
    drop_log["after_status_delivered_filter"] = len(df)
    print(f"  Dropped {n_before - len(df)} rows where order_status != 'delivered'")

    n_before = len(df)
    df = df.dropna(subset=["order_delivered_customer_date", "order_estimated_delivery_date"])
    drop_log["after_both_dates_present_filter"] = len(df)
    print(f"  Dropped {n_before - len(df)} rows missing delivered/estimated date")

    print(f"  Final joined+filtered shape: {df.shape}")
    return df, drop_log


# ---------------------------------------------------------------------------
# Step 3: Compliance outcome
# ---------------------------------------------------------------------------
def step3_compliance(df: pd.DataFrame) -> pd.DataFrame:
    print("\n=== Step 3: Compliance outcome ===")
    df = df.copy()
    df["order_estimated_delivery_date"] = pd.to_datetime(df["order_estimated_delivery_date"])
    df["order_delivered_customer_date"] = pd.to_datetime(df["order_delivered_customer_date"])
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    df["days_late"] = (df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]).dt.days
    df["compliance_failure"] = (df["days_late"] > 0).astype(int)
    df["month"] = df["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp()

    print("days_late describe:")
    print(df["days_late"].describe().round(2))
    print("\ndays_late percentiles:")
    print(df["days_late"].quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).round(2))
    print(f"\nOverall compliance_failure rate: {df['compliance_failure'].mean():.4f}")
    print(f"Rows with both dates present: {len(df)}")
    print("\nCompliance failure rate by category (top 20 by row count):")
    by_cat = df.groupby("product_category_name_english", observed=True).agg(
        n=("compliance_failure", "size"), failure_rate=("compliance_failure", "mean")
    ).sort_values("n", ascending=False).head(20)
    print(by_cat.round(4))
    return df


# ---------------------------------------------------------------------------
# Step 4: Product-seller-month panel
# ---------------------------------------------------------------------------
def step4_panel(df: pd.DataFrame) -> pd.DataFrame:
    print("\n=== Step 4: Product-seller-month panel ===")
    panel = df.groupby(["product_id", "seller_id", "month"], observed=True).agg(
        compliance_failure=("compliance_failure", "max"),
        n_orders=("order_id", "count"),
        mean_days_late=("days_late", "mean"),
        any_late=("compliance_failure", "max"),
        price=("price", "mean"),
    ).reset_index()
    print(f"Panel shape: {panel.shape}")
    print("\nn_orders per product-seller-month distribution:")
    print(panel["n_orders"].describe().round(2))
    print(f"Share with n_orders==1: {(panel['n_orders'] == 1).mean():.4f}")
    return panel


# ---------------------------------------------------------------------------
# Step 5: ABC proxy classification
# ---------------------------------------------------------------------------
def step5_abc(df: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    print("\n=== Step 5: ABC proxy classification ===")
    price_by_product = df.groupby("product_id")["price"].mean().rename("mean_price")
    monthly_counts = panel.groupby(["product_id", "month"], observed=True)["n_orders"].sum().reset_index()
    demand_cv = monthly_counts.groupby("product_id")["n_orders"].agg(
        lambda s: float(s.std(ddof=1) / s.mean()) if s.mean() > 0 and len(s) > 1 else 0.0
    ).rename("demand_cv")

    prod = pd.concat([price_by_product, demand_cv], axis=1).reset_index()
    prod["price_rank"] = prod["mean_price"].rank(pct=True)
    prod["demand_cv_rank"] = prod["demand_cv"].rank(pct=True)
    prod["composite_score"] = 0.7 * prod["price_rank"] + 0.3 * prod["demand_cv_rank"]

    q_a, q_b = prod["composite_score"].quantile([0.80, 0.50])
    prod["criticality_class"] = np.where(
        prod["composite_score"] >= q_a, "A", np.where(prod["composite_score"] >= q_b, "B", "C")
    )

    shares = prod["criticality_class"].value_counts(normalize=True)
    print("ABC shares:", shares.round(4).to_dict())

    cat_map = df.drop_duplicates("product_id").set_index("product_id")["product_category_name_english"]
    prod["product_category_en"] = prod["product_id"].map(cat_map)
    print("\nTop 5 categories per class:")
    for c in ["A", "B", "C"]:
        top = prod[prod["criticality_class"] == c]["product_category_en"].value_counts().head(5)
        print(f"  {c}: {top.to_dict()}")
    return prod


# ---------------------------------------------------------------------------
# Step 6: Part-level features
# ---------------------------------------------------------------------------
def step6_part_features(df: pd.DataFrame, panel: pd.DataFrame, abc: pd.DataFrame) -> pd.DataFrame:
    print("\n=== Step 6: Part-level features ===")
    price_stats = df.groupby("product_id")["price"].agg(
        unit_cost_mean="mean", unit_cost_std="std"
    ).reset_index()
    price_stats["unit_cost_cv"] = (price_stats["unit_cost_std"] / price_stats["unit_cost_mean"]).fillna(0.0)

    monthly = panel.groupby(["product_id", "month"], observed=True)["n_orders"].sum().reset_index()
    demand_stats = monthly.groupby("product_id")["n_orders"].agg(
        demand_mean_monthly="mean",
        demand_cv_monthly=lambda s: float(s.std(ddof=1) / s.mean()) if s.mean() > 0 and len(s) > 1 else 0.0,
    ).reset_index()

    n_sellers = df.groupby("product_id")["seller_id"].nunique().rename("n_sellers").reset_index()

    part = price_stats.merge(demand_stats, on="product_id").merge(n_sellers, on="product_id")
    part = part.merge(abc[["product_id", "criticality_class", "product_category_en"]], on="product_id")
    print(f"Part-level table shape: {part.shape} (n unique products)")
    return part


# ---------------------------------------------------------------------------
# Step 7: Supplier-level rolling features
# ---------------------------------------------------------------------------
def step7_supplier_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n=== Step 7: Supplier-level rolling features ===")
    df = df.copy()
    seller_month = df.groupby(["seller_id", "month"], observed=True).agg(
        otd_rate_month=("compliance_failure", lambda s: 1.0 - s.mean()),
        n_orders=("order_id", "count"),
    ).reset_index()
    seller_month = seller_month.sort_values(["seller_id", "month"])
    g = seller_month.groupby("seller_id", group_keys=False)
    seller_month["otd_roll3"] = g["otd_rate_month"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    seller_month["otd_roll6"] = g["otd_rate_month"].transform(lambda s: s.rolling(6, min_periods=1).mean())

    seller_state = df.drop_duplicates("seller_id").set_index("seller_id")["seller_state"]
    seller_month["seller_state"] = seller_month["seller_id"].map(seller_state)
    print(f"Supplier-month table shape: {seller_month.shape}")
    return seller_month


# ---------------------------------------------------------------------------
# Step 8: Final join / modeling-ready file
# ---------------------------------------------------------------------------
def step8_final_join(panel: pd.DataFrame, part: pd.DataFrame, seller_feat: pd.DataFrame) -> pd.DataFrame:
    print("\n=== Step 8: Final join ===")
    final = panel.merge(part, on="product_id", how="left")
    final = final.merge(seller_feat, on=["seller_id", "month"], how="left", suffixes=("", "_sellermonth"))
    print(f"Final modeling-ready shape: {final.shape}")
    print(f"Compliance failure rate: {final['compliance_failure'].mean():.4f}")
    print(f"Unique products (N): {final['product_id'].nunique()}")
    print(f"Unique sellers: {final['seller_id'].nunique()}")
    return final


# ---------------------------------------------------------------------------
# Step 9: Feasibility Q&A
# ---------------------------------------------------------------------------
def step9_feasibility(df: pd.DataFrame, panel: pd.DataFrame, final: pd.DataFrame) -> dict:
    print("\n=== Step 9: Feasibility Q&A ===")
    n_products = final["product_id"].nunique()
    n_pairs = len(panel)
    n_months = df["order_purchase_timestamp"].dt.to_period("M").nunique()

    answers = {
        "Q1_compliance_outcome": {"verdict": "YES", "note": "order_delivered_customer_date vs order_estimated_delivery_date gives an unambiguous binary label; both fields verified present."},
        "Q2_part_proxy": {"verdict": "YES", "note": "product_id is a real, stable identifier."},
        "Q3_supplier_proxy": {"verdict": "YES", "note": "seller_id is a real, stable identifier, distinct from customer_id."},
        "Q4_unit_cost": {"verdict": "YES", "note": "price field present at order-item grain, real values."},
        "Q5_product_category": {"verdict": "YES", "note": "product_category_name (+ English translation) present, usable for ABC proxy."},
        "Q6_rolling_otd": {"verdict": "YES", "note": "Computed 3-month and 6-month rolling seller-level OTD in Step 7."},
        "Q7_n_products": {"verdict": "YES" if n_products >= 500 else "NO", "note": f"{n_products} unique products in the final panel."},
        "Q8_n_pairs": {"verdict": "YES" if n_pairs >= 5000 else "NO", "note": f"{n_pairs} unique product-seller-month pairs."},
        "Q9_temporal_span": {"verdict": "YES" if n_months >= 18 else "NO", "note": f"{n_months} unique months of order data."},
        "Q10_license": {"verdict": "PARTIAL", "note": "Kaggle page states CC-BY 4.0 for the official dataset; this run used a third-party GitHub mirror, not the official Kaggle download, so the license should be re-confirmed against the official Kaggle listing before citing, same caveat as the USAID check."},
        "Q11_domain_difference": {"verdict": "YES", "note": "Brazilian e-commerce retail vs. automotive/industrial manufacturing -- a real, manageable-in-one-paragraph domain gap, same class of caveat as our synthetic dataset's DataCo/UCI backbone."},
        "Q12_bom_structure": {"verdict": "NO", "note": "No BOM/assembly structure exists, expected -- same limitation already acknowledged for our own synthetic BOM and for the USAID feasibility check."},
    }
    for k, v in answers.items():
        print(f"  {k}: {v['verdict']} -- {v['note']}")
    return answers


# ---------------------------------------------------------------------------
# Step 10: Write report
# ---------------------------------------------------------------------------
def step10_report(download_log: dict, drop_log: dict, df: pd.DataFrame, panel: pd.DataFrame, final: pd.DataFrame, answers: dict) -> str:
    print("\n=== Step 10: Writing report ===")
    n_yes = sum(1 for a in answers.values() if a["verdict"] == "YES")
    verdict = "VIABLE" if n_yes >= 10 else ("PARTIALLY VIABLE" if n_yes >= 7 else "NOT VIABLE")

    lines = [
        "OLIST BRAZILIAN E-COMMERCE DATASET FEASIBILITY REPORT",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "PROVENANCE NOTE: Kaggle API not configured in this environment; sourced from a verified",
        "third-party GitHub mirror (spdrio/Brazilian-E-Commerce-Public-Dataset-by-Olist), NOT the",
        "official Kaggle download. Schema independently verified (column-by-column) against the",
        "known official schema before use. License should be re-confirmed against the official",
        "Kaggle listing before citing in the paper.",
        "",
        f"OVERALL VERDICT: {verdict}",
        (
            "Real, unambiguous compliance outcome (estimated vs. actual delivery date), real "
            "product/seller/price identifiers, sufficient scale and temporal span. "
            f"{n_yes}/12 feasibility questions answered YES."
        ),
        "",
        "DATASET STATISTICS:",
        f"- Total order-item rows (post-join): {drop_log['after_joins_rows']}",
        f"- Delivered orders with both date fields: {drop_log['after_both_dates_present_filter']}",
        f"- Overall compliance failure rate: {df['compliance_failure'].mean():.4f}",
        f"- Date range: {df['order_purchase_timestamp'].min().date()} to {df['order_purchase_timestamp'].max().date()}",
        f"- Unique products (part proxies): {final['product_id'].nunique()}",
        f"- Unique sellers (supplier proxies): {final['seller_id'].nunique()}",
        f"- Unique product-seller-month pairs (panel size): {len(panel)}",
        f"- Unique months: {df['order_purchase_timestamp'].dt.to_period('M').nunique()}",
        "",
        "Q1-Q12 ANSWERS:",
    ]
    for k, v in answers.items():
        lines.append(f"{k}: {v['verdict']} -- {v['note']}")
    lines += [
        "",
        "FRAMEWORK COLUMN MAPPING:",
        f"{'Our framework column':32s} | Olist analog",
        "-" * 70,
        f"{'part_id':32s} | product_id",
        f"{'supplier_id':32s} | seller_id",
        f"{'unit_cost':32s} | price (order_items)",
        f"{'product_category':32s} | product_category_name_english",
        f"{'compliance_failure':32s} | order_delivered_customer_date > order_estimated_delivery_date",
        f"{'month':32s} | order_purchase_timestamp truncated to month",
        f"{'standard_lead_time_days':32s} | (derivable: purchase to delivered)",
        f"{'number_of_qualified_suppliers':32s} | n_sellers per product (distinct sellers who fulfilled it)",
        f"{'BOM structure':32s} | (none -- not applicable to retail e-commerce)",
        "",
        "DOMAIN DIFFERENCES TO ACKNOWLEDGE IN PAPER:",
        "- Brazilian e-commerce retail vs. automotive/industrial manufacturing -- different criticality drivers entirely (consumer demand/price vs. production-line risk).",
        "- No BOM/assembly structure -- matches our synthetic dataset's own limitation, not a new one.",
        "- compliance_failure here is defined relative to Olist's own ESTIMATED delivery date (a promise made to the customer), not a fixed production schedule -- a different but analogous notion of 'compliance.'",
        "- Product-seller relationship may be less exclusive than a manufacturing supplier relationship; a product can have many sellers, unlike typical sole/dual-sourced manufacturing parts.",
        "",
        "CITATION (IEEE format):",
        'O. Sionek et al., "Brazilian E-Commerce Public Dataset by Olist," Kaggle, 2018. [Online]. Available: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce. License: CC BY 4.0 (per Kaggle listing; re-confirm before final citation since this run used a third-party mirror, not the official download).',
        "",
        "RECOMMENDED NEXT STEPS:",
    ]
    if verdict in ("VIABLE", "PARTIALLY VIABLE"):
        lines.append("- Re-confirm license via the official Kaggle listing (requires Kaggle account/API setup, not done in this environment).")
        lines.append("- outputs/olist_modeling_ready.csv has been written -- review its columns against Layer 2's expected feature set before integration; column names differ from our synthetic pipeline's naming and will need an adapter, not a drop-in replacement.")
        lines.append("- Consider whether n_orders==1 dominating the panel (see Step 4 output) affects the reliability of month-level aggregates -- may need a minimum-orders filter, same class of thin-sample issue we handled in our own supplier panel.")
    else:
        lines.append("- Not viable as-is; revisit which specific questions failed above.")

    report_text = "\n".join(lines)
    (OUT / "olist_feasibility_report.txt").write_text(report_text, encoding="utf-8")
    return verdict


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    download_log = step1_download()
    df, drop_log = step2_load_and_join()
    df = step3_compliance(df)
    panel = step4_panel(df)
    abc = step5_abc(df, panel)
    part = step6_part_features(df, panel, abc)
    seller_feat = step7_supplier_features(df)
    final = step8_final_join(panel, part, seller_feat)
    answers = step9_feasibility(df, panel, final)
    verdict = step10_report(download_log, drop_log, df, panel, final, answers)

    if verdict in ("VIABLE", "PARTIALLY VIABLE"):
        final.to_csv(OUT / "olist_modeling_ready.csv", index=False)
        print("\nMODELING-READY FILE WRITTEN: outputs/olist_modeling_ready.csv")
        print(f"Rows: {len(final)}")
        print(f"Unique products (N): {final['product_id'].nunique()}")
        print(f"Unique sellers: {final['seller_id'].nunique()}")
        print(f"Compliance failure rate: {final['compliance_failure'].mean() * 100:.2f}%")
        print("Ready for Layer 2 integration (will need a column-name adapter -- see report).")
    else:
        print(f"\nVerdict: {verdict} -- modeling-ready file not written.")

    print(f"\nWrote {OUT / 'olist_feasibility_report.txt'}")


if __name__ == "__main__":
    main()
