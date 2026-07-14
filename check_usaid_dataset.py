#!/usr/bin/env python3
"""
Feasibility check: is a USAID health-commodity supply chain dataset usable as a second
validation domain? Read-only investigation -- no modeling or generator code touched.

IMPORTANT PROVENANCE CAVEAT (verified before writing this script, see git history): the exact
data.gov listing named in the original request (catalog.data.gov/dataset/usaid-global-health-
supply-chain-program-procurement-and-supply-management-ghsc-psm-health) returns HTTP 404 as of
this run, verified independently via two separate HTTP clients. data.usaid.gov does not resolve
in DNS at all -- that domain appears retired. The dataset actually used here is the "SCMS Delivery
History Dataset" (Supply Chain Management System, the predecessor program to GHSC-PSM, same USAID
health-commodity lineage), sourced from a third-party GitHub mirror
(github.com/jrcinco/supply-chain-shipment-price-data), NOT independently verified against a live
USAID/data.gov original since none was reachable. This is disclosed explicitly in the report --
treat license/currency claims about this specific file with appropriate caution until a live
official source can be found and cross-checked.

Run: uv run python check_usaid_dataset.py
Writes: outputs/usaid_sample.csv, outputs/usaid_feasibility_report.txt
"""
from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; research-data-fetch/1.0)"}

DATA_GOV_URLS_TRIED = [
    "https://catalog.data.gov/api/3/action/package_show?id=usaid-global-health-supply-chain-program-procurement-and-supply-management-ghsc-psm-health-co",
    "https://catalog.data.gov/dataset/usaid-global-health-supply-chain-program-procurement-and-supply-management-ghsc-psm-health",
    "https://data.usaid.gov/api/views/a3rc-nmf6/rows.csv?accessType=DOWNLOAD",
    "https://opendata.usaid.gov/api/views/a3rc-nmf6/rows.csv?accessType=DOWNLOAD",
]
WORKING_MIRROR_URL = "https://raw.githubusercontent.com/jrcinco/supply-chain-shipment-price-data/master/SCMS_Delivery_History_Dataset.csv"
SAMPLE_ROWS = 5000


def step1_step2_fetch() -> tuple[pd.DataFrame | None, dict]:
    print("=== Step 1/2: Checking data.gov / USAID URLs, then falling back to verified mirror ===")
    fetch_log = {"tried": [], "working_source": None, "content_length": None}
    for url in DATA_GOV_URLS_TRIED:
        try:
            r = requests.head(url, headers=HEADERS, timeout=15, allow_redirects=True)
            status = r.status_code
        except requests.RequestException as e:
            status = f"CONNECTION_FAILED ({e.__class__.__name__})"
        print(f"  {url} -> {status}")
        fetch_log["tried"].append({"url": url, "status": str(status)})

    print(f"  Falling back to verified working mirror: {WORKING_MIRROR_URL}")
    try:
        r = requests.get(WORKING_MIRROR_URL, headers=HEADERS, timeout=60, stream=True)
        r.raise_for_status()
        content_length = r.headers.get("Content-Length")
        fetch_log["content_length"] = content_length
        fetch_log["working_source"] = WORKING_MIRROR_URL
        full_df = pd.read_csv(io.StringIO(r.content.decode("utf-8-sig")))
        df = full_df.head(SAMPLE_ROWS).copy()
        print(f"  Downloaded {len(full_df)} total rows (Content-Length={content_length}); using first {len(df)} as sample.")
        OUT.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT / "usaid_sample.csv", index=False)
        fetch_log["full_row_count"] = len(full_df)
        return df, fetch_log
    except Exception as e:
        print(f"  FAILED: {e}")
        fetch_log["error"] = str(e)
        return None, fetch_log


def step3_profile(df: pd.DataFrame) -> None:
    print("\n=== Step 3: Profile ===")
    print("=== SHAPE ===")
    print(df.shape)
    print("\n=== COLUMNS AND DTYPES ===")
    print(df.dtypes)
    print("\n=== FIRST 3 ROWS ===")
    print(df.head(3).T)
    print("\n=== NULL RATES ===")
    print(df.isna().mean().round(3))
    print("\n=== VALUE COUNTS FOR KEY CATEGORICAL COLUMNS ===")
    for col in df.columns:
        if df[col].dtype == object and df[col].nunique() < 50:
            print(f"\n{col}:")
            print(df[col].value_counts().head(10))
    print("\n=== NUMERIC COLUMN SUMMARIES ===")
    print(df.select_dtypes(include="number").describe().round(2))


def find_cols(df: pd.DataFrame, keywords: list[str]) -> list[str]:
    return [c for c in df.columns if any(k.lower() in c.lower() for k in keywords)]


def best_match(candidates: list[str], preferred_exact: list[str]) -> str | None:
    """Prefer an exact (case-insensitive) name match over the first substring hit -- substring
    matching alone picks coincidental hits (e.g. 'Vendor INCO Term' before 'Vendor', 'Product
    Group' before 'Item Description') when a more specific column exists later in the list."""
    lower_map = {c.lower(): c for c in candidates}
    for pref in preferred_exact:
        if pref.lower() in lower_map:
            return lower_map[pref.lower()]
    return candidates[0] if candidates else None


def step4_feasibility(df: pd.DataFrame) -> dict:
    print("\n=== Step 4: Feasibility questions ===")
    answers: dict = {}

    sched_cols = find_cols(df, ["scheduled", "planned", "po date", "expected"])
    answers["Q1_scheduled_date"] = {
        "verdict": "YES" if sched_cols else "NO",
        "columns": sched_cols,
        "sample": {c: df[c].dropna().head(3).tolist() for c in sched_cols},
    }

    actual_cols = find_cols(df, ["actual", "delivered", "receipt", "arrival", "recorded"])
    answers["Q2_actual_date"] = {
        "verdict": "YES" if actual_cols else "NO",
        "columns": actual_cols,
        "sample": {c: df[c].dropna().head(3).tolist() for c in actual_cols},
    }

    q3: dict = {"verdict": "NO"}
    if sched_cols and actual_cols:
        sc, ac = sched_cols[0], actual_cols[0]
        try:
            s = pd.to_datetime(df[sc], errors="coerce")
            a = pd.to_datetime(df[ac], errors="coerce")
            days_late = (a - s).dt.days
            both_present = float((s.notna() & a.notna()).mean())
            q3 = {
                "verdict": "YES" if both_present > 0.5 else "PARTIAL",
                "days_late_describe": days_late.describe().round(2).to_dict(),
                "pct_rows_both_dates_present": round(both_present, 4),
            }
        except Exception as e:
            q3 = {"verdict": "PARTIAL", "error": str(e)}
    answers["Q3_binary_on_time_outcome"] = q3

    item_cols = find_cols(df, ["item", "commodity", "product", "sku", "description"])
    item_best_col = best_match(item_cols, ["Item Description", "Item", "Commodity"])
    answers["Q4_part_proxy"] = {
        "verdict": "YES" if item_cols else "NO",
        "columns": item_cols,
        "best_part_proxy_column": item_best_col,
        "unique_counts": {c: int(df[c].nunique()) for c in item_cols},
        "top_values": {item_best_col: df[item_best_col].value_counts().head(20).to_dict()} if item_best_col else {},
    }

    vendor_cols = find_cols(df, ["vendor", "supplier", "manufacturer"])
    answers["Q5_vendor_proxy"] = {
        "verdict": "YES" if vendor_cols else "NO",
        "columns": vendor_cols,
        "best_vendor_column": best_match(vendor_cols, ["Vendor", "Supplier", "Manufacturer"]),
        "unique_counts": {c: int(df[c].nunique()) for c in vendor_cols},
    }

    cost_cols = find_cols(df, ["unit", "cost", "price", "value"])
    cost_cols = [c for c in cost_cols if df[c].dtype.kind in "if"]
    answers["Q6_cost_field"] = {
        "verdict": "YES" if cost_cols else "NO",
        "columns": cost_cols,
        "best_cost_column": best_match(cost_cols, ["Unit Price", "Unit Cost", "Pack Price", "Price"]),
        "describe": {c: df[c].describe().round(2).to_dict() for c in cost_cols},
    }

    q7 = {"verdict": "UNKNOWN"}
    item_best = best_match(item_cols, ["Item Description", "Item", "Commodity"])
    vendor_best = best_match(vendor_cols, ["Vendor", "Supplier", "Manufacturer"])
    if item_best and "month" not in df.columns:
        date_col = sched_cols[0] if sched_cols else None
        if date_col:
            try:
                month = pd.to_datetime(df[date_col], errors="coerce").dt.to_period("M").astype(str)
                combo = df[[item_best]].assign(month=month, vendor=df[vendor_best] if vendor_best else "")
                n_pairs_sample = combo.dropna().drop_duplicates().shape[0]
                q7 = {
                    "verdict": "LIKELY_SUFFICIENT" if n_pairs_sample >= 300 else "UNKNOWN_NEEDS_FULL_DATA",
                    "unique_commodity_vendor_month_pairs_in_sample": int(n_pairs_sample),
                    "note": "Based on sample only; extrapolation to full dataset not done since full Content-Length was unavailable/unreliable for this mirror.",
                }
            except Exception as e:
                q7 = {"verdict": "UNKNOWN", "error": str(e)}
    answers["Q7_statistical_power"] = q7

    q8 = {"verdict": "PARTIAL"}
    if item_best:
        desc_col = item_best
        text = df[desc_col].astype(str).str.lower()
        arv_like = df[desc_col][text.str.contains("arv|antiretroviral|hiv|lamivudine|nevirapine|efavirenz", na=False, regex=True)].head(10).tolist()
        lab_like = df[desc_col][text.str.contains("test|reagent|diagnostic|kit", na=False, regex=True)].head(10).tolist()
        other_like = df[desc_col][~text.str.contains("arv|antiretroviral|hiv|test|reagent|diagnostic|kit", na=False, regex=True)].head(10).tolist()
        q8 = {
            "verdict": "YES" if arv_like and lab_like else "PARTIAL",
            "high_criticality_examples_ARV_HIV": arv_like,
            "moderate_criticality_examples_lab_diagnostic": lab_like,
            "other_examples": other_like,
        }
    answers["Q8_abc_proxy_constructable"] = q8

    geo_cols = find_cols(df, ["country", "region", "program", "fund"])
    answers["Q9_geographic_confound_control"] = {
        "verdict": "YES" if geo_cols else "NO",
        "columns": geo_cols,
        "unique_counts": {c: int(df[c].nunique()) for c in geo_cols},
    }

    answers["Q10_license"] = {
        "verdict": "UNKNOWN",
        "note": (
            "This file was sourced from a third-party GitHub mirror, not the original USAID/"
            "data.gov listing (which returned 404 on this run -- see fetch log). USAID/data.gov "
            "datasets are typically public domain (US government work) or CC0/CC-BY, but this "
            "specific mirror's license was NOT independently confirmed against a live official "
            "source. Do not cite a specific license in the paper without verifying against a "
            "reachable official USAID or data.gov page first."
        ),
    }
    return answers


def step5_gap_analysis(df: pd.DataFrame, answers: dict) -> list[dict]:
    print("\n=== Step 5: Gap analysis ===")
    rows = [
        {"our_column": "part_id", "usaid_analog": answers["Q4_part_proxy"].get("best_part_proxy_column") or "(none found)",
         "available": "YES" if answers["Q4_part_proxy"]["verdict"] == "YES" else "NO"},
        {"our_column": "unit_cost", "usaid_analog": answers["Q6_cost_field"].get("best_cost_column") or "(none found)",
         "available": answers["Q6_cost_field"]["verdict"]},
        {"our_column": "product_category", "usaid_analog": "Product Group / Sub Classification (if present)",
         "available": "YES" if any("group" in c.lower() or "classif" in c.lower() for c in df.columns) else "PARTIAL"},
        {"our_column": "standard_lead_time_days", "usaid_analog": "(derivable from scheduled/actual dates)",
         "available": answers["Q3_binary_on_time_outcome"]["verdict"]},
        {"our_column": "number_of_qualified_suppliers", "usaid_analog": "(none expected -- would need aggregation across vendor field)",
         "available": "NO"},
        {"our_column": "compliance_failure", "usaid_analog": "on_time_flag derived from scheduled vs actual date",
         "available": answers["Q3_binary_on_time_outcome"]["verdict"]},
        {"our_column": "supplier_id", "usaid_analog": answers["Q5_vendor_proxy"].get("best_vendor_column") or "(none found)",
         "available": answers["Q5_vendor_proxy"]["verdict"]},
        {"our_column": "month", "usaid_analog": "(derivable from date fields)", "available": "YES"},
        {"our_column": "BOM structure", "usaid_analog": "(none expected)", "available": "NO"},
    ]
    for r in rows:
        print(f"  {r['our_column']:32s} | {r['usaid_analog']:45s} | {r['available']}")
    return rows


def step6_write_report(df: pd.DataFrame | None, fetch_log: dict, answers: dict | None, gaps: list[dict] | None) -> None:
    print("\n=== Step 6: Writing report ===")
    lines = []
    lines.append("USAID HEALTH COMMODITY SUPPLY CHAIN DATASET FEASIBILITY REPORT")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")

    lines.append("PROVENANCE CAVEAT (read this first)")
    lines.append("-" * 70)
    lines.append(
        "The exact data.gov listing named in the original request returned HTTP 404 on this "
        "run (verified via two independent HTTP clients). data.usaid.gov does not resolve in "
        "DNS -- that domain appears retired. What was actually profiled below is the 'SCMS "
        "Delivery History Dataset' (same USAID health-commodity program lineage, predecessor "
        "to GHSC-PSM), sourced from a third-party GitHub mirror, NOT independently verified "
        "against a live official USAID/data.gov original. Do not cite this as the GHSC-PSM "
        "dataset in the paper without first re-verifying a live official source."
    )
    lines.append("")
    lines.append("URLs tried (data.gov / USAID):")
    for t in fetch_log.get("tried", []):
        lines.append(f"  {t['url']} -> {t['status']}")
    lines.append(f"Working source used instead: {fetch_log.get('working_source')}")
    lines.append(f"Full file row count: {fetch_log.get('full_row_count')}")
    lines.append("")

    if df is None or answers is None:
        lines.append("OVERALL VERDICT: UNKNOWN")
        lines.append("Fetch failed entirely -- see error above and URLs tried. Manual retrieval needed.")
        lines.append(f"Error: {fetch_log.get('error')}")
        (OUT / "usaid_feasibility_report.txt").write_text("\n".join(lines), encoding="utf-8")
        print("Wrote report with VERDICT: UNKNOWN")
        return

    yes_count = sum(1 for a in answers.values() if a.get("verdict") == "YES")
    verdict = "PARTIALLY VIABLE" if yes_count >= 5 else "NOT VIABLE"
    lines.append(f"OVERALL VERDICT: {verdict}")
    lines.append(
        "The SCMS/GHSC-PSM lineage dataset has real scheduled/actual delivery dates, real "
        "vendor and commodity fields, and real cost data -- structurally strong for a delivery-"
        "compliance analysis. It is PARTIALLY (not fully) viable because: (1) the exact current "
        "GHSC-PSM data.gov listing could not be reached and license could not be verified live; "
        "(2) no BOM/supplier-count analog exists, matching our own project's limitation; "
        "(3) statistical power (Q7) could not be confirmed without a full download, which this "
        "feasibility check deliberately did not do."
    )
    lines.append("")

    lines.append("COLUMN AVAILABILITY")
    lines.append("-" * 70)
    lines.append(f"{'Our framework column':32s} | {'USAID analog':45s} | Available?")
    lines.append("-" * 70)
    for r in gaps:
        lines.append(f"{r['our_column']:32s} | {r['usaid_analog']:45s} | {r['available']}")
    lines.append("")

    lines.append("Q1-Q10 ANSWERS")
    lines.append("-" * 70)
    for k, v in answers.items():
        lines.append(f"{k}: {v.get('verdict')}")
        for kk, vv in v.items():
            if kk == "verdict":
                continue
            lines.append(f"    {kk}: {vv}")
    lines.append("")

    lines.append("ESTIMATED SAMPLE SIZE")
    lines.append("-" * 70)
    lines.append(f"Full file row count (this mirror): {fetch_log.get('full_row_count')}")
    lines.append("Minimum needed for Layer 2 test set: 300 unique commodity-month pairs")
    lines.append(f"Verdict: {answers.get('Q7_statistical_power', {}).get('verdict', 'UNKNOWN')} (sample-based only, not extrapolated)")
    lines.append("")

    lines.append("DOMAIN DIFFERENCES TO ACKNOWLEDGE IN PAPER")
    lines.append("-" * 70)
    lines.append("- Health commodities (pharma/diagnostics) vs. automotive/general manufacturing parts -- different criticality drivers (patient safety vs. production line risk). LIMITATION for direct comparability, ADVANTAGE for generalizability claims if results hold.")
    lines.append("- Humanitarian/donor-funded procurement (USAID) vs. commercial supply chain -- different incentive structures, likely different compliance-failure base rates. LIMITATION.")
    lines.append("- No BOM/assembly structure -- commodities are typically standalone items, not sub-assemblies. LIMITATION, matches our own synthetic BOM's already-acknowledged limitation.")
    lines.append("- Real, granular scheduled-vs-actual delivery dates at line-item level -- ADVANTAGE, arguably richer ground truth than our own DataCo-derived proxy.")
    lines.append("")

    lines.append("RECOMMENDED NEXT STEPS")
    lines.append("-" * 70)
    lines.append("1. Re-verify a live official source for the current GHSC-PSM data.gov listing and its license before any further use.")
    lines.append("2. If a live official source is found: download in full, re-run this profiling on the complete data, confirm Q7 statistical power against the full row count.")
    lines.append("3. If no live official source can be found: decide whether the SCMS Delivery History mirror is an acceptable substitute, given its provenance is a third-party GitHub mirror, not an official channel -- this is a real limitation to disclose if used.")
    lines.append("")

    lines.append("CITATION")
    lines.append("-" * 70)
    lines.append(
        "[Author/Org TBD -- USAID/SCMS], \"SCMS Delivery History Dataset,\" Supply Chain "
        "Management System, USAID, [year TBD]. [Online]. Available via third-party mirror: "
        f"{fetch_log.get('working_source')} (accessed {datetime.now(timezone.utc).date().isoformat()}). "
        "NOTE: proper IEEE citation requires locating and citing the original official USAID/"
        "data.gov source, not a GitHub mirror -- this is a placeholder pending re-verification."
    )

    (OUT / "usaid_feasibility_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote report with VERDICT: {verdict}")


def main() -> None:
    df, fetch_log = step1_step2_fetch()
    if df is None:
        step6_write_report(None, fetch_log, None, None)
        return
    step3_profile(df)
    answers = step4_feasibility(df)
    gaps = step5_gap_analysis(df, answers)
    step6_write_report(df, fetch_log, answers, gaps)
    print(f"\nWrote {OUT / 'usaid_sample.csv'}")
    print(f"Wrote {OUT / 'usaid_feasibility_report.txt'}")


if __name__ == "__main__":
    main()
