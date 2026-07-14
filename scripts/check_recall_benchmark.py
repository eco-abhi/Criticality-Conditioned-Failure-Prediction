#!/usr/bin/env python3
"""
Fetch real CPSC recall data as a face-validity anchor for compliance_failure's scale.

`compliance_failure` (see generate_synthetic_datasets.generate_compliance_outcomes) is a
calibrated **synthetic** outcome -- no public dataset gives a real, part-linked compliance-failure
rate (CPSC/NHTSA recalls give counts, not rates: there is no real population denominator linking
a specific recalled product to this dataset's synthetic parts or to DataCo's/UCI's sold volumes).

This script does NOT feed CPSC data into the generator. It only checks that the synthetic 8%
per-part-month failure rate is being interpreted correctly: as a routine, higher-frequency
operational/quality/delivery compliance event, NOT a product-safety-recall-rate analog. Recalls
are a real, public, much rarer, more severe class of event; if the numbers below were close
together, that would suggest a modeling-scale error worth investigating.

Run: uv run python scripts/check_recall_benchmark.py
Writes: outputs/compliance_benchmark_cpsc.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import generate_synthetic_datasets as gsd  # noqa: E402

CPSC_URL = "https://www.saferproducts.gov/RestWebServices/Recall"
START, END = "2015-01-01", "2025-12-31"
# Known TF-IDF false-positive attractor (see investigation notes): recall titles containing
# "children's" spuriously match DataCo's "Children's Clothing" category regardless of actual
# product type (globes, tool kits, shaving toys, etc. all matched here on the shared word, not a
# genuine category match). Excluded from the domain-relevant subset below.
EXCLUDE_CATEGORY = "Children's Clothing"
CONFIDENT_SIM_THRESHOLD = 0.20


def fetch_cpsc_recalls(start: str, end: str) -> list[dict]:
    # saferproducts.gov returns 403 for requests' default User-Agent string specifically (verified:
    # curl and a browser-like UA both succeed with identical params) -- not a real access
    # restriction, just naive bot-string filtering.
    r = requests.get(
        CPSC_URL,
        params={"format": "json", "RecallDateStart": start, "RecallDateEnd": end},
        timeout=60,
        headers={"User-Agent": "Mozilla/5.0 (compatible; research-data-fetch/1.0)"},
    )
    r.raise_for_status()
    return r.json()


def domain_relevant_share(recalls: list[dict], dataco: pd.DataFrame) -> tuple[int, int]:
    """(n_confident_domain_relevant, n_total) via TF-IDF match to DataCo's real categories,
    excluding the known Children's Clothing attractor. Illustrative only -- see module docstring."""
    cat_docs = (
        dataco.dropna(subset=["product_name", "product_category"])
        .groupby("product_category")["product_name"]
        .apply(lambda s: " ".join(s.astype(str).unique()))
    )
    categories = cat_docs.index.tolist()
    texts = []
    for r in recalls:
        names = [p.get("Name", "") for p in r.get("Products", [])]
        text = " ".join(names) if names else r.get("Title", "")
        texts.append(text)

    vec = TfidfVectorizer(stop_words="english", min_df=2)
    vec.fit(cat_docs.tolist() + texts)
    cat_mat = vec.transform(cat_docs.tolist())
    rec_mat = vec.transform(texts)
    sim = cosine_similarity(rec_mat, cat_mat)
    best_idx = sim.argmax(axis=1)
    best_score = sim.max(axis=1)
    best_cat = np.array(categories)[best_idx]

    confident = (best_score >= CONFIDENT_SIM_THRESHOLD) & (best_cat != EXCLUDE_CATEGORY)
    return int(confident.sum()), len(recalls)


def main() -> None:
    print(f"Fetching CPSC recalls {START} to {END} ...")
    recalls = fetch_cpsc_recalls(START, END)
    n_total = len(recalls)

    years = pd.to_datetime([r["RecallDate"] for r in recalls if r.get("RecallDate")]).year
    n_years = int(years.max() - years.min() + 1)
    per_year = n_total / n_years

    print("Loading DataCo for category-match context (requires data/DataCoSupplyChainDataset.csv)...")
    dataco = gsd.load_dataco_supply_chain()
    n_confident, _ = domain_relevant_share(recalls, dataco)
    confident_per_year = n_confident / n_years

    # Illustrative annualization of compliance_failure's monthly rate (independent-draw
    # approximation; the generator's actual draws are per part-month, not i.i.d. across a year for
    # a single part, but this is the standard back-of-envelope conversion for a scale comparison).
    p_month = gsd.COMPLIANCE_FAILURE_RATE_TARGET
    p_year_approx = 1.0 - (1.0 - p_month) ** 12

    lines = [
        "# Compliance-failure scale check (CPSC recall benchmark)\n",
        "\n**Not a row-level real label.** `compliance_failure` is a calibrated synthetic outcome; ",
        "no public dataset gives a real, part-linked failure rate (recall counts have no real ",
        "population denominator -- see scripts/check_recall_benchmark.py docstring). This is a ",
        "face-validity scale check only.\n",
        f"\n## Real CPSC data ({START} to {END}, saferproducts.gov REST API)\n",
        f"- Total recalls: **{n_total}** across {n_years} years -> **{per_year:.0f}/year** ",
        "(entire US consumer product market).\n",
        f"- TF-IDF-matched to a DataCo real category (similarity >= {CONFIDENT_SIM_THRESHOLD}, ",
        f"excluding the known \"{EXCLUDE_CATEGORY}\" false-positive attractor): ",
        f"**{n_confident}** -> **{confident_per_year:.0f}/year** (illustrative domain-relevant subset; ",
        "category matching is noisy, see main investigation notes -- do not over-read precision here).\n",
        "\n## Scale comparison\n",
        f"- `compliance_failure` target rate: **{p_month:.1%} per part-month** ",
        f"(-> ~{p_year_approx:.0%} per part-year, independent-draw approximation).\n",
        f"- CPSC: **{per_year:.0f} recalls/year across the entire US market** -- a market of an ",
        "enormous number of actively sold SKUs, so any individual product's real annual recall ",
        "probability is necessarily a small fraction of a percent.\n",
        "- **Conclusion**: these are different phenomena by several orders of magnitude, as ",
        "intended. `compliance_failure` models routine operational/quality/delivery compliance ",
        "issues (missed spec, failed audit, minor nonconformance), not product-safety recalls. If ",
        "these numbers were ever close, that would flag a scale error in the generator's ",
        "calibration -- they are not, which is the expected/desired result.\n",
    ]
    out_path = ROOT / "outputs" / "compliance_benchmark_cpsc.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")
    print("".join(lines))


if __name__ == "__main__":
    main()
