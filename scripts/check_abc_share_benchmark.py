#!/usr/bin/env python3
"""
Real revenue-concentration anchor for the ABC criticality class-size convention
(A_SHARE/B_SHARE/C_SHARE = 0.20/0.30/0.50).

No public dataset gives real, part-level criticality ground truth (spare-parts criticality
classifications are proprietary business judgments -- investigated early in this project, see
project history; only methodology papers are public, no raw labeled datasets). So unlike the BOM
and compliance-outcome checks, this cannot validate *which* specific parts are A/B/C.

What CAN be checked with data already in hand (no new external source needed): whether the
underlying PREMISE of ABC analysis -- a small share of items disproportionately drives value --
actually holds in the real DataCo/UCI backbones, and whether the chosen 20/30/50 class-size split
is a reasonable reflection of that real concentration, rather than an arbitrary convention.

IMPORTANT CAVEAT: this validates the class-SIZE convention, not the label-assignment mechanism.
The frozen paper config uses SYNTHETIC_GENERATOR_MODE=latent, which deliberately assigns labels
from an independent latent score (NOT from price/revenue directly) specifically to avoid a
trivial price->label->feature chain that would make Layer 1 classification circular. This check
is most directly relevant to why a 20/30/50-style skewed split is a reasonable choice at all, not
a validation of the latent-score mechanism itself.

Run: uv run python scripts/check_abc_share_benchmark.py
Writes: outputs/abc_share_benchmark.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import generate_synthetic_datasets as gsd  # noqa: E402

SHARES_TO_CHECK = (0.10, 0.20, 0.30, 0.50)


def concentration_stats(values: np.ndarray) -> dict:
    v = np.asarray(values, dtype=float)
    v = v[v > 0]
    desc = np.sort(v)[::-1]
    n = len(desc)
    total = float(desc.sum())
    cum_share_desc = np.cumsum(desc) / total
    frac_items = np.arange(1, n + 1) / n

    out: dict = {"n_items": n}
    for f in SHARES_TO_CHECK:
        idx = min(int(np.searchsorted(frac_items, f)), n - 1)
        out[f"top_{f:.0%}_items_pct_of_value"] = float(cum_share_desc[idx])

    asc = np.sort(v)
    cum = np.insert(np.cumsum(asc), 0, 0)
    lorenz = cum / cum[-1]
    x = np.linspace(0, 1, len(lorenz))
    out["gini"] = float(1 - 2 * np.trapezoid(lorenz, x))
    return out


def main() -> None:
    uci = gsd.load_uci_online_retail()
    uci["revenue"] = uci["quantity"] * uci["unit_price"]
    uci_stats = concentration_stats(uci.groupby("stock_code")["revenue"].sum().to_numpy())

    dataco = gsd.load_dataco_supply_chain()
    dataco_stats = concentration_stats(dataco.groupby("product_name")["sales"].sum().to_numpy())

    a, b, c = gsd.A_SHARE, gsd.B_SHARE, gsd.C_SHARE

    lines = [
        "# ABC class-size convention scale check (real revenue concentration)\n",
        "\n**Validates the class-SIZE convention, not the label-assignment mechanism.** No public "
        "dataset gives real part-level criticality ground truth (proprietary business judgments; "
        "only methodology papers are public). This checks whether the ABC premise -- a small "
        "share of items disproportionately drives value -- holds in the real DataCo/UCI backbones "
        "already in hand, as context for why a 20/30/50-style skewed class-size split is a "
        "reasonable convention. It does NOT validate `SYNTHETIC_GENERATOR_MODE=latent`'s "
        "label-assignment mechanism, which deliberately decouples labels from price/revenue.\n",
        f"\n## Generator convention\nA_SHARE={a:.0%}, B_SHARE={b:.0%}, C_SHARE={c:.0%}\n",
        "\n## Real revenue concentration\n",
        f"\n### UCI Online Retail (n={uci_stats['n_items']} real SKUs, real per-SKU revenue)\n",
    ]
    for f in SHARES_TO_CHECK:
        lines.append(f"- top {f:.0%} of SKUs by real revenue capture **{uci_stats[f'top_{f:.0%}_items_pct_of_value']:.1%}** of real total revenue\n")
    lines.append(f"- Gini coefficient: **{uci_stats['gini']:.3f}**\n")

    lines.append(f"\n### DataCo (n={dataco_stats['n_items']} real products, real per-product sales)\n")
    for f in SHARES_TO_CHECK:
        lines.append(f"- top {f:.0%} of products by real sales capture **{dataco_stats[f'top_{f:.0%}_items_pct_of_value']:.1%}** of real total sales\n")
    lines.append(f"- Gini coefficient: **{dataco_stats['gini']:.3f}**\n")

    lines.append(
        "\n## Conclusion\n"
        f"In real UCI data, the top 20% of SKUs by revenue capture "
        f"{uci_stats['top_20%_items_pct_of_value']:.1%} of total revenue -- close to the classic "
        "80/20 Pareto pattern the ABC convention is named after. DataCo's real product-level "
        f"concentration is sharper still ({dataco_stats['top_20%_items_pct_of_value']:.1%} at the "
        "top 20%, on a much smaller n=118 product catalog). Both confirm the ABC premise (a "
        "minority of items disproportionately drives value) holds in the real backbone data, "
        "supporting the choice of a skewed 20/30/50 class-size convention over an arbitrary or "
        "uniform one. This is real-data context for the convention, not a validation of which "
        "specific parts the generator assigns to each class.\n"
    )

    out_path = ROOT / "outputs" / "abc_share_benchmark.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")
    print("".join(lines))


if __name__ == "__main__":
    main()
