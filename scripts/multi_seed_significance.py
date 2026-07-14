#!/usr/bin/env python3
"""
Repeated-run significance testing across independent seeds (train/test split + GAT training +
OOF fold assignment all vary per seed) -- addresses reviewer feedback that a single-run point
estimate (e.g. "0.732 vs 0.731") is not publishable without confidence intervals across
multiple runs/seeds, distinct from the existing bootstrap CIs which resample one run's test set.

Scope limitation, disclosed rather than hidden: LightGBM's own internal `random_state` (fixed at
42 in LGBM_MULTICLASS_PARAMS / LGBM_BINARY_PARAMS) is NOT varied across seeds here -- only the
train/test split, GAT initialization/training, and OOF fold assignment vary. This captures the two
dominant variance sources (data partition, GAT stochasticity) without threading a seed through
every LGBMClassifier construction site in modeling_lib.py. Total run-to-run variance is likely
modestly understated as a result.

Each run is a full, independent subprocess invocation of scripts/run_modeling_core.py with a
different RUN_SEED -- not bootstrap resampling. Use --fast for a feasible-in-session run (reduced
GAT epochs); omit it for full-fidelity results before final submission (much slower -- budget
~30-90 min PER SEED, so N seeds needs N x that; run in background).

Run: uv run --group modeling python scripts/multi_seed_significance.py --n-seeds 20 --fast
Writes: outputs/multi_seed/summary.csv, outputs/multi_seed/significance.csv
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parent.parent

PAPER_ENV = {
    "SYNTHETIC_GENERATOR_MODE": "latent",
    "LATENT_TO_LABEL_NOISE": "0.45",
    "LATENT_TO_FEATURE_NOISE": "1.05",
    "LAYER1_FEATURES": "clean",
    "COMPLIANCE_GRAIN": "part_month",
    "CRIT_PROB_SHARPEN": "0.88",
    "L2_NUM_LEAVES": "127",
    "N_PARTS": "3500",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}

METRICS = ["auc_roc", "auc_pr", "f1", "precision", "recall", "brier"]
MODELS = ["uniform", "conditioned", "oracle"]


def run_one_seed(seed: int, out_dir: Path, scope: str, fast: bool) -> dict:
    env = os.environ.copy()
    env.update(PAPER_ENV)
    env["RUN_SEED"] = str(seed)
    env["LAYER2_SCOPE"] = scope
    cmd = [
        "uv", "run", "--group", "modeling", "python", "scripts/run_modeling_core.py",
        "--out-dir", str(out_dir),
    ]
    if fast:
        cmd.append("--fast")
    r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"seed {seed} failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")

    row: dict = {"seed": seed}
    for model, fname in [
        ("uniform", "layer2_baseline_uniform.json"),
        ("conditioned", "layer2_full_conditioned.json"),
        ("oracle", "layer2_oracle_true_class.json"),
    ]:
        d = json.loads((out_dir / fname).read_text(encoding="utf-8"))
        for m in METRICS:
            row[f"{model}_{m}"] = d["overall"][m]
    return row


def paired_t_and_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 5000, seed: int = 0) -> dict:
    """Paired t-test (as the reviewer explicitly suggested) plus a percentile bootstrap CI on the
    mean cross-run difference, treating each seed's result as one independent observation."""
    diff = b - a
    t_stat, p_t = stats.ttest_rel(b, a)
    rng = np.random.default_rng(seed)
    n = len(diff)
    boot_means = np.array([rng.choice(diff, size=n, replace=True).mean() for _ in range(n_boot)])
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
    return {
        "mean_diff": float(diff.mean()),
        "std_diff": float(diff.std(ddof=1)) if n > 1 else float("nan"),
        "t_stat": float(t_stat),
        "p_value_paired_t": float(p_t),
        "ci95_low": float(ci_lo),
        "ci95_high": float(ci_hi),
        "n_seeds": int(n),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-seeds", type=int, default=20)
    ap.add_argument("--base-seed", type=int, default=100)
    ap.add_argument("--scope", default="real_category_only", choices=("real_category_only", "all"))
    ap.add_argument("--fast", action="store_true", help="Reduced GAT epochs -- feasible in-session, not final-submission fidelity.")
    ap.add_argument("--out-dir", type=Path, default=REPO / "outputs" / "multi_seed")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [args.base_seed + i for i in range(args.n_seeds)]

    rows = []
    for i, seed in enumerate(seeds):
        print(f"[{i + 1}/{len(seeds)}] seed={seed} ...", flush=True)
        run_out = args.out_dir / f"seed_{seed}"
        row = run_one_seed(seed, run_out, args.scope, args.fast)
        rows.append(row)
        print(f"  uniform_auc_pr={row['uniform_auc_pr']:.4f} conditioned_auc_pr={row['conditioned_auc_pr']:.4f} "
              f"oracle_auc_pr={row['oracle_auc_pr']:.4f}", flush=True)

    summary = pd.DataFrame(rows)
    summary.to_csv(args.out_dir / "summary.csv", index=False)

    sig_rows = []
    for b_name, a_name in [("conditioned", "uniform"), ("oracle", "uniform"), ("oracle", "conditioned")]:
        for m in METRICS:
            a = summary[f"{a_name}_{m}"].to_numpy(dtype=float)
            b = summary[f"{b_name}_{m}"].to_numpy(dtype=float)
            res = paired_t_and_bootstrap(a, b, seed=0)
            sig_rows.append({"comparison": f"{b_name}_vs_{a_name}", "metric": m, **res})
    sig_df = pd.DataFrame(sig_rows)

    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    import modeling_lib as ml  # noqa: E402

    sig_df = ml.add_multiple_comparison_corrections(sig_df, p_col="p_value_paired_t")
    sig_df.to_csv(args.out_dir / "significance.csv", index=False)

    print(f"\nWrote {args.out_dir / 'summary.csv'} ({len(summary)} seeds)")
    print(f"Wrote {args.out_dir / 'significance.csv'}")
    pd.set_option("display.width", 160)
    print(
        sig_df[
            ["comparison", "metric", "mean_diff", "ci95_low", "ci95_high", "p_value_paired_t", "p_value_fdr_bh"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
