#!/usr/bin/env python3
"""
Tune latent noise + CRIT_PROB_SHARPEN so Layer 2 conditioned AUC-PR beats uniform.

Phase 1: grid over (LLN, LFN) with sharpen=1.0 (fast modeling).
Phase 2: on best latent cell, sweep sharpen with --skip-layer1 (reuses layer1_bundle.pkl).
Phase 3: regenerate data + full notebook execute with best config (unless --fast-only).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "outputs"
OUT_MODELING = OUT / "modeling"
OUT_TUNE = OUT / "tune"

_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def _run(cmd: list[str], env: dict[str, str]) -> None:
    r = subprocess.run(cmd, cwd=REPO, env=env, check=False)
    if r.returncode != 0:
        raise SystemExit(f"failed ({r.returncode}): {' '.join(cmd)}")


def _read_metrics() -> dict:
    cmp = {r["model"]: r for r in csv.DictReader((OUT_MODELING / "compliance_comparison.csv").open())}
    l1_path = OUT_MODELING / "layer1_results.json"
    l1 = json.loads(l1_path.read_text(encoding="utf-8")) if l1_path.exists() else {}
    return {
        "layer1_f1_macro": float(l1.get("f1_macro", float("nan"))),
        "uniform_auc_pr": float(cmp["uniform"]["auc_pr"]),
        "conditioned_auc_pr": float(cmp["conditioned"]["auc_pr"]),
        "oracle_auc_pr": float(cmp["oracle"]["auc_pr"]),
        "delta_auc_pr_cond_minus_uniform": float(cmp["conditioned"]["auc_pr"])
        - float(cmp["uniform"]["auc_pr"]),
        "uniform_brier": float(cmp["uniform"]["brier"]),
        "conditioned_brier": float(cmp["conditioned"]["brier"]),
    }


def _score(row: dict) -> tuple:
    return (
        float(row["delta_auc_pr_cond_minus_uniform"]),
        float(row["conditioned_auc_pr"]),
        float(row.get("layer1_f1_macro", 0.0)),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-parts", type=int, default=3500)
    ap.add_argument("--lln-values", type=float, nargs="+", default=[0.45, 0.55, 0.65])
    ap.add_argument("--lfn-values", type=float, nargs="+", default=[0.85, 1.0, 1.1])
    ap.add_argument(
        "--sharpen-values",
        type=float,
        nargs="+",
        default=[0.75, 0.82, 0.88, 0.94, 1.0],
    )
    ap.add_argument("--fast-only", action="store_true")
    ap.add_argument("--skip-phase3", action="store_true")
    args = ap.parse_args()

    OUT_TUNE.mkdir(parents=True, exist_ok=True)
    base = os.environ.copy()
    base.update(_THREAD_ENV)
    base["LAYER1_FEATURES"] = "clean"
    base["N_PARTS"] = str(args.n_parts)
    base["SYNTHETIC_GENERATOR_MODE"] = "latent"
    base["L2_NUM_LEAVES"] = "63"
    base["L2_LEARNING_RATE"] = "0.0143"
    base["LAYER2_SCOPE"] = "real_category_only"

    rows: list[dict] = []

    print("=== Phase 1: latent grid (sharpen=1.0) ===", flush=True)
    for lln in args.lln_values:
        for lfn in args.lfn_values:
            env = base.copy()
            env["LATENT_TO_LABEL_NOISE"] = str(lln)
            env["LATENT_TO_FEATURE_NOISE"] = str(lfn)
            env["CRIT_PROB_SHARPEN"] = "1.0"
            tag = f"lln{lln:g}_lfn{lfn:g}_sh1"
            print(f"\n--- {tag} ---", flush=True)
            _run(
                ["uv", "run", "python", "generate_synthetic_datasets.py", "--n-parts", str(args.n_parts)],
                env,
            )
            _run(
                ["uv", "run", "--group", "modeling", "python", "scripts/run_modeling_core.py", "--fast"],
                env,
            )
            m = _read_metrics()
            row = {"phase": 1, "lln": lln, "lfn": lfn, "crit_prob_sharpen": 1.0, "n_parts": args.n_parts, **m}
            rows.append(row)
            (OUT_TUNE / f"metrics_{tag}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
            print(json.dumps(row, indent=2), flush=True)

    positive = [r for r in rows if r["delta_auc_pr_cond_minus_uniform"] > 0]
    best_latent = max(positive, key=_score) if positive else max(rows, key=_score)
    print("\nBest latent cell:", json.dumps(best_latent, indent=2), flush=True)

    print("\n=== Phase 2: sharpen sweep on best latent (skip Layer 1 retrain) ===", flush=True)
    env_base = base.copy()
    env_base["LATENT_TO_LABEL_NOISE"] = str(best_latent["lln"])
    env_base["LATENT_TO_FEATURE_NOISE"] = str(best_latent["lfn"])
    _run(
        ["uv", "run", "python", "generate_synthetic_datasets.py", "--n-parts", str(args.n_parts)],
        env_base,
    )
    _run(
        ["uv", "run", "--group", "modeling", "python", "scripts/run_modeling_core.py", "--fast"],
        env_base,
    )

    sharpen_rows: list[dict] = []
    for sharpen in args.sharpen_values:
        env = env_base.copy()
        env["CRIT_PROB_SHARPEN"] = str(sharpen)
        tag = f"sh{sharpen:g}"
        print(f"\n--- sharpen {sharpen} ---", flush=True)
        _run(
            [
                "uv",
                "run",
                "--group",
                "modeling",
                "python",
                "scripts/run_modeling_core.py",
                "--fast",
                "--skip-layer1",
            ],
            env,
        )
        m = _read_metrics()
        row = {
            "phase": 2,
            "lln": best_latent["lln"],
            "lfn": best_latent["lfn"],
            "crit_prob_sharpen": sharpen,
            "n_parts": args.n_parts,
            **m,
        }
        sharpen_rows.append(row)
        rows.append(row)
        (OUT_TUNE / f"metrics_{tag}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        print(json.dumps(row, indent=2), flush=True)

    all_positive = [r for r in rows if r["delta_auc_pr_cond_minus_uniform"] > 0]
    best = max(all_positive, key=_score) if all_positive else max(rows, key=_score)
    (OUT_TUNE / "best_config.json").write_text(json.dumps(best, indent=2), encoding="utf-8")

    summary_path = OUT_TUNE / "tune_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {summary_path}")
    print("Best overall:", json.dumps(best, indent=2))

    if args.fast_only or args.skip_phase3:
        return

    print("\n=== Phase 3: apply best config (regen + full Layer 1/2) ===", flush=True)
    env = base.copy()
    env["LATENT_TO_LABEL_NOISE"] = str(best["lln"])
    env["LATENT_TO_FEATURE_NOISE"] = str(best["lfn"])
    env["CRIT_PROB_SHARPEN"] = str(best["crit_prob_sharpen"])
    _run(
        ["uv", "run", "python", "generate_synthetic_datasets.py", "--n-parts", str(args.n_parts)],
        env,
    )
    _run(
        ["uv", "run", "--group", "modeling", "python", "scripts/run_modeling_core.py"],
        env,
    )
    print(
        "\nOptional: run full notebook for SHAP/plots:\n"
        f"  CRIT_PROB_SHARPEN={best['crit_prob_sharpen']} LATENT_TO_LABEL_NOISE={best['lln']} "
        f"LATENT_TO_FEATURE_NOISE={best['lfn']} LAYER1_FEATURES=clean N_PARTS={args.n_parts} "
        "uv run --group dev --group modeling jupyter nbconvert --execute modeling.ipynb --inplace",
        flush=True,
    )


if __name__ == "__main__":
    main()
