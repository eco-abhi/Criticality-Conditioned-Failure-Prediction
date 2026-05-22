#!/usr/bin/env python3
"""Regenerate data + run full modeling with paper-tuned environment defaults."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

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
}


def main() -> None:
    env = os.environ.copy()
    env.update(PAPER_ENV)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")

    print("=== Paper reproduction ===", flush=True)
    print("Environment:", flush=True)
    for k in sorted(PAPER_ENV):
        print(f"  {k}={env[k]}", flush=True)

    subprocess.run(
        ["uv", "run", "python", "generate_synthetic_datasets.py"],
        cwd=REPO,
        env=env,
        check=True,
    )
    subprocess.run(
        ["uv", "run", "--group", "modeling", "python", "scripts/run_modeling_core.py"],
        cwd=REPO,
        env=env,
        check=True,
    )
    print("\nDone. See outputs/modeling/ and REPRODUCE.md", flush=True)


if __name__ == "__main__":
    main()
