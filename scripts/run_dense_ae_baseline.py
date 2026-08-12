#!/usr/bin/env python3
"""Run dense autoencoder baseline on EDP data and export metrics."""

from __future__ import annotations

import torch  # noqa: F401 — import before sklearn (Windows DLL order)

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wind_turbine_anomaly.config import DATA_RAW
from wind_turbine_anomaly.eval.baseline_runner import run_detector_baseline
from wind_turbine_anomaly.models.dense_autoencoder import fit_dense_autoencoder


def main() -> int:
    if not DATA_RAW.exists():
        print(f"Raw data directory not found: {DATA_RAW}")
        print("Run: python scripts/download_edp.py --instructions")
        return 1

    try:
        run_detector_baseline("dense_autoencoder", fit_dense_autoencoder)
    except FileNotFoundError as exc:
        print(exc)
        print("Run: python scripts/download_edp.py --check")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
