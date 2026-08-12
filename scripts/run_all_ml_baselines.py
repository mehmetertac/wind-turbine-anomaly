#!/usr/bin/env python3
"""Run all pure-ML baselines, write benchmark CSV, and plot score trajectories."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd

from wind_turbine_anomaly.config import DATA_RAW
from wind_turbine_anomaly.eval.metrics_table import write_metrics_csv
from wind_turbine_anomaly.eval.plots import plot_all_failure_trajectories
from wind_turbine_anomaly.utils import to_utc

DETECTORS = [
    "isolation_forest",
    "dense_autoencoder",
    "lstm_autoencoder",
]

DETECTOR_SCRIPTS = {
    "isolation_forest": PROJECT_ROOT / "scripts" / "run_if_baseline.py",
    "dense_autoencoder": PROJECT_ROOT / "scripts" / "run_dense_ae_baseline.py",
    "lstm_autoencoder": PROJECT_ROOT / "scripts" / "run_lstm_ae_baseline.py",
}


def main() -> int:
    if not DATA_RAW.exists():
        print(f"Raw data directory not found: {DATA_RAW}")
        print("Run: python scripts/generate_synthetic_edp.py --force")
        return 1

    for detector in DETECTORS:
        print(f"\n=== {detector} ===", flush=True)
        script = DETECTOR_SCRIPTS[detector]
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=PROJECT_ROOT,
            check=False,
        )
        if result.returncode != 0:
            print(f"Detector run failed: {detector} (exit {result.returncode})")
            return result.returncode

    write_metrics_csv(DETECTORS)

    failure_turbines = {
        "T01": to_utc(pd.Timestamp("2016-07-18T02:10:00+00:00")),
        "T06": to_utc(pd.Timestamp("2017-10-17T08:38:00+00:00")),
    }
    plot_all_failure_trajectories(failure_turbines, DETECTORS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
