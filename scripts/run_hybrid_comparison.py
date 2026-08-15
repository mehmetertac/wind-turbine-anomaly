#!/usr/bin/env python3
"""Run head-to-head hybrid vs pure ML comparison and write analysis artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wind_turbine_anomaly.config import DATA_RAW, PHYSICS_HYBRID_DETECTOR, RESULTS_DIR
from wind_turbine_anomaly.eval.hybrid_comparison import write_hybrid_comparison
from wind_turbine_anomaly.eval.metrics_table import load_detector_metrics


def main() -> int:
    if not DATA_RAW.exists():
        print(f"Raw data directory not found: {DATA_RAW}")
        print("Run: python scripts/generate_synthetic_edp.py --force")
        return 1

    hybrid_metrics_path = RESULTS_DIR / PHYSICS_HYBRID_DETECTOR / "metrics.json"
    if not hybrid_metrics_path.exists():
        print(f"Missing hybrid metrics: {hybrid_metrics_path}")
        print("Run: python scripts/run_physics_hybrid_baseline.py")
        return 1

    for detector in ("isolation_forest", "dense_autoencoder", "lstm_autoencoder"):
        try:
            load_detector_metrics(detector, RESULTS_DIR)
        except FileNotFoundError:
            print(f"Missing pure-ML metrics for {detector}")
            print("Run: python scripts/run_all_ml_baselines.py")
            return 1

    write_hybrid_comparison(raw_dir=DATA_RAW, results_dir=RESULTS_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
