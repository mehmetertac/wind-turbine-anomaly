#!/usr/bin/env python3
"""Sweep anomaly-score thresholds and write lead-time vs false-alarm trade-off."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wind_turbine_anomaly.config import DATA_RAW, RESULTS_DIR
from wind_turbine_anomaly.eval.plots import plot_threshold_tradeoff
from wind_turbine_anomaly.eval.threshold_sweep import (
    pick_best_pure_ml_detector,
    write_threshold_sweep,
)


def main() -> int:
    if not DATA_RAW.exists():
        print(f"Raw data directory not found: {DATA_RAW}")
        print("Run: python scripts/generate_synthetic_edp.py --force")
        return 1

    hybrid_dir = RESULTS_DIR / "physics_hybrid"
    if not hybrid_dir.exists() or not any(hybrid_dir.glob("*_train_scores.parquet")):
        print("Missing physics_hybrid train scores. Run scripts/run_all_ml_baselines.py first.")
        return 1

    _, _, sweep_df = write_threshold_sweep(raw_dir=DATA_RAW, results_dir=RESULTS_DIR)
    best_ml = pick_best_pure_ml_detector(results_dir=RESULTS_DIR)
    plot_threshold_tradeoff(
        sweep_df,
        best_pure_ml_detector=best_ml,
        results_dir=RESULTS_DIR,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
