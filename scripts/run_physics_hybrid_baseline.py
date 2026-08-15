#!/usr/bin/env python3
"""Run physics-residual hybrid baseline on EDP data and export metrics."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wind_turbine_anomaly.config import DATA_RAW, PHYSICS_HYBRID_DETECTOR, THERMAL_MIN_POWER_KW
from wind_turbine_anomaly.eval.baseline_runner import run_detector_baseline
from wind_turbine_anomaly.models.physics_hybrid import fit_physics_hybrid


def main() -> int:
    if not DATA_RAW.exists():
        print(f"Raw data directory not found: {DATA_RAW}")
        print("Run: python scripts/generate_synthetic_edp.py --force")
        return 1

    try:
        run_detector_baseline(
            PHYSICS_HYBRID_DETECTOR,
            fit_physics_hybrid,
            min_power_kw=THERMAL_MIN_POWER_KW,
        )
    except FileNotFoundError as exc:
        print(exc)
        print("Run: python scripts/download_edp.py --check")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
