#!/usr/bin/env python3
"""Thermal model interpretability: SHAP drivers + failure case study."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wind_turbine_anomaly.config import DATA_RAW, DEFAULT_BUFFER_DAYS, RESULTS_DIR  # noqa: E402
from wind_turbine_anomaly.eval.thermal_interpretability import run_thermal_interpretability  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Thermal model interpretability")
    parser.add_argument("--raw-dir", type=Path, default=DATA_RAW)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--buffer-days", type=int, default=DEFAULT_BUFFER_DAYS)
    parser.add_argument(
        "--case-study-turbine",
        default="T06",
        help="Turbine for end-to-end failure case study (default T06)",
    )
    args = parser.parse_args()

    print("Thermal interpretability — SHAP + case study", flush=True)
    summary = run_thermal_interpretability(
        raw_dir=args.raw_dir,
        results_dir=args.results_dir,
        buffer_days=args.buffer_days,
        case_study_turbine=args.case_study_turbine,
    )
    print(f"Summary: {summary.get('summary_path')}", flush=True)
    if "plot" in summary.get("case_study", {}):
        print(f"Case study: {summary['case_study']['plot']}", flush=True)


if __name__ == "__main__":
    main()
