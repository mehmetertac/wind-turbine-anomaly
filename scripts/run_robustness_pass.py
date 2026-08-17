#!/usr/bin/env python3
"""Run robustness pass: multi-turbine, seasonal residuals, leakage audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wind_turbine_anomaly.config import DATA_RAW, DEFAULT_BUFFER_DAYS, RESULTS_DIR  # noqa: E402
from wind_turbine_anomaly.eval.robustness import run_robustness_pass  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Robustness pass for physics hybrid")
    parser.add_argument("--raw-dir", type=Path, default=DATA_RAW)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--buffer-days", type=int, default=DEFAULT_BUFFER_DAYS)
    parser.add_argument(
        "--no-seasonal-refit",
        action="store_true",
        help="Do not retry with seasonal terms if seasonal check fails",
    )
    args = parser.parse_args()

    print("Robustness pass — multi-turbine, seasonal, leakage", flush=True)
    result = run_robustness_pass(
        raw_dir=args.raw_dir,
        results_dir=args.results_dir,
        buffer_days=args.buffer_days,
        apply_seasonal_refit=not args.no_seasonal_refit,
    )
    print(f"Overall passed: {result['passed']}", flush=True)
    for name, path in result["paths"].items():
        print(f"  {name}: {path}", flush=True)
    if not result["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
