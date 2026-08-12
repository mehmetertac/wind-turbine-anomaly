"""Consolidated benchmark metrics table for pure-ML detectors."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from wind_turbine_anomaly.config import DEFAULT_HORIZON_DAYS, RESULTS_DIR

FAILURE_TURBINES = ("T01", "T06")


def load_detector_metrics(detector: str, results_dir: Path = RESULTS_DIR) -> dict:
    """Load metrics.json for one detector."""
    path = results_dir / detector / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics for detector '{detector}': {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_wide_metrics_table(
    detectors: list[str],
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    results_dir: Path = RESULTS_DIR,
) -> pd.DataFrame:
    """
    Build wide-format benchmark table: one row per detector.

    Columns include per-failure lead times, successful-warning flags, and
    aggregate false_alarms_per_turbine_year.
    """
    rows: list[dict] = []
    for detector in detectors:
        metrics = load_detector_metrics(detector, results_dir)
        by_turbine = {t["turbine_id"]: t for t in metrics.get("turbines", [])}
        row: dict = {
            "detector": detector,
            "false_alarms_per_turbine_year": metrics.get(
                "false_alarms_per_turbine_year"
            ),
        }
        for tid in FAILURE_TURBINES:
            t = by_turbine.get(tid, {})
            row[f"{tid}_lead_time_days"] = t.get("lead_time_days")
            row[f"{tid}_successful_warning_{horizon_days}d"] = t.get(
                "successful_warning"
            )
        rows.append(row)
    return pd.DataFrame(rows)


def build_long_metrics_table(
    detectors: list[str],
    results_dir: Path = RESULTS_DIR,
) -> pd.DataFrame:
    """Build long-format table: one row per detector × turbine."""
    rows: list[dict] = []
    for detector in detectors:
        metrics = load_detector_metrics(detector, results_dir)
        for t in metrics.get("turbines", []):
            rows.append(
                {
                    "detector": detector,
                    "turbine_id": t["turbine_id"],
                    "has_gearbox_failure": t.get("has_gearbox_failure"),
                    "lead_time_days": t.get("lead_time_days"),
                    "successful_warning": t.get("successful_warning"),
                    "false_alarm_episodes": t.get("false_alarm_episodes"),
                    "scored_days": t.get("scored_days"),
                    "precision_at_horizon": t.get("precision_at_horizon"),
                }
            )
    return pd.DataFrame(rows)


def write_metrics_csv(
    detectors: list[str],
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    results_dir: Path = RESULTS_DIR,
) -> tuple[Path, Path]:
    """Write results/metrics.csv and results/metrics_by_turbine.csv."""
    wide = build_wide_metrics_table(detectors, horizon_days, results_dir)
    long = build_long_metrics_table(detectors, results_dir)

    wide_path = results_dir / "metrics.csv"
    long_path = results_dir / "metrics_by_turbine.csv"
    results_dir.mkdir(parents=True, exist_ok=True)
    wide.to_csv(wide_path, index=False)
    long.to_csv(long_path, index=False)
    print(f"Benchmark table written to {wide_path}")
    print(f"Per-turbine table written to {long_path}")
    return wide_path, long_path
