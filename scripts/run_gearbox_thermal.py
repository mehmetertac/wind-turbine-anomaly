#!/usr/bin/env python3
"""Fit gearbox thermal normal-behavior models and emit residual signals."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wind_turbine_anomaly.config import (  # noqa: E402
    DATA_RAW,
    DEFAULT_BUFFER_DAYS,
    POWER_COLUMN,
    RESULTS_DIR,
    THERMAL_MIN_POWER_KW,
    THERMAL_RESULTS_DIR,
    THERMAL_TARGET_COLUMNS,
)
from wind_turbine_anomaly.data.clean import (  # noqa: E402
    clean_turbine_df,
    get_failure_for_turbine,
    healthy_training_mask,
)
from wind_turbine_anomaly.data.load_edp import load_edp_dataset  # noqa: E402
from wind_turbine_anomaly.eval.plots import plot_gearbox_residual_trajectories  # noqa: E402
from wind_turbine_anomaly.models.gearbox_thermal import (  # noqa: E402
    fit_gearbox_thermal_with_selection,
    validate_thermal_model,
)

TARGET_SLUGS = {
    "Gear_Oil_Temp_Avg": "oil",
    "Gear_Bear_Temp_Avg": "bear",
}


def _target_slug(target_col: str) -> str:
    return TARGET_SLUGS.get(target_col, target_col.lower())


def run_gearbox_thermal(
    raw_dir: Path = DATA_RAW,
    buffer_days: int = DEFAULT_BUFFER_DAYS,
    lookback_days: int = 90,
) -> dict:
    """Fit per-turbine thermal models, validate, and write residuals + plots."""
    turbines, failures = load_edp_dataset(raw_dir)
    out_dir = THERMAL_RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = RESULTS_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {"turbines": {}, "plots": []}

    for turbine_id, raw_df in sorted(turbines.items()):
        print(f"  {turbine_id}: processing...", flush=True)
        df = clean_turbine_df(raw_df, min_power_kw=THERMAL_MIN_POWER_KW)
        failure = get_failure_for_turbine(turbine_id, failures)
        healthy_mask = healthy_training_mask(df.index, failure, buffer_days)
        healthy_df = df.loc[healthy_mask]

        if len(healthy_df) < 100:
            print(f"  SKIP {turbine_id}: insufficient healthy rows ({len(healthy_df)})", flush=True)
            continue

        turbine_summary: dict = {"targets": {}}

        for target_col in THERMAL_TARGET_COLUMNS:
            model, train_df, val_df, selection = fit_gearbox_thermal_with_selection(
                healthy_df, target_col
            )
            validation = validate_thermal_model(model, val_df)
            validation["selection"] = selection

            if not validation.get("passed", False):
                print(
                    f"    WARN {turbine_id} {target_col}: validation checks not all passed",
                    flush=True,
                )

            slug = _target_slug(target_col)
            val_path = out_dir / f"{turbine_id}_{slug}_validation.json"
            val_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")

            residual_frame = model.residual_frame(df)
            parquet_path = out_dir / f"{turbine_id}_{slug}_residuals.parquet"
            residual_frame.to_parquet(parquet_path)

            label = "oil temp" if "Oil" in target_col else "bear temp"
            if failure is not None:
                plot_path = plot_gearbox_residual_trajectories(
                    turbine_id,
                    label,
                    failure.timestamp,
                    residual_frame,
                    power_series=df[POWER_COLUMN],
                    lookback_days=lookback_days,
                    out_path=plots_dir / f"residual_{turbine_id}_{slug}.png",
                )
                summary["plots"].append(str(plot_path))
                print(f"    Plot: {plot_path}", flush=True)

            turbine_summary["targets"][target_col] = {
                "model": selection["chosen_model"],
                "validation_passed": validation.get("passed", False),
                "val_rmse": validation.get("rmse"),
                "parquet": str(parquet_path),
            }
            print(
                f"    {target_col}: model={selection['chosen_model']}, "
                f"val_rmse={validation.get('rmse', float('nan')):.3f}, "
                f"passed={validation.get('passed')}",
                flush=True,
            )

        summary["turbines"][turbine_id] = turbine_summary

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary written to {summary_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Gearbox thermal normal-behavior model (Day 1)")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DATA_RAW,
        help="Directory containing EDP CSV files",
    )
    parser.add_argument(
        "--buffer-days",
        type=int,
        default=DEFAULT_BUFFER_DAYS,
        help="Healthy training buffer before failure (days)",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=90,
        help="Days before failure to plot",
    )
    args = parser.parse_args()

    print("Gearbox thermal model — Day 1", flush=True)
    run_gearbox_thermal(
        raw_dir=args.raw_dir,
        buffer_days=args.buffer_days,
        lookback_days=args.lookback_days,
    )


if __name__ == "__main__":
    main()
