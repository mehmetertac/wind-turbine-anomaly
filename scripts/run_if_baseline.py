#!/usr/bin/env python3
"""Run Isolation Forest baseline on EDP data and export metrics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wind_turbine_anomaly.config import (
    DATA_RAW,
    DEFAULT_BUFFER_DAYS,
    DEFAULT_CONTAMINATION,
    DEFAULT_HORIZON_DAYS,
    DEFAULT_THRESHOLD_PERCENTILE,
    FEATURE_COLUMNS,
    RESULTS_DIR,
)
from wind_turbine_anomaly.data.clean import (
    clean_turbine_df,
    get_failure_for_turbine,
    healthy_training_mask,
)
from wind_turbine_anomaly.data.load_edp import load_edp_dataset
from wind_turbine_anomaly.eval.protocol import evaluate_turbine, results_to_dict
from wind_turbine_anomaly.models.isolation_forest import (
    fit_isolation_forest,
    save_scores,
    threshold_from_training,
)


def run_baseline(
    raw_dir: Path = DATA_RAW,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    buffer_days: int = DEFAULT_BUFFER_DAYS,
) -> dict:
    turbines, failures = load_edp_dataset(raw_dir)
    out_dir = RESULTS_DIR / "isolation_forest"
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_results = []
    for turbine_id, raw_df in sorted(turbines.items()):
        df = clean_turbine_df(raw_df)
        failure = get_failure_for_turbine(turbine_id, failures)
        train_mask = healthy_training_mask(df.index, failure, buffer_days)
        train_df = df.loc[train_mask]

        if len(train_df) < 100:
            print(f"  SKIP {turbine_id}: insufficient training rows ({len(train_df)})")
            continue

        pipeline = fit_isolation_forest(
            train_df,
            FEATURE_COLUMNS,
            contamination=DEFAULT_CONTAMINATION,
        )
        train_scores = pipeline.score(train_df)
        threshold = threshold_from_training(
            train_scores, percentile=DEFAULT_THRESHOLD_PERCENTILE
        )

        score_df = df.loc[~train_mask]
        if score_df.empty:
            score_df = df
        scores = pipeline.score(score_df)
        save_scores(scores, out_dir / f"{turbine_id}_scores.parquet")

        score_start = train_df.index[-1] if not train_df.empty else None
        result = evaluate_turbine(
            turbine_id,
            scores,
            threshold=threshold,
            failure=failure,
            horizon_days=horizon_days,
            score_start=score_start,
        )
        eval_results.append(result)
        print(
            f"  {turbine_id}: lead_time={result.lead_time_days}, "
            f"false_alarms={result.false_alarm_episodes}, threshold={threshold:.4f}"
        )

    metrics = results_to_dict(eval_results, horizon_days)
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Metrics written to {metrics_path}")
    return metrics


def main() -> int:
    if not DATA_RAW.exists():
        print(f"Raw data directory not found: {DATA_RAW}")
        print("Run: python scripts/download_edp.py --instructions")
        return 1

    try:
        run_baseline()
    except FileNotFoundError as exc:
        print(exc)
        print("Run: python scripts/download_edp.py --check")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
