"""Unified baseline runner for pure-ML anomaly detectors."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from wind_turbine_anomaly.config import (
    DATA_RAW,
    DEFAULT_BUFFER_DAYS,
    DEFAULT_HORIZON_DAYS,
    DEFAULT_THRESHOLD_PERCENTILE,
    FEATURE_COLUMNS,
    MIN_POWER_KW,
    RESULTS_DIR,
)
from wind_turbine_anomaly.data.clean import (
    clean_turbine_df,
    get_failure_for_turbine,
    healthy_training_mask,
)
from wind_turbine_anomaly.data.load_edp import load_edp_dataset
from wind_turbine_anomaly.eval.protocol import TurbineEvalResult, evaluate_turbine, results_to_dict
from wind_turbine_anomaly.models.scoring import save_scores, threshold_from_training


class AnomalyPipeline(Protocol):
    """Minimal interface for per-turbine anomaly scoring pipelines."""

    def score(self, X: pd.DataFrame) -> pd.Series:
        ...


FitFn = Callable[[pd.DataFrame, list[str]], AnomalyPipeline]


def run_detector_baseline(
    detector_name: str,
    fit_fn: FitFn,
    raw_dir: Path = DATA_RAW,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    buffer_days: int = DEFAULT_BUFFER_DAYS,
    feature_columns: list[str] | None = None,
    threshold_percentile: float = DEFAULT_THRESHOLD_PERCENTILE,
    min_power_kw: float = MIN_POWER_KW,
) -> dict[str, Any]:
    """
    Run rolling-origin baseline for one detector across all turbines.

    Writes per-turbine score parquets and metrics.json under
    results/{detector_name}/.
    """
    feature_columns = feature_columns or FEATURE_COLUMNS
    turbines, failures = load_edp_dataset(raw_dir)
    out_dir = RESULTS_DIR / detector_name
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_results: list[TurbineEvalResult] = []
    thresholds: dict[str, float] = {}

    for turbine_id, raw_df in sorted(turbines.items()):
        df = clean_turbine_df(raw_df, min_power_kw=min_power_kw)
        failure = get_failure_for_turbine(turbine_id, failures)
        train_mask = healthy_training_mask(df.index, failure, buffer_days)
        train_df = df.loc[train_mask]

        if len(train_df) < 100:
            print(f"  SKIP {turbine_id}: insufficient training rows ({len(train_df)})", flush=True)
            continue

        print(f"  {turbine_id}: fitting...", flush=True)
        pipeline = fit_fn(train_df, feature_columns)
        train_scores = pipeline.score(train_df)
        threshold = threshold_from_training(
            train_scores, percentile=threshold_percentile
        )
        thresholds[turbine_id] = threshold

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
            f"false_alarms={result.false_alarm_episodes}, threshold={threshold:.6f}"
        )

    metrics = results_to_dict(eval_results, horizon_days)
    metrics["detector"] = detector_name
    metrics["thresholds"] = thresholds
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Metrics written to {metrics_path}")
    return metrics
