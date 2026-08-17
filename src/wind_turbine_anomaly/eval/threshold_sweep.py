"""Threshold sweep: lead-time vs false-alarm trade-off and headline claim."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from wind_turbine_anomaly.config import (
    DATA_RAW,
    DEFAULT_BUFFER_DAYS,
    DEFAULT_HORIZON_DAYS,
    DEFAULT_THRESHOLD_PERCENTILE,
    FAILURE_TURBINES,
    PHYSICS_HYBRID_DETECTOR,
    RESULTS_DIR,
    THRESHOLD_SWEEP_PERCENTILES,
)
from wind_turbine_anomaly.data.clean import get_failure_for_turbine
from wind_turbine_anomaly.data.load_edp import load_edp_dataset
from wind_turbine_anomaly.eval.hybrid_comparison import PURE_ML_DETECTORS
from wind_turbine_anomaly.eval.metrics_table import load_detector_metrics
from wind_turbine_anomaly.eval.protocol import (
    TurbineEvalResult,
    evaluate_turbine,
    false_alarms_per_turbine_year,
)
from wind_turbine_anomaly.models.scoring import threshold_from_training


def threshold_at_percentile(train_scores: pd.Series, percentile: float) -> float:
    """Compute alarm threshold from healthy training score distribution."""
    return threshold_from_training(train_scores, percentile=percentile)


def median_lead_time_failure_turbines(
    results: list[TurbineEvalResult],
    failure_turbines: tuple[str, ...] = FAILURE_TURBINES,
) -> tuple[float | None, int]:
    """Median lead time across failure turbines with a qualifying pre-failure alarm."""
    leads: list[float] = []
    for result in results:
        if result.turbine_id not in failure_turbines:
            continue
        if result.lead_time_days is not None:
            leads.append(result.lead_time_days)
    if not leads:
        return None, 0
    return float(np.median(leads)), len(leads)


def _load_turbine_score_data(
    detector: str,
    turbine_id: str,
    results_dir: Path,
) -> tuple[pd.Series, pd.Series, pd.Timestamp | None] | None:
    """Load train scores, post-train scores, and score_start for one turbine."""
    det_dir = results_dir / detector
    train_path = det_dir / f"{turbine_id}_train_scores.parquet"
    score_path = det_dir / f"{turbine_id}_scores.parquet"
    if not train_path.exists() or not score_path.exists():
        return None

    train_scores = pd.read_parquet(train_path).iloc[:, 0]
    train_scores.index = pd.to_datetime(train_scores.index, utc=True)
    scores = pd.read_parquet(score_path).iloc[:, 0]
    scores.index = pd.to_datetime(scores.index, utc=True)
    score_start = train_scores.index[-1] if not train_scores.empty else None
    return train_scores, scores, score_start


def sweep_detector(
    detector: str,
    raw_dir: Path = DATA_RAW,
    results_dir: Path = RESULTS_DIR,
    percentiles: list[float] | None = None,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    buffer_days: int = DEFAULT_BUFFER_DAYS,
) -> pd.DataFrame:
    """
    Sweep training-score percentiles for one detector.

    Returns one row per percentile with aggregate false-alarm rate and median lead time.
    """
    percentiles = percentiles or THRESHOLD_SWEEP_PERCENTILES
    _, failures = load_edp_dataset(raw_dir)
    turbine_ids = sorted(
        p.stem.replace("_train_scores", "")
        for p in (results_dir / detector).glob("*_train_scores.parquet")
    )

    rows: list[dict[str, Any]] = []
    for pct in percentiles:
        eval_results: list[TurbineEvalResult] = []
        thresholds: dict[str, float] = {}

        for turbine_id in turbine_ids:
            loaded = _load_turbine_score_data(detector, turbine_id, results_dir)
            if loaded is None:
                continue
            train_scores, scores, score_start = loaded
            failure = get_failure_for_turbine(turbine_id, failures)
            threshold = threshold_at_percentile(train_scores, pct)
            thresholds[turbine_id] = threshold

            result = evaluate_turbine(
                turbine_id,
                scores,
                threshold=threshold,
                failure=failure,
                horizon_days=horizon_days,
                score_start=score_start,
            )
            eval_results.append(result)

        median_lead, n_with_alarm = median_lead_time_failure_turbines(eval_results)
        by_turbine = {r.turbine_id: r for r in eval_results}
        row: dict[str, Any] = {
            "detector": detector,
            "threshold_percentile": pct,
            "false_alarms_per_turbine_year": false_alarms_per_turbine_year(
                eval_results
            ),
            "median_lead_time_days": median_lead,
            "n_failure_turbines_with_alarm": n_with_alarm,
        }
        for tid in FAILURE_TURBINES:
            t = by_turbine.get(tid)
            row[f"{tid}_lead_time_days"] = t.lead_time_days if t else None
            row[f"{tid}_threshold"] = thresholds.get(tid)
        rows.append(row)

    return pd.DataFrame(rows)


def pick_best_pure_ml_detector(
    results_dir: Path = RESULTS_DIR,
) -> str:
    """Pick pure ML detector with best median lead time at the default operating point."""
    best_detector: str | None = None
    best_median: float = -1.0

    for detector in PURE_ML_DETECTORS:
        metrics_path = results_dir / detector / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = load_detector_metrics(detector, results_dir)
        by_turbine = {t["turbine_id"]: t for t in metrics.get("turbines", [])}
        leads = [
            by_turbine[tid]["lead_time_days"]
            for tid in FAILURE_TURBINES
            if tid in by_turbine and by_turbine[tid].get("lead_time_days") is not None
        ]
        median = float(np.median(leads)) if leads else None

        if median is not None and median > best_median:
            best_median = median
            best_detector = detector

    if best_detector is None:
        raise FileNotFoundError(
            "No pure-ML baseline metrics found. Run scripts/run_all_ml_baselines.py first."
        )
    return best_detector


def _detector_claim_row(sweep_row: pd.Series) -> dict[str, Any]:
    """Extract headline fields from one sweep row."""
    return {
        "threshold_percentile": float(sweep_row["threshold_percentile"]),
        "median_lead_time_days": (
            None
            if pd.isna(sweep_row["median_lead_time_days"])
            else float(sweep_row["median_lead_time_days"])
        ),
        "false_alarms_per_turbine_year": float(
            sweep_row["false_alarms_per_turbine_year"]
        ),
        "n_failure_turbines_with_alarm": int(
            sweep_row["n_failure_turbines_with_alarm"]
        ),
        "T01_lead_time_days": (
            None
            if pd.isna(sweep_row.get("T01_lead_time_days"))
            else float(sweep_row["T01_lead_time_days"])
        ),
        "T06_lead_time_days": (
            None
            if pd.isna(sweep_row.get("T06_lead_time_days"))
            else float(sweep_row["T06_lead_time_days"])
        ),
    }


def build_headline_claim(
    sweep_df: pd.DataFrame,
    hybrid_detector: str = PHYSICS_HYBRID_DETECTOR,
    best_pure_ml_detector: str | None = None,
    operating_pct: float = DEFAULT_THRESHOLD_PERCENTILE,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    data_note: str = "synthetic EDP; re-run on real data before publication",
) -> dict[str, Any]:
    """Build headline claim JSON from threshold sweep results."""
    hybrid_rows = sweep_df[
        (sweep_df["detector"] == hybrid_detector)
        & (sweep_df["threshold_percentile"] == operating_pct)
    ]
    if hybrid_rows.empty:
        raise ValueError(
            f"No sweep row for {hybrid_detector} at percentile {operating_pct}"
        )
    hybrid_row = hybrid_rows.iloc[0]

    ml_row_data: dict[str, Any] | None = None
    if best_pure_ml_detector:
        ml_rows = sweep_df[
            (sweep_df["detector"] == best_pure_ml_detector)
            & (sweep_df["threshold_percentile"] == operating_pct)
        ]
        if not ml_rows.empty:
            ml_row_data = _detector_claim_row(ml_rows.iloc[0])
            ml_row_data["detector"] = best_pure_ml_detector

    hybrid_claim = _detector_claim_row(hybrid_row)
    median = hybrid_claim["median_lead_time_days"]
    far = hybrid_claim["false_alarms_per_turbine_year"]
    headline = (
        f"At the {operating_pct:g}th-percentile threshold, {hybrid_detector} flags "
        f"gearbox failures a median of {median:.1f} days in advance at "
        f"{far:.1f} false alarms per turbine-year ({data_note})."
        if median is not None
        else (
            f"At the {operating_pct:g}th-percentile threshold, {hybrid_detector} "
            f"produces {far:.1f} false alarms per turbine-year but no qualifying "
            f"pre-failure alarms on failure turbines ({data_note})."
        )
    )

    return {
        "horizon_days": horizon_days,
        "operating_threshold_percentile": operating_pct,
        "data_note": data_note,
        PHYSICS_HYBRID_DETECTOR: hybrid_claim,
        "best_pure_ml": ml_row_data,
        "headline": headline,
    }


def write_threshold_sweep(
    raw_dir: Path = DATA_RAW,
    results_dir: Path = RESULTS_DIR,
    percentiles: list[float] | None = None,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    operating_pct: float = DEFAULT_THRESHOLD_PERCENTILE,
    data_note: str = "synthetic EDP; re-run on real data before publication",
) -> tuple[Path, Path, pd.DataFrame]:
    """
    Run threshold sweep for hybrid vs best pure ML; write CSV and headline JSON.

    Returns (csv_path, headline_path, sweep_df).
    """
    percentiles = percentiles or THRESHOLD_SWEEP_PERCENTILES
    best_ml = pick_best_pure_ml_detector(results_dir=results_dir)
    detectors = [PHYSICS_HYBRID_DETECTOR, best_ml]

    frames = [
        sweep_detector(
            detector,
            raw_dir=raw_dir,
            results_dir=results_dir,
            percentiles=percentiles,
            horizon_days=horizon_days,
        )
        for detector in detectors
    ]
    sweep_df = pd.concat(frames, ignore_index=True)

    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "threshold_sweep.csv"
    sweep_df.to_csv(csv_path, index=False)

    headline = build_headline_claim(
        sweep_df,
        best_pure_ml_detector=best_ml,
        operating_pct=operating_pct,
        horizon_days=horizon_days,
        data_note=data_note,
    )
    headline_path = results_dir / "headline_claim.json"
    headline_path.write_text(json.dumps(headline, indent=2), encoding="utf-8")

    print(f"Threshold sweep written to {csv_path}")
    print(f"Headline claim written to {headline_path}")
    print(f"Headline: {headline['headline']}")
    return csv_path, headline_path, sweep_df


__all__ = [
    "build_headline_claim",
    "median_lead_time_failure_turbines",
    "pick_best_pure_ml_detector",
    "sweep_detector",
    "threshold_at_percentile",
    "write_threshold_sweep",
]
