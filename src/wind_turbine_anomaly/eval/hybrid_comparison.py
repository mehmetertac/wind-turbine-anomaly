"""Head-to-head comparison: physics-residual hybrid vs pure ML detectors."""

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
    MIN_POWER_KW,
    PHYSICS_HYBRID_DETECTOR,
    POWER_COLUMN,
    RESULTS_DIR,
    THERMAL_MIN_POWER_KW,
)
from wind_turbine_anomaly.data.clean import (
    clean_turbine_df,
    get_failure_for_turbine,
    healthy_training_mask,
)
from wind_turbine_anomaly.data.load_edp import load_edp_dataset
from wind_turbine_anomaly.eval.metrics_table import load_detector_metrics
from wind_turbine_anomaly.eval.protocol import (
    detect_alarm_episodes,
    is_false_alarm,
)
from wind_turbine_anomaly.utils import to_utc

PURE_ML_DETECTORS = [
    "isolation_forest",
    "dense_autoencoder",
    "lstm_autoencoder",
]
ALL_DETECTORS = PURE_ML_DETECTORS + [PHYSICS_HYBRID_DETECTOR]
FAILURE_TURBINES = ("T01", "T06")
NAC_COLUMN = "Nac_Temp_Avg"


def _load_scores_and_threshold(
    detector: str,
    turbine_id: str,
    results_dir: Path = RESULTS_DIR,
) -> tuple[pd.Series, float] | None:
    score_path = results_dir / detector / f"{turbine_id}_scores.parquet"
    metrics_path = results_dir / detector / "metrics.json"
    if not score_path.exists() or not metrics_path.exists():
        return None
    scores = pd.read_parquet(score_path).iloc[:, 0]
    scores.index = pd.to_datetime(scores.index, utc=True)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    threshold = metrics.get("thresholds", {}).get(turbine_id)
    if threshold is None:
        return None
    return scores, float(threshold)


def _training_regime_thresholds(
    train_df: pd.DataFrame,
) -> dict[str, float]:
    """P90 power and nacelle temp from healthy training rows."""
    return {
        "power_p90": float(train_df[POWER_COLUMN].quantile(0.90)),
        "nac_p90": float(train_df[NAC_COLUMN].quantile(0.90)),
    }


def classify_regime(
    row: pd.Series,
    thresholds: dict[str, float],
) -> str:
    """Classify operating point at alarm time."""
    high_load = row[POWER_COLUMN] > thresholds["power_p90"]
    hot_ambient = row[NAC_COLUMN] > thresholds["nac_p90"]
    if high_load and hot_ambient:
        return "high_load_and_hot_ambient"
    if high_load:
        return "high_load"
    if hot_ambient:
        return "hot_ambient"
    return "normal"


def analyze_regime_false_alarms(
    raw_dir: Path = DATA_RAW,
    detectors: list[str] | None = None,
    results_dir: Path = RESULTS_DIR,
    buffer_days: int = DEFAULT_BUFFER_DAYS,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    min_power_kw_by_detector: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Tag false-alarm episodes by operating regime at alarm time.

    Regimes use training P90 for power and nacelle temperature.
    """
    detectors = detectors or ALL_DETECTORS
    min_power_kw_by_detector = min_power_kw_by_detector or {
        PHYSICS_HYBRID_DETECTOR: THERMAL_MIN_POWER_KW,
    }
    turbines, failures = load_edp_dataset(raw_dir)

    by_detector: dict[str, dict[str, Any]] = {}
    for detector in detectors:
        regime_counts: dict[str, int] = {
            "high_load": 0,
            "hot_ambient": 0,
            "high_load_and_hot_ambient": 0,
            "normal": 0,
        }
        total_false = 0
        episodes_detail: list[dict[str, Any]] = []

        for turbine_id, raw_df in sorted(turbines.items()):
            min_power = min_power_kw_by_detector.get(detector, MIN_POWER_KW)
            df = clean_turbine_df(raw_df, min_power_kw=min_power)
            failure = get_failure_for_turbine(turbine_id, failures)
            train_mask = healthy_training_mask(df.index, failure, buffer_days)
            train_df = df.loc[train_mask]
            if train_df.empty:
                continue

            loaded = _load_scores_and_threshold(detector, turbine_id, results_dir)
            if loaded is None:
                continue
            scores, threshold = loaded
            failure_time = to_utc(failure.timestamp) if failure else None
            regime_thresholds = _training_regime_thresholds(train_df)

            episodes = detect_alarm_episodes(scores, threshold=threshold)
            for ep in episodes:
                if not is_false_alarm(ep.start, failure_time, horizon_days):
                    continue
                total_false += 1
                nearest_idx = df.index.get_indexer([ep.start], method="nearest")[0]
                if nearest_idx < 0:
                    continue
                row = df.iloc[nearest_idx]
                regime = classify_regime(row, regime_thresholds)
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
                episodes_detail.append(
                    {
                        "turbine_id": turbine_id,
                        "alarm_start": ep.start.isoformat(),
                        "regime": regime,
                        "power_kw": float(row[POWER_COLUMN]),
                        "nac_temp_c": float(row[NAC_COLUMN]),
                    }
                )

        by_detector[detector] = {
            "total_false_alarms": total_false,
            "by_regime": regime_counts,
            "episodes": episodes_detail,
        }

    return {"detectors": by_detector}


def build_hybrid_vs_ml_summary(
    hybrid_detector: str = PHYSICS_HYBRID_DETECTOR,
    pure_ml_detectors: list[str] | None = None,
    results_dir: Path = RESULTS_DIR,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> dict[str, Any]:
    """Side-by-side lead times, false-alarm rates, and deltas vs best pure ML."""
    pure_ml_detectors = pure_ml_detectors or PURE_ML_DETECTORS
    hybrid_metrics = load_detector_metrics(hybrid_detector, results_dir)
    ml_metrics_list = [
        load_detector_metrics(d, results_dir) for d in pure_ml_detectors
    ]

    hybrid_rate = hybrid_metrics.get("false_alarms_per_turbine_year")
    ml_rates = [m.get("false_alarms_per_turbine_year") for m in ml_metrics_list]
    best_ml_rate = min(r for r in ml_rates if r is not None)

    per_turbine: dict[str, dict[str, Any]] = {}
    for tid in FAILURE_TURBINES:
        hybrid_t = next(
            (t for t in hybrid_metrics.get("turbines", []) if t["turbine_id"] == tid),
            {},
        )
        ml_leads = []
        for det, metrics in zip(pure_ml_detectors, ml_metrics_list):
            t = next(
                (x for x in metrics.get("turbines", []) if x["turbine_id"] == tid),
                {},
            )
            lead = t.get("lead_time_days")
            if lead is not None:
                ml_leads.append((det, lead))
        best_ml_lead = max((lead for _, lead in ml_leads), default=None)
        best_ml_det = next(
            (det for det, lead in ml_leads if lead == best_ml_lead),
            None,
        )
        hybrid_lead = hybrid_t.get("lead_time_days")
        per_turbine[tid] = {
            "hybrid_lead_time_days": hybrid_lead,
            "hybrid_successful_warning": hybrid_t.get("successful_warning"),
            "best_pure_ml_detector": best_ml_det,
            "best_pure_ml_lead_time_days": best_ml_lead,
            "lead_time_delta_days": (
                (hybrid_lead - best_ml_lead)
                if hybrid_lead is not None and best_ml_lead is not None
                else None
            ),
            "hybrid_wins_lead_time": (
                hybrid_lead is not None
                and best_ml_lead is not None
                and hybrid_lead >= best_ml_lead
            ),
        }

    return {
        "horizon_days": horizon_days,
        "hybrid_detector": hybrid_detector,
        "pure_ml_detectors": pure_ml_detectors,
        "false_alarms_per_turbine_year": {
            "hybrid": hybrid_rate,
            "best_pure_ml": best_ml_rate,
            "delta_hybrid_minus_best_ml": (
                (hybrid_rate - best_ml_rate)
                if hybrid_rate is not None and best_ml_rate is not None
                else None
            ),
            "hybrid_wins_false_alarms": (
                hybrid_rate is not None
                and best_ml_rate is not None
                and hybrid_rate < best_ml_rate
            ),
        },
        "per_failure_turbine": per_turbine,
    }


def write_hybrid_comparison(
    raw_dir: Path = DATA_RAW,
    results_dir: Path = RESULTS_DIR,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> tuple[Path, Path]:
    """Write hybrid_vs_ml_summary.json and hybrid_vs_ml_regime.json."""
    results_dir.mkdir(parents=True, exist_ok=True)

    summary = build_hybrid_vs_ml_summary(results_dir=results_dir, horizon_days=horizon_days)
    regime = analyze_regime_false_alarms(raw_dir=raw_dir, results_dir=results_dir)

    summary_path = results_dir / "hybrid_vs_ml_summary.json"
    regime_path = results_dir / "hybrid_vs_ml_regime.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    regime_path.write_text(json.dumps(regime, indent=2), encoding="utf-8")
    print(f"Hybrid comparison summary written to {summary_path}")
    print(f"Regime false-alarm analysis written to {regime_path}")
    return summary_path, regime_path


__all__ = [
    "ALL_DETECTORS",
    "PURE_ML_DETECTORS",
    "analyze_regime_false_alarms",
    "build_hybrid_vs_ml_summary",
    "write_hybrid_comparison",
]
