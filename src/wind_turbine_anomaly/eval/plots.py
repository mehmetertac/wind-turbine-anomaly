"""Score trajectory plots comparing pure-ML detectors."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from wind_turbine_anomaly.config import RESULTS_DIR
from wind_turbine_anomaly.eval.protocol import detect_alarm_episodes
from wind_turbine_anomaly.utils import to_utc


def plot_failure_trajectories(
    turbine_id: str,
    failure_time: pd.Timestamp,
    score_series_by_detector: dict[str, pd.Series],
    thresholds: dict[str, float],
    lookback_days: int = 60,
    out_path: Path | None = None,
) -> Path:
    """
    Plot normalized anomaly score trajectories for all detectors before failure.

    Scores are normalized by each detector's threshold (99th percentile = 1.0).
    """
    failure_time = to_utc(failure_time)
    window_start = failure_time - timedelta(days=lookback_days)

    n_detectors = len(score_series_by_detector)
    fig, axes = plt.subplots(
        n_detectors,
        1,
        figsize=(14, 3.5 * n_detectors),
        sharex=True,
        squeeze=False,
    )

    for ax, (detector, scores) in zip(
        axes.flatten(), score_series_by_detector.items()
    ):
        window = scores[(scores.index >= window_start) & (scores.index <= failure_time)]
        if window.empty:
            ax.set_title(f"{detector} — no scores in window")
            continue

        threshold = thresholds.get(detector, thresholds.get(turbine_id, 1.0))
        if threshold <= 0:
            threshold = 1.0
        normalized = window / threshold

        days_before = (window.index - failure_time).total_seconds() / 86400.0
        ax.plot(days_before, normalized.values, lw=0.8, label="Score / threshold")
        ax.axhline(1.0, color="red", ls="--", lw=0.8, label="Threshold")
        ax.axvline(0.0, color="black", ls="-", lw=1.0, label="Failure")

        raw_threshold = thresholds.get(detector, thresholds.get(turbine_id))
        if raw_threshold is not None:
            episodes = detect_alarm_episodes(window, threshold=raw_threshold)
            pre_failure = [ep for ep in episodes if ep.start < failure_time]
            if pre_failure:
                first = min(pre_failure, key=lambda ep: ep.start)
                first_day = (first.start - failure_time).total_seconds() / 86400.0
                ax.axvline(
                    first_day,
                    color="orange",
                    ls=":",
                    lw=1.0,
                    label="First alarm",
                )

        ax.set_ylabel("Normalized score")
        ax.set_title(detector)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Days before failure")
    fig.suptitle(f"{turbine_id} — anomaly score trajectories ({lookback_days}d window)")
    fig.tight_layout()

    if out_path is None:
        out_path = RESULTS_DIR / "plots" / f"trajectory_{turbine_id}.png"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_all_failure_trajectories(
    failure_turbines: dict[str, pd.Timestamp],
    detectors: list[str],
    lookback_days: int = 60,
) -> list[Path]:
    """Load scores/thresholds from results/ and plot trajectories for each failure."""
    paths: list[Path] = []
    for turbine_id, failure_time in failure_turbines.items():
        score_series: dict[str, pd.Series] = {}
        thresholds: dict[str, float] = {}
        for detector in detectors:
            det_dir = RESULTS_DIR / detector
            score_path = det_dir / f"{turbine_id}_scores.parquet"
            metrics_path = det_dir / "metrics.json"
            if not score_path.exists():
                continue
            scores = pd.read_parquet(score_path).iloc[:, 0]
            scores.index = pd.to_datetime(scores.index, utc=True)
            score_series[detector] = scores
            if metrics_path.exists():
                import json

                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                det_thresholds = metrics.get("thresholds", {})
                if turbine_id in det_thresholds:
                    thresholds[detector] = det_thresholds[turbine_id]

        if not score_series:
            continue

        path = plot_failure_trajectories(
            turbine_id,
            failure_time,
            score_series,
            thresholds,
            lookback_days=lookback_days,
        )
        paths.append(path)
        print(f"Plot written to {path}")

    return paths
