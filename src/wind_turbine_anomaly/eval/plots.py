"""Score trajectory plots comparing pure-ML and physics-hybrid detectors."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wind_turbine_anomaly.config import PHYSICS_HYBRID_DETECTOR, RESULTS_DIR
from wind_turbine_anomaly.eval.hybrid_comparison import (
    PURE_ML_DETECTORS,
    analyze_regime_false_alarms,
    build_hybrid_vs_ml_summary,
)
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


def plot_gearbox_residual_trajectories(
    turbine_id: str,
    target_label: str,
    failure_time: pd.Timestamp | None,
    residual_frame: pd.DataFrame,
    power_series: pd.Series | None = None,
    lookback_days: int = 90,
    out_path: Path | None = None,
) -> Path:
    """
    Plot gearbox thermal residuals around a failure (or full series if no failure).

    Three panels: actual vs predicted, residual over time, residual vs power.
    """
    if failure_time is not None:
        failure_time = to_utc(failure_time)
        window_start = failure_time - timedelta(days=lookback_days)
        window = residual_frame[
            (residual_frame.index >= window_start)
            & (residual_frame.index <= failure_time)
        ]
    else:
        window = residual_frame
        failure_time = window.index[-1] if not window.empty else pd.Timestamp.now(tz="UTC")

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    days_before = (window.index - failure_time).total_seconds() / 86400.0

    ax0 = axes[0]
    ax0.plot(days_before, window["actual"].values, lw=0.8, label="Actual", alpha=0.9)
    ax0.plot(days_before, window["predicted"].values, lw=0.8, label="Predicted", alpha=0.9)
    ax0.set_ylabel("Temperature (°C)")
    ax0.set_title(f"{turbine_id} — {target_label}: actual vs predicted")
    ax0.legend(loc="upper left", fontsize=8)
    ax0.grid(True, alpha=0.3)

    ax1 = axes[1]
    ax1.plot(days_before, window["residual"].values, lw=0.8, color="tab:orange", label="Residual")
    ax1.axhline(0.0, color="black", ls="--", lw=0.8)
    if failure_time is not None:
        ax1.axvline(0.0, color="red", ls="-", lw=1.0, label="Failure")
    ax1.set_ylabel("Residual (°C)")
    ax1.set_title("Predicted − actual (negative = hotter than expected)")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[2]
    if power_series is not None:
        power_window = power_series.reindex(window.index)
        sc = ax2.scatter(
            power_window.values,
            window["residual"].values,
            c=days_before,
            cmap="viridis",
            s=8,
            alpha=0.6,
        )
        plt.colorbar(sc, ax=ax2, label="Days before failure")
        ax2.set_xlabel("Power (kW)")
    else:
        ax2.scatter(range(len(window)), window["residual"].values, s=8, alpha=0.6)
        ax2.set_xlabel("Sample index")
    ax2.axhline(0.0, color="black", ls="--", lw=0.8)
    ax2.set_ylabel("Residual (°C)")
    ax2.set_title("Residual vs power")
    ax2.grid(True, alpha=0.3)

    axes[1].set_xlabel("Days before failure")
    title_suffix = f"({lookback_days}d window)" if failure_time is not None else ""
    fig.suptitle(f"{turbine_id} — {target_label} thermal residuals {title_suffix}")
    fig.tight_layout()

    if out_path is None:
        slug = target_label.replace(" ", "_").lower()
        out_path = RESULTS_DIR / "plots" / f"residual_{turbine_id}_{slug}.png"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_hybrid_vs_ml_comparison(
    raw_dir: Path | None = None,
    results_dir: Path = RESULTS_DIR,
    out_path: Path | None = None,
) -> Path:
    """
    Grouped bar chart: lead time and false alarms by detector;
    stacked regime breakdown for false alarms.
    """
    from wind_turbine_anomaly.config import DATA_RAW

    raw_dir = raw_dir or DATA_RAW
    summary = build_hybrid_vs_ml_summary(results_dir=results_dir)
    regime = analyze_regime_false_alarms(raw_dir=raw_dir, results_dir=results_dir)

    detectors = PURE_ML_DETECTORS + [PHYSICS_HYBRID_DETECTOR]
    lead_t01 = []
    lead_t06 = []
    false_rates = []
    for det in detectors:
        metrics_path = results_dir / det / "metrics.json"
        if not metrics_path.exists():
            lead_t01.append(0.0)
            lead_t06.append(0.0)
            false_rates.append(0.0)
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        by_turbine = {t["turbine_id"]: t for t in metrics.get("turbines", [])}
        lead_t01.append(by_turbine.get("T01", {}).get("lead_time_days") or 0.0)
        lead_t06.append(by_turbine.get("T06", {}).get("lead_time_days") or 0.0)
        false_rates.append(metrics.get("false_alarms_per_turbine_year") or 0.0)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(len(detectors))
    width = 0.35

    ax0 = axes[0]
    ax0.bar(x - width / 2, lead_t01, width, label="T01")
    ax0.bar(x + width / 2, lead_t06, width, label="T06")
    ax0.set_xticks(x)
    ax0.set_xticklabels(detectors, rotation=25, ha="right")
    ax0.set_ylabel("Lead time (days)")
    ax0.set_title("Pre-failure lead time")
    ax0.legend()
    ax0.grid(True, alpha=0.3, axis="y")

    ax1 = axes[1]
    ax1.bar(x, false_rates, color="tab:orange", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(detectors, rotation=25, ha="right")
    ax1.set_ylabel("False alarms / turbine-year")
    ax1.set_title("Aggregate false-alarm rate")
    ax1.grid(True, alpha=0.3, axis="y")

    ax2 = axes[2]
    regime_labels = ["high_load", "hot_ambient", "high_load_and_hot_ambient", "normal"]
    bottom = np.zeros(len(detectors))
    colors = ["tab:red", "tab:purple", "tab:brown", "tab:gray"]
    for label, color in zip(regime_labels, colors):
        counts = [
            regime["detectors"].get(det, {}).get("by_regime", {}).get(label, 0)
            for det in detectors
        ]
        ax2.bar(x, counts, bottom=bottom, label=label, color=color, alpha=0.85)
        bottom = bottom + np.array(counts, dtype=float)
    ax2.set_xticks(x)
    ax2.set_xticklabels(detectors, rotation=25, ha="right")
    ax2.set_ylabel("False alarm episodes")
    ax2.set_title("False alarms by operating regime")
    ax2.legend(fontsize=7, loc="upper right")
    ax2.grid(True, alpha=0.3, axis="y")

    hybrid_wins = summary.get("false_alarms_per_turbine_year", {}).get(
        "hybrid_wins_false_alarms", False
    )
    fig.suptitle(
        f"Hybrid vs pure ML — hybrid wins false alarms: {hybrid_wins}",
        fontsize=11,
    )
    fig.tight_layout()

    if out_path is None:
        out_path = results_dir / "plots" / "hybrid_vs_ml_comparison.png"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Hybrid comparison plot written to {out_path}")
    return out_path
