"""Robustness validation: multi-turbine, seasonal residuals, leakage audit."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wind_turbine_anomaly.config import (
    DATA_RAW,
    DEFAULT_BUFFER_DAYS,
    DEFAULT_THRESHOLD_PERCENTILE,
    FAILURE_TURBINES,
    HEALTHY_MAX_RESIDUAL_DRIFT_C,
    PHYSICS_HYBRID_DETECTOR,
    RESULTS_DIR,
    ROBUSTNESS_RESULTS_DIR,
    SEASONAL_RMSE_RATIO_MAX,
    THERMAL_MIN_POWER_KW,
    THERMAL_RESULTS_DIR,
    THERMAL_TARGET_COLUMNS,
)
from wind_turbine_anomaly.data.clean import (
    clean_turbine_df,
    get_failure_for_turbine,
    healthy_training_mask,
)
from wind_turbine_anomaly.data.load_edp import load_edp_dataset
from wind_turbine_anomaly.eval.protocol import evaluate_turbine
from wind_turbine_anomaly.models.gearbox_thermal import (
    THERMAL_VALIDATION_THRESHOLDS as _THERMAL_THRESHOLDS,
    GearboxThermalModel,
    fit_gearbox_thermal_with_selection,
    healthy_train_validate_split,
    validate_thermal_model,
)
from wind_turbine_anomaly.models.scoring import threshold_from_training
from wind_turbine_anomaly.utils import to_utc

SeasonLabel = Literal["winter", "summer", "shoulder"]
TARGET_SLUGS = {"Gear_Oil_Temp_Avg": "oil", "Gear_Bear_Temp_Avg": "bear"}


def meteorological_season(month: int) -> SeasonLabel:
    """Map calendar month to meteorological season bucket."""
    if month in (12, 1, 2):
        return "winter"
    if month in (6, 7, 8):
        return "summer"
    return "shoulder"


def seasonal_bucket(index: pd.DatetimeIndex) -> pd.Series:
    """Return season label per timestamp."""
    return pd.Series([meteorological_season(m) for m in index.month], index=index)


def compute_seasonal_residual_metrics(
    model: GearboxThermalModel,
    val_df: pd.DataFrame,
) -> dict[str, Any]:
    """RMSE and mean residual per meteorological season on healthy validation rows."""
    if val_df.empty:
        return {"seasons": {}, "passed": False, "reason": "empty validation set"}

    residuals = model.residual(val_df)
    actual = val_df[model.target_column]
    predicted = model.predict(val_df)
    seasons = seasonal_bucket(val_df.index)

    season_metrics: dict[str, dict[str, float | int]] = {}
    for season in ("winter", "summer", "shoulder"):
        mask = seasons == season
        n = int(mask.sum())
        if n == 0:
            season_metrics[season] = {"n_samples": 0}
            continue
        res = residuals[mask]
        act = actual[mask]
        pred = predicted[mask]
        rmse = float(np.sqrt(np.mean((act - pred) ** 2)))
        mean_res = float(res.mean())
        season_metrics[season] = {
            "n_samples": n,
            "rmse": rmse,
            "mean_residual": mean_res,
            "abs_mean_residual": abs(mean_res),
        }

    rmse_values = [
        m["rmse"]
        for m in season_metrics.values()
        if m.get("n_samples", 0) > 0 and "rmse" in m
    ]
    ratio = float(max(rmse_values) / min(rmse_values)) if len(rmse_values) >= 2 else 1.0

    max_abs_mean = max(
        (
            m.get("abs_mean_residual", 0.0)
            for m in season_metrics.values()
            if m.get("n_samples", 0) > 0
        ),
        default=0.0,
    )

    ratio_ok = ratio < SEASONAL_RMSE_RATIO_MAX
    bias_ok = max_abs_mean < _THERMAL_THRESHOLDS["max_abs_mean_residual_c"]
    passed = ratio_ok and bias_ok

    return {
        "seasons": season_metrics,
        "rmse_ratio": ratio,
        "max_abs_mean_residual": max_abs_mean,
        "ratio_ok": ratio_ok,
        "bias_ok": bias_ok,
        "passed": passed,
    }


def compute_healthy_residual_drift(
    residual_frame: pd.DataFrame,
    window_days: int = 90,
) -> float:
    """Max |mean residual| over trailing windows (healthy turbine sanity check)."""
    if residual_frame.empty or len(residual_frame) < 2:
        return 0.0
    window = timedelta(days=window_days)
    start = residual_frame.index[0]
    end = residual_frame.index[-1]
    max_abs = 0.0
    t = start
    while t + window <= end:
        chunk = residual_frame.loc[t : t + window]
        if not chunk.empty:
            max_abs = max(max_abs, abs(float(chunk["residual"].mean())))
        t += window / 2
    return max_abs


def audit_leakage_for_turbine(
    turbine_id: str,
    raw_df: pd.DataFrame,
    failure,
    buffer_days: int,
    detector: str = PHYSICS_HYBRID_DETECTOR,
    results_dir: Path = RESULTS_DIR,
    threshold_percentile: float = DEFAULT_THRESHOLD_PERCENTILE,
) -> dict[str, Any]:
    """Run leakage assertions for one turbine."""
    checks: dict[str, dict[str, Any]] = {}
    df = clean_turbine_df(raw_df, min_power_kw=THERMAL_MIN_POWER_KW)
    train_mask = healthy_training_mask(df.index, failure, buffer_days)
    train_df = df.loc[train_mask]

    if failure is not None:
        cutoff = to_utc(failure.timestamp) - timedelta(days=buffer_days)
        train_before_cutoff = bool(train_df.index.max() < cutoff) if not train_df.empty else False
        checks["train_before_failure_buffer"] = {
            "passed": train_before_cutoff,
            "detail": f"max_train={train_df.index.max()}, cutoff={cutoff}",
        }
    else:
        cutoff_idx = int(len(df.index) * 0.9)
        expected_cutoff = df.index[cutoff_idx - 1] if cutoff_idx > 0 else df.index[0]
        train_ok = (
            bool(train_df.index.max() <= expected_cutoff) if not train_df.empty else False
        )
        checks["train_before_failure_buffer"] = {
            "passed": train_ok,
            "detail": f"no failure; max_train={train_df.index.max()}, expected<={expected_cutoff}",
        }

    train_path = results_dir / detector / f"{turbine_id}_train_scores.parquet"
    score_path = results_dir / detector / f"{turbine_id}_scores.parquet"
    metrics_path = results_dir / detector / "metrics.json"

    if train_path.exists() and score_path.exists() and metrics_path.exists():
        train_scores = pd.read_parquet(train_path).iloc[:, 0]
        all_scores = pd.read_parquet(score_path).iloc[:, 0]
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        threshold = metrics.get("thresholds", {}).get(turbine_id)
        recomputed = threshold_from_training(
            train_scores, percentile=threshold_percentile
        )

        threshold_ok = threshold is not None and abs(threshold - recomputed) < 1e-6
        checks["threshold_from_train_only"] = {
            "passed": threshold_ok,
            "detail": f"stored={threshold}, recomputed={recomputed}",
        }

        overlap = all_scores.index.intersection(train_df.index)
        score_start = train_df.index[-1] if not train_df.empty else None
        eval_result = evaluate_turbine(
            turbine_id,
            all_scores,
            threshold=float(threshold),
            failure=failure,
            score_start=score_start,
        )
        score_start_ok = score_start is None or (
            eval_result.scored_days
            <= (all_scores.index[-1] - score_start).total_seconds() / 86400.0 + 1.0
        )
        checks["score_start_excludes_training"] = {
            "passed": score_start_ok and len(overlap) == 0,
            "detail": f"score_start={score_start}, overlap_rows={len(overlap)}",
        }
    else:
        checks["threshold_from_train_only"] = {
            "passed": False,
            "detail": "missing score parquets or metrics.json",
        }
        checks["score_start_excludes_training"] = {
            "passed": False,
            "detail": "missing score parquets",
        }

    healthy_mask = healthy_training_mask(df.index, failure, buffer_days)
    train_df_thermal, val_df = healthy_train_validate_split(df.loc[healthy_mask])
    split_ok = (
        train_df_thermal.empty
        or val_df.empty
        or train_df_thermal.index.max() <= val_df.index.min()
    )
    checks["thermal_time_ordered_split"] = {
        "passed": split_ok,
        "detail": (
            f"train_max={train_df_thermal.index.max() if not train_df_thermal.empty else None}, "
            f"val_min={val_df.index.min() if not val_df.empty else None}"
        ),
    }

    thermal_subset_ok = True
    if not train_df_thermal.empty:
        thermal_subset_ok = train_df_thermal.index.isin(df.loc[healthy_mask].index).all()
    checks["thermal_fit_on_healthy_mask"] = {
        "passed": bool(thermal_subset_ok),
        "detail": f"thermal_train_rows={len(train_df_thermal)}",
    }

    all_passed = all(c["passed"] for c in checks.values())
    return {"turbine_id": turbine_id, "checks": checks, "passed": all_passed}


def build_multi_turbine_summary(
    results_dir: Path = RESULTS_DIR,
    thermal_dir: Path = THERMAL_RESULTS_DIR,
    detector: str = PHYSICS_HYBRID_DETECTOR,
) -> dict[str, Any]:
    """Aggregate hybrid + thermal validation across turbines."""
    metrics_path = results_dir / detector / "metrics.json"
    if not metrics_path.exists():
        return {"passed": False, "reason": f"missing {metrics_path}", "turbines": {}}

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    by_turbine = {t["turbine_id"]: t for t in metrics.get("turbines", [])}
    summary: dict[str, Any] = {"turbines": {}, "passed": True}

    for turbine_id, row in sorted(by_turbine.items()):
        tid = str(turbine_id)
        is_failure = tid in FAILURE_TURBINES
        score_exists = (results_dir / detector / f"{tid}_scores.parquet").exists()

        thermal_passed = True
        for target_col, slug in TARGET_SLUGS.items():
            val_path = thermal_dir / f"{tid}_{slug}_validation.json"
            if val_path.exists():
                val_data = json.loads(val_path.read_text(encoding="utf-8"))
                if not val_data.get("passed", False):
                    thermal_passed = False
            else:
                thermal_passed = False

        lead = row.get("lead_time_days")
        checks: dict[str, Any] = {
            "scores_saved": score_exists,
            "thermal_validation_passed": thermal_passed,
        }
        if is_failure:
            checks["lead_time_non_negative"] = lead is not None and lead >= 0
        else:
            residual_drift = 0.0
            bear_path = thermal_dir / f"{tid}_bear_residuals.parquet"
            if bear_path.exists():
                residual_drift = compute_healthy_residual_drift(
                    pd.read_parquet(bear_path)
                )
            checks["residual_drift_bounded"] = (
                residual_drift < HEALTHY_MAX_RESIDUAL_DRIFT_C
            )
            checks["residual_drift_c"] = residual_drift

        turbine_passed = all(
            v for k, v in checks.items() if isinstance(v, bool)
        )
        summary["turbines"][tid] = {
            "role": "failure" if is_failure else "healthy",
            "checks": checks,
            "lead_time_days": lead,
            "false_alarm_episodes": row.get("false_alarm_episodes"),
            "passed": turbine_passed,
        }
        if not turbine_passed:
            summary["passed"] = False

    summary["false_alarms_per_turbine_year"] = metrics.get(
        "false_alarms_per_turbine_year"
    )
    return summary


def plot_seasonal_rmse_by_turbine(
    seasonal_report: dict[str, Any],
    out_path: Path | None = None,
    results_dir: Path = RESULTS_DIR,
) -> Path:
    """Bar chart of seasonal RMSE per turbine × target."""
    turbines = sorted(seasonal_report.get("turbines", {}).keys())
    if not turbines:
        out_path = out_path or results_dir / "plots" / "seasonal_rmse_by_turbine.png"
        return Path(out_path)

    seasons = ["winter", "summer", "shoulder"]
    n_panels = len(turbines)
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 3.5 * n_panels), squeeze=False)

    for ax, tid in zip(axes.flatten(), turbines):
        targets = seasonal_report["turbines"][tid].get("targets", {})
        x = np.arange(len(seasons))
        width = 0.35
        for i, (target_col, slug) in enumerate(TARGET_SLUGS.items()):
            tdata = targets.get(target_col, {})
            rmse_vals = [
                tdata.get("seasons", {}).get(s, {}).get("rmse", 0.0) for s in seasons
            ]
            ax.bar(x + (i - 0.5) * width, rmse_vals, width, label=slug)
        ax.set_xticks(x)
        ax.set_xticklabels(seasons)
        ax.set_ylabel("RMSE (°C)")
        ax.set_title(tid)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Seasonal thermal-model RMSE on healthy validation")
    fig.tight_layout()
    if out_path is None:
        out_path = results_dir / "plots" / "seasonal_rmse_by_turbine.png"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def run_seasonal_analysis(
    raw_dir: Path = DATA_RAW,
    buffer_days: int = DEFAULT_BUFFER_DAYS,
    seasonal_terms: bool = False,
) -> dict[str, Any]:
    """Fit thermal models and compute per-season validation metrics."""
    turbines, failures = load_edp_dataset(raw_dir)
    report: dict[str, Any] = {"turbines": {}, "passed": True, "seasonal_terms": seasonal_terms}

    for turbine_id, raw_df in sorted(turbines.items()):
        df = clean_turbine_df(raw_df, min_power_kw=THERMAL_MIN_POWER_KW)
        failure = get_failure_for_turbine(turbine_id, failures)
        healthy_mask = healthy_training_mask(df.index, failure, buffer_days)
        healthy_df = df.loc[healthy_mask]
        if len(healthy_df) < 100:
            continue

        turbine_data: dict[str, Any] = {"targets": {}}
        for target_col in THERMAL_TARGET_COLUMNS:
            model, _train, val_df, _info = fit_gearbox_thermal_with_selection(
                healthy_df, target_col, seasonal_terms=seasonal_terms
            )
            metrics = compute_seasonal_residual_metrics(model, val_df)
            turbine_data["targets"][target_col] = metrics
            if not metrics.get("passed", False):
                report["passed"] = False

        report["turbines"][turbine_id] = turbine_data

    return report


def run_leakage_audit(
    raw_dir: Path = DATA_RAW,
    buffer_days: int = DEFAULT_BUFFER_DAYS,
    results_dir: Path = RESULTS_DIR,
) -> dict[str, Any]:
    """Leakage audit across all turbines."""
    turbines, failures = load_edp_dataset(raw_dir)
    report: dict[str, Any] = {"turbines": {}, "passed": True}

    for turbine_id, raw_df in sorted(turbines.items()):
        failure = get_failure_for_turbine(turbine_id, failures)
        audit = audit_leakage_for_turbine(
            turbine_id, raw_df, failure, buffer_days, results_dir=results_dir
        )
        report["turbines"][turbine_id] = audit
        if not audit["passed"]:
            report["passed"] = False

    return report


def run_robustness_pass(
    raw_dir: Path = DATA_RAW,
    results_dir: Path = RESULTS_DIR,
    out_dir: Path = ROBUSTNESS_RESULTS_DIR,
    buffer_days: int = DEFAULT_BUFFER_DAYS,
    apply_seasonal_refit: bool = True,
) -> dict[str, Any]:
    """
    Full robustness pass: multi-turbine, seasonal, leakage.

    If seasonal check fails and apply_seasonal_refit is True, re-runs seasonal
    analysis with THERMAL_SEASONAL_TERMS enabled and records before/after.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    multi = build_multi_turbine_summary(results_dir=results_dir)
    multi_path = out_dir / "multi_turbine_summary.json"
    multi_path.write_text(json.dumps(multi, indent=2), encoding="utf-8")

    seasonal = run_seasonal_analysis(raw_dir=raw_dir, buffer_days=buffer_days)
    refit_info: dict[str, Any] | None = None

    if not seasonal.get("passed") and apply_seasonal_refit:
        seasonal_before = seasonal
        seasonal_after = run_seasonal_analysis(
            raw_dir=raw_dir, buffer_days=buffer_days, seasonal_terms=True
        )
        refit_info = {
            "triggered": True,
            "before_passed": seasonal_before.get("passed"),
            "after_passed": seasonal_after.get("passed"),
            "before": seasonal_before,
            "after": seasonal_after,
        }
        seasonal = seasonal_after
        refit_path = out_dir / "seasonal_refit.json"
        refit_path.write_text(json.dumps(refit_info, indent=2), encoding="utf-8")

    seasonal_path = out_dir / "seasonal_residuals.json"
    seasonal_path.write_text(json.dumps(seasonal, indent=2), encoding="utf-8")
    plot_seasonal_rmse_by_turbine(seasonal, results_dir=results_dir)

    leakage = run_leakage_audit(raw_dir=raw_dir, buffer_days=buffer_days, results_dir=results_dir)
    leakage_path = out_dir / "leakage_audit.json"
    leakage_path.write_text(json.dumps(leakage, indent=2), encoding="utf-8")

    overall_passed = (
        multi.get("passed", False)
        and seasonal.get("passed", False)
        and leakage.get("passed", False)
    )

    summary = {
        "passed": overall_passed,
        "multi_turbine": multi_path.name,
        "seasonal_residuals": seasonal_path.name,
        "leakage_audit": leakage_path.name,
        "seasonal_refit_applied": refit_info is not None,
    }
    summary_path = out_dir / "robustness_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "passed": overall_passed,
        "multi_turbine": multi,
        "seasonal": seasonal,
        "leakage": leakage,
        "refit": refit_info,
        "paths": {
            "multi_turbine": str(multi_path),
            "seasonal": str(seasonal_path),
            "leakage": str(leakage_path),
            "summary": str(summary_path),
        },
    }


__all__ = [
    "audit_leakage_for_turbine",
    "build_multi_turbine_summary",
    "compute_healthy_residual_drift",
    "compute_seasonal_residual_metrics",
    "meteorological_season",
    "plot_seasonal_rmse_by_turbine",
    "run_leakage_audit",
    "run_robustness_pass",
    "run_seasonal_analysis",
    "seasonal_bucket",
]
