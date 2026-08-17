"""Thermal model interpretability: SHAP, coefficients, failure case study."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from wind_turbine_anomaly.config import (
    DATA_RAW,
    DEFAULT_BUFFER_DAYS,
    DEFAULT_THRESHOLD_PERCENTILE,
    INTERPRETABILITY_RESULTS_DIR,
    PHYSICS_HYBRID_DETECTOR,
    POWER_COLUMN,
    RESULTS_DIR,
    THERMAL_MIN_POWER_KW,
    THERMAL_TARGET_COLUMNS,
)
from wind_turbine_anomaly.data.clean import (
    clean_turbine_df,
    get_failure_for_turbine,
    healthy_training_mask,
)
from wind_turbine_anomaly.data.load_edp import load_edp_dataset
from wind_turbine_anomaly.eval.plots import plot_failure_case_study
from wind_turbine_anomaly.eval.protocol import evaluate_turbine
from wind_turbine_anomaly.models.gearbox_thermal import (
    GearboxThermalModel,
    fit_gearbox_thermal_with_selection,
)
from wind_turbine_anomaly.utils import to_utc

TARGET_SLUGS = {"Gear_Oil_Temp_Avg": "oil", "Gear_Bear_Temp_Avg": "bear"}
SHAP_SAMPLE_SIZE = 500


def _stratified_sample(df: pd.DataFrame, n: int = SHAP_SAMPLE_SIZE) -> pd.DataFrame:
    """Sample rows stratified by power decile."""
    if len(df) <= n:
        return df
    try:
        deciles = pd.qcut(df[POWER_COLUMN], q=10, duplicates="drop")
        per_group = max(1, n // deciles.nunique())
        parts = [g.head(per_group) for _, g in df.groupby(deciles, observed=True)]
        sampled = pd.concat(parts).head(n)
        return sampled if len(sampled) >= 10 else df.sample(n=min(n, len(df)), random_state=42)
    except ValueError:
        return df.sample(n=min(n, len(df)), random_state=42)


def _feature_matrix(model: GearboxThermalModel, df: pd.DataFrame) -> np.ndarray:
    return model._build_features(df)


def explain_thermal_model(
    model: GearboxThermalModel,
    healthy_df: pd.DataFrame,
) -> dict[str, Any]:
    """Compute SHAP or coefficient-based driver rankings."""
    sample = _stratified_sample(healthy_df)
    feature_names = model.feature_names or model.driver_columns

    if model.model_kind == "linear":
        X = _feature_matrix(model, sample)
        pipeline = model.model
        scaler = pipeline.named_steps["scaler"]
        ridge = pipeline.named_steps["ridge"]
        X_scaled = scaler.transform(X)
        explainer = shap.LinearExplainer(ridge, X_scaled)
        shap_values = explainer.shap_values(X_scaled)
        mean_abs = np.abs(shap_values).mean(axis=0)
        drivers = [
            {
                "feature": name,
                "mean_abs_shap": float(v),
                "coefficient": float(ridge.coef_[i]),
            }
            for i, (name, v) in enumerate(zip(feature_names, mean_abs))
        ]
        drivers.sort(key=lambda d: d["mean_abs_shap"], reverse=True)
        return {
            "model_kind": "linear",
            "n_samples": len(sample),
            "drivers": drivers,
            "shap_values": shap_values,
            "feature_names": feature_names,
            "X_scaled": X_scaled,
        }

    X = sample[model.driver_columns].values
    explainer = shap.TreeExplainer(model.model)
    shap_values = explainer.shap_values(X)
    mean_abs = np.abs(shap_values).mean(axis=0)
    drivers = [
        {"feature": name, "mean_abs_shap": float(v)}
        for name, v in zip(feature_names, mean_abs)
    ]
    drivers.sort(key=lambda d: d["mean_abs_shap"], reverse=True)
    return {
        "model_kind": "gbm",
        "n_samples": len(sample),
        "drivers": drivers,
        "shap_values": shap_values,
        "feature_names": feature_names,
        "X": X,
    }


def plot_shap_summary(
    explanation: dict[str, Any],
    title: str,
    out_path: Path,
) -> Path:
    """Write SHAP summary bar plot."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    drivers = explanation["drivers"]
    names = [d["feature"] for d in drivers]
    values = [d["mean_abs_shap"] for d in drivers]
    ax.barh(names[::-1], values[::-1], color="tab:blue", alpha=0.85)
    ax.set_xlabel("Mean |SHAP|")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_case_study_json(
    turbine_id: str,
    raw_dir: Path = DATA_RAW,
    results_dir: Path = RESULTS_DIR,
    detector: str = PHYSICS_HYBRID_DETECTOR,
    buffer_days: int = DEFAULT_BUFFER_DAYS,
) -> dict[str, Any]:
    """Build JSON sidecar for failure case study."""
    turbines, failures = load_edp_dataset(raw_dir)
    if turbine_id not in turbines:
        return {"error": f"turbine {turbine_id} not found"}

    raw_df = turbines[turbine_id]
    df = clean_turbine_df(raw_df, min_power_kw=THERMAL_MIN_POWER_KW)
    failure = get_failure_for_turbine(turbine_id, failures)
    if failure is None:
        return {"error": f"turbine {turbine_id} has no failure"}

    failure_time = to_utc(failure.timestamp)
    train_mask = healthy_training_mask(df.index, failure, buffer_days)
    train_df = df.loc[train_mask]

    score_path = results_dir / detector / f"{turbine_id}_scores.parquet"
    metrics_path = results_dir / detector / "metrics.json"
    if not score_path.exists() or not metrics_path.exists():
        return {"error": "missing hybrid scores; run baselines first"}

    scores = pd.read_parquet(score_path).iloc[:, 0]
    scores.index = pd.to_datetime(scores.index, utc=True)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    threshold = metrics.get("thresholds", {}).get(turbine_id)
    score_start = train_df.index[-1] if not train_df.empty else None

    eval_result = evaluate_turbine(
        turbine_id,
        scores,
        threshold=float(threshold),
        failure=failure,
        score_start=score_start,
    )

    model, _, _, _ = fit_gearbox_thermal_with_selection(
        train_df, "Gear_Bear_Temp_Avg"
    )
    residual_frame = model.residual_frame(df)

    return {
        "turbine_id": turbine_id,
        "failure_time": failure_time.isoformat(),
        "failure_remarks": failure.remarks,
        "first_alarm_time": (
            eval_result.first_alarm_time.isoformat()
            if eval_result.first_alarm_time is not None
            else None
        ),
        "lead_time_days": eval_result.lead_time_days,
        "successful_warning": eval_result.successful_warning,
        "threshold": threshold,
        "train_end": train_df.index[-1].isoformat() if not train_df.empty else None,
        "residual_drift_90d_before_failure": _residual_drift_before_failure(
            residual_frame, failure_time, days=90
        ),
    }


def _residual_drift_before_failure(
    residual_frame: pd.DataFrame,
    failure_time: pd.Timestamp,
    days: int = 90,
) -> float:
    window_start = failure_time - pd.Timedelta(days=days)
    window = residual_frame.loc[window_start:failure_time]
    if window.empty:
        return 0.0
    early = window.iloc[: len(window) // 2]["residual"].mean()
    late = window.iloc[len(window) // 2 :]["residual"].mean()
    return float(late - early)


def run_thermal_interpretability(
    raw_dir: Path = DATA_RAW,
    results_dir: Path = RESULTS_DIR,
    out_dir: Path = INTERPRETABILITY_RESULTS_DIR,
    buffer_days: int = DEFAULT_BUFFER_DAYS,
    case_study_turbine: str = "T06",
) -> dict[str, Any]:
    """Run SHAP/coefficient analysis and failure case study."""
    out_dir.mkdir(parents=True, exist_ok=True)
    turbines, failures = load_edp_dataset(raw_dir)
    summary: dict[str, Any] = {"turbines": {}, "case_study": {}}

    for turbine_id, raw_df in sorted(turbines.items()):
        df = clean_turbine_df(raw_df, min_power_kw=THERMAL_MIN_POWER_KW)
        failure = get_failure_for_turbine(turbine_id, failures)
        healthy_mask = healthy_training_mask(df.index, failure, buffer_days)
        healthy_df = df.loc[healthy_mask]
        if len(healthy_df) < 100:
            continue

        turbine_summary: dict[str, Any] = {}
        for target_col in THERMAL_TARGET_COLUMNS:
            slug = TARGET_SLUGS[target_col]
            model, _, _, info = fit_gearbox_thermal_with_selection(
                healthy_df, target_col
            )
            explanation = explain_thermal_model(model, healthy_df)
            drivers_path = out_dir / f"{turbine_id}_{slug}_drivers.json"
            drivers_payload = {
                "turbine_id": turbine_id,
                "target": target_col,
                "model_kind": explanation["model_kind"],
                "selection": info,
                "drivers": explanation["drivers"],
            }
            drivers_path.write_text(json.dumps(drivers_payload, indent=2), encoding="utf-8")

            plot_path = out_dir / f"{turbine_id}_{slug}_shap_summary.png"
            plot_shap_summary(
                explanation,
                title=f"{turbine_id} — {slug} temp drivers",
                out_path=plot_path,
            )
            turbine_summary[target_col] = {
                "drivers_json": str(drivers_path),
                "shap_plot": str(plot_path),
                "top_driver": explanation["drivers"][0]["feature"]
                if explanation["drivers"]
                else None,
            }

        summary["turbines"][turbine_id] = turbine_summary

    case_json = build_case_study_json(
        case_study_turbine, raw_dir=raw_dir, results_dir=results_dir
    )
    case_json_path = out_dir / f"case_study_{case_study_turbine}.json"
    case_json_path.write_text(json.dumps(case_json, indent=2), encoding="utf-8")
    summary["case_study"]["json"] = str(case_json_path)

    if "error" not in case_json:
        plot_path = plot_failure_case_study(
            case_study_turbine,
            raw_dir=raw_dir,
            results_dir=results_dir,
            lookback_days=120,
        )
        summary["case_study"]["plot"] = str(plot_path)

    summary_path = out_dir / "interpretability_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


__all__ = [
    "build_case_study_json",
    "explain_thermal_model",
    "plot_shap_summary",
    "run_thermal_interpretability",
]
