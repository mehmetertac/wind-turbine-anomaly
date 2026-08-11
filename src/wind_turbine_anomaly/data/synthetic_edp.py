"""Synthetic EDP-shaped SCADA and failure logs for local development."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from wind_turbine_anomaly.config import FEATURE_COLUMNS, GEARBOX_FAILURES
from wind_turbine_anomaly.utils import to_utc

SYNTHETIC_TURBINES = ["T01", "T06", "T07", "T11"]
SYNTHETIC_MARKER = ".synthetic_edp"


@dataclass(frozen=True)
class TurbineProfile:
    """Per-turbine operating offsets for synthetic SCADA."""

    turbine_id: str
    wind_scale: float
    power_scale: float
    gear_base_oil: float
    gear_base_bear: float
    failure_time: datetime | None = None
    degradation_onset_days: float = 75.0
    max_extra_gear_temp: float = 7.0


def _default_profiles() -> list[TurbineProfile]:
    """Build turbine profiles aligned with documented EDP gearbox failures."""
    failure_lookup = {
        entry["turbine_id"]: datetime.fromisoformat(entry["timestamp"])
        for entry in GEARBOX_FAILURES
        if entry["turbine_id"] in SYNTHETIC_TURBINES
    }
    profiles = []
    for tid in SYNTHETIC_TURBINES:
        ft = failure_lookup.get(tid)
        profiles.append(
            TurbineProfile(
                turbine_id=tid,
                wind_scale=1.0 + 0.05 * (SYNTHETIC_TURBINES.index(tid)),
                power_scale=1.0 + 0.03 * (SYNTHETIC_TURBINES.index(tid)),
                gear_base_oil=44.0 + SYNTHETIC_TURBINES.index(tid),
                gear_base_bear=49.0 + SYNTHETIC_TURBINES.index(tid),
                failure_time=ft,
                degradation_onset_days=80.0 if tid == "T06" else 55.0,
                max_extra_gear_temp=9.0 if tid == "T06" else 6.0,
            )
        )
    return profiles


def _power_from_wind(wind_ms: np.ndarray, rated_kw: float = 2000.0) -> np.ndarray:
    """Simple piecewise-linear power curve (cut-in 4, rated 12, cut-out 25 m/s)."""
    power = np.zeros_like(wind_ms)
    mask = (wind_ms >= 4.0) & (wind_ms < 12.0)
    power[mask] = rated_kw * ((wind_ms[mask] - 4.0) / 8.0) ** 3
    power[(wind_ms >= 12.0) & (wind_ms <= 25.0)] = rated_kw
    return np.clip(power, 0.0, rated_kw)


def _degradation_ramp(
    timestamps: pd.DatetimeIndex,
    failure_time: datetime | None,
    onset_days: float,
    max_extra: float,
) -> np.ndarray:
    """Return additive gearbox temperature ramp before failure."""
    if failure_time is None:
        return np.zeros(len(timestamps))

    failure_ts = to_utc(failure_time)
    onset = failure_ts - pd.Timedelta(days=onset_days)
    ts = pd.DatetimeIndex(timestamps)
    ramp = np.zeros(len(ts))
    mask = (ts >= onset) & (ts <= failure_ts)
    if mask.any():
        frac = (ts[mask] - onset) / (failure_ts - onset)
        ramp[mask] = max_extra * np.power(frac.to_numpy(), 1.5)
    return ramp


def generate_turbine_signals(
    profile: TurbineProfile,
    start: str = "2016-01-01",
    end: str = "2018-01-01",
    freq: str = "10min",
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate one turbine's multivariate SCADA time series."""
    rng = np.random.default_rng(
        random_state + sum(ord(c) for c in profile.turbine_id)
    )
    index = pd.date_range(start, end, freq=freq, tz="UTC", inclusive="left")
    n = len(index)
    hours = index.hour + index.dayofyear / 365.25
    season = np.sin(2 * np.pi * index.dayofyear / 365.25)

    wind = np.clip(
        profile.wind_scale * (7.5 + 3.0 * season + 1.5 * np.sin(2 * np.pi * hours / 24))
        + rng.normal(0, 0.8, n),
        0.0,
        25.0,
    )
    power = _power_from_wind(wind) * profile.power_scale
    power *= rng.normal(1.0, 0.03, n)
    power = np.clip(power, 0.0, 2000.0)

    rotor_rpm = np.where(power > 50, 8.0 + 0.004 * power + rng.normal(0, 0.2, n), 0.0)
    rotor_rpm = np.clip(rotor_rpm, 0.0, 20.0)

    nacelle = 12.0 + 8.0 * season + 0.01 * power + rng.normal(0, 0.4, n)
    wind_dir = np.mod(
        180.0 + 40.0 * season + np.cumsum(rng.normal(0, 2.0, n)),
        360.0,
    )

    load_factor = power / 2000.0
    gear_oil = (
        profile.gear_base_oil
        + 8.0 * load_factor
        + 0.15 * nacelle
        + rng.normal(0, 0.25, n)
    )
    gear_bear = (
        profile.gear_base_bear
        + 10.0 * load_factor
        + 0.12 * nacelle
        + rng.normal(0, 0.3, n)
    )

    extra = _degradation_ramp(
        index,
        profile.failure_time,
        profile.degradation_onset_days,
        profile.max_extra_gear_temp,
    )
    gear_oil += extra
    gear_bear += extra * 1.15

    df = pd.DataFrame(
        {
            "Timestamp": index,
            "Turbine_ID": profile.turbine_id,
            "Gear_Oil_Temp_Avg": gear_oil,
            "Gear_Bear_Temp_Avg": gear_bear,
            "Grd_Prod_Pwr_Avg": power,
            "Rtr_RPM_Avg": rotor_rpm,
            "Amb_WindSpeed_Avg": wind,
            "Amb_WindDir_Relative_Avg": wind_dir,
            "Nac_Temp_Avg": nacelle,
        }
    )
    return df


def generate_failure_logs(profiles: list[TurbineProfile]) -> pd.DataFrame:
    """Build failure log rows for synthetic turbines."""
    rows = []
    extras = [
        ("T06", "2016-07-11T19:48:00+00:00", "GENERATOR", "Generator replaced"),
        ("T07", "2016-08-23T02:21:00+00:00", "TRANSFORMER", "High temperature transformer"),
        ("T11", "2016-10-17T17:44:00+00:00", "HYDRAULIC_GROUP", "Hydraulic group error"),
    ]
    for profile in profiles:
        if profile.failure_time is None:
            continue
        ts = to_utc(profile.failure_time)
        remarks = "Gearbox bearings damaged" if profile.turbine_id == "T06" else "Gearbox pump damaged"
        rows.append(
            {
                "Timestamp": ts.isoformat(),
                "Turbine_ID": profile.turbine_id,
                "Component": "GEARBOX",
                "Remarks": remarks,
            }
        )
    for tid, ts, component, remarks in extras:
        if tid in SYNTHETIC_TURBINES:
            rows.append(
                {
                    "Timestamp": ts,
                    "Turbine_ID": tid,
                    "Component": component,
                    "Remarks": remarks,
                }
            )
    return pd.DataFrame(rows).sort_values("Timestamp")


def generate_synthetic_edp_dataset(
    output_dir: Path | str,
    random_state: int = 42,
) -> dict[str, Path]:
    """
    Write synthetic EDP CSV files to output_dir.

    Returns mapping of canonical filename -> path written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    profiles = _default_profiles()
    all_signals = pd.concat(
        [
            generate_turbine_signals(p, random_state=random_state)
            for p in profiles
        ],
        ignore_index=True,
    )
    all_signals["Timestamp"] = pd.to_datetime(all_signals["Timestamp"], utc=True)

    signals_2016 = all_signals[all_signals["Timestamp"].dt.year == 2016]
    signals_2017 = all_signals[all_signals["Timestamp"].dt.year == 2017]

    failures = generate_failure_logs(profiles)
    failures["Timestamp"] = pd.to_datetime(failures["Timestamp"], utc=True)
    failures_2016 = failures[failures["Timestamp"].dt.year == 2016]
    failures_2017 = failures[failures["Timestamp"].dt.year == 2017]

    paths = {
        "wind-farm-1-signals-2016.csv": output_dir / "wind-farm-1-signals-2016.csv",
        "wind-farm-1-signals-2017.csv": output_dir / "wind-farm-1-signals-2017.csv",
        "htw-failures-2016.csv": output_dir / "htw-failures-2016.csv",
        "htw-failures-2017.csv": output_dir / "htw-failures-2017.csv",
    }
    signals_2016.to_csv(paths["wind-farm-1-signals-2016.csv"], index=False)
    signals_2017.to_csv(paths["wind-farm-1-signals-2017.csv"], index=False)
    failures_2016.to_csv(paths["htw-failures-2016.csv"], index=False)
    failures_2017.to_csv(paths["htw-failures-2017.csv"], index=False)

    marker = output_dir / SYNTHETIC_MARKER
    marker.write_text(
        "Synthetic EDP-shaped data for local development. Not for publication metrics.\n"
        f"Generated at {datetime.now(timezone.utc).isoformat()}\n",
        encoding="utf-8",
    )
    return paths


def is_synthetic_dataset(data_dir: Path | str) -> bool:
    """True when data_dir contains the synthetic marker file."""
    return (Path(data_dir) / SYNTHETIC_MARKER).exists()
