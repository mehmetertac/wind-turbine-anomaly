"""Load EDP open wind-farm SCADA signals and failure logs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from wind_turbine_anomaly.config import (
    FAILURE_FILE_ALIASES,
    GearboxFailure,
    SIGNAL_FILE_ALIASES,
)


def _resolve_file(data_dir: Path, aliases: list[str]) -> Path:
    """Return the first existing file from a list of alias names."""
    for name in aliases:
        path = data_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(
        f"None of these files found in {data_dir}: {', '.join(aliases)}"
    )


def load_signal_csvs(data_dir: Path | str) -> pd.DataFrame:
    """Load and concatenate 2016 and 2017 SCADA signal CSVs."""
    data_dir = Path(data_dir)
    path_2016 = _resolve_file(data_dir, SIGNAL_FILE_ALIASES["signals_2016"])
    path_2017 = _resolve_file(data_dir, SIGNAL_FILE_ALIASES["signals_2017"])

    frames = []
    for path in (path_2016, path_2017):
        df = pd.read_csv(path, low_memory=False)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["Turbine_ID", "Timestamp"])
    return combined


def split_by_turbine(signals: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split combined signals into per-turbine DataFrames indexed by timestamp."""
    result: dict[str, pd.DataFrame] = {}
    for turbine_id, group in signals.groupby("Turbine_ID"):
        df = group.drop(columns=["Turbine_ID"]).copy()
        df = df.set_index("Timestamp")
        df = df[~df.index.duplicated(keep="first")].sort_index()
        result[str(turbine_id)] = df
    return result


def load_failure_logs(data_dir: Path | str) -> pd.DataFrame:
    """Load and concatenate failure log CSVs."""
    data_dir = Path(data_dir)
    path_2016 = _resolve_file(data_dir, FAILURE_FILE_ALIASES["failures_2016"])
    path_2017 = _resolve_file(data_dir, FAILURE_FILE_ALIASES["failures_2017"])

    frames = []
    for path in (path_2016, path_2017):
        df = pd.read_csv(path)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values("Timestamp")


def extract_gearbox_failures(failures: pd.DataFrame) -> list[GearboxFailure]:
    """Filter failure log for gearbox component events."""
    mask = failures["Component"].astype(str).str.upper() == "GEARBOX"
    gearbox = failures.loc[mask].copy()
    events: list[GearboxFailure] = []
    for _, row in gearbox.iterrows():
        events.append(
            GearboxFailure(
                turbine_id=str(row["Turbine_ID"]),
                timestamp=row["Timestamp"].to_pydatetime(),
                remarks=str(row.get("Remarks", "")),
            )
        )
    return events


def load_edp_dataset(data_dir: Path | str) -> tuple[dict[str, pd.DataFrame], list[GearboxFailure]]:
    """Load SCADA signals and gearbox failure events from EDP raw directory."""
    signals = load_signal_csvs(data_dir)
    turbines = split_by_turbine(signals)
    failures = load_failure_logs(data_dir)
    gearbox_failures = extract_gearbox_failures(failures)
    return turbines, gearbox_failures
