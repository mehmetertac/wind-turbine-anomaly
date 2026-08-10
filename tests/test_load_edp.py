"""Tests for EDP data loader."""

from pathlib import Path

import pytest

from wind_turbine_anomaly.data.load_edp import (
    extract_gearbox_failures,
    load_edp_dataset,
    load_failure_logs,
    load_signal_csvs,
    split_by_turbine,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_load_signal_csvs():
    df = load_signal_csvs(FIXTURES)
    assert "Turbine_ID" in df.columns
    assert len(df) >= 9
    assert df["Timestamp"].dt.tz is not None


def test_split_by_turbine():
    signals = load_signal_csvs(FIXTURES)
    turbines = split_by_turbine(signals)
    assert "T01" in turbines
    assert "T06" in turbines
    assert turbines["T01"].index.is_monotonic_increasing


def test_extract_gearbox_failures():
    failures = load_failure_logs(FIXTURES)
    gearbox = extract_gearbox_failures(failures)
    ids = {g.turbine_id for g in gearbox}
    assert "T01" in ids
    assert "T06" in ids
    assert all(g.remarks for g in gearbox)


def test_load_edp_dataset():
    turbines, gearbox = load_edp_dataset(FIXTURES)
    assert len(turbines) == 2
    assert len(gearbox) == 2


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_signal_csvs(tmp_path)
