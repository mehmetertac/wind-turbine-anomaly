"""Timestamp helpers."""

from __future__ import annotations

import pandas as pd


def to_utc(ts) -> pd.Timestamp:
    """Normalize datetime-like values to UTC Timestamp."""
    stamp = pd.Timestamp(ts)
    if stamp.tzinfo is None:
        return stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC")
