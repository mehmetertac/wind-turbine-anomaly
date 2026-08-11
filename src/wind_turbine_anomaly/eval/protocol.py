"""Lead-time and false-alarm evaluation protocol."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import timedelta

import numpy as np
import pandas as pd

from wind_turbine_anomaly.utils import to_utc

from wind_turbine_anomaly.config import (
    DEFAULT_COOLDOWN_HOURS,
    DEFAULT_HORIZON_DAYS,
    DEFAULT_PERSISTENCE_SAMPLES,
    GearboxFailure,
)


@dataclass
class AlarmEpisode:
    """Single sustained alarm episode."""

    start: pd.Timestamp
    end: pd.Timestamp


@dataclass
class TurbineEvalResult:
    """Evaluation metrics for one turbine."""

    turbine_id: str
    has_gearbox_failure: bool
    failure_time: pd.Timestamp | None
    first_alarm_time: pd.Timestamp | None
    lead_time_days: float | None
    successful_warning: bool
    false_alarm_episodes: int
    scored_days: float
    precision_at_horizon: float | None


def detect_alarm_episodes(
    scores: pd.Series,
    threshold: float,
    persistence_samples: int = DEFAULT_PERSISTENCE_SAMPLES,
    cooldown_hours: float = DEFAULT_COOLDOWN_HOURS,
) -> list[AlarmEpisode]:
    """
    Detect alarm episodes from anomaly scores.

    Alarm triggers when score exceeds threshold for `persistence_samples`
    consecutive samples. Episodes separated by at least `cooldown_hours`.
    """
    if scores.empty:
        return []

    above = scores.values > threshold
    episodes: list[AlarmEpisode] = []
    cooldown = timedelta(hours=cooldown_hours)
    i = 0
    n = len(scores)

    while i <= n - persistence_samples:
        window = above[i : i + persistence_samples]
        if window.all():
            start = scores.index[i]
            j = i + persistence_samples
            while j < n and above[j]:
                j += 1
            end = scores.index[j - 1]
            episodes.append(AlarmEpisode(start=start, end=end))
            # Skip ahead past cooldown
            next_allowed = end + cooldown
            while i < n and scores.index[i] < next_allowed:
                i += 1
        else:
            i += 1

    return episodes


def compute_lead_time_days(
    episodes: list[AlarmEpisode],
    failure_time: pd.Timestamp,
) -> tuple[pd.Timestamp | None, float | None]:
    """Return first alarm before failure and lead time in days."""
    failure_time = to_utc(failure_time)
    pre_failure = [ep for ep in episodes if ep.start < failure_time]
    if not pre_failure:
        return None, None
    first = min(pre_failure, key=lambda ep: ep.start)
    delta = failure_time - first.start
    return first.start, delta.total_seconds() / 86400.0


def is_false_alarm(
    alarm_start: pd.Timestamp,
    failure_time: pd.Timestamp | None,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> bool:
    """
    True when no gearbox failure occurs within horizon_days after alarm_start.

    Post-failure alarms and turbines without logged failures are false alarms.
    """
    if failure_time is None:
        return True

    failure_time = to_utc(failure_time)
    alarm_start = to_utc(alarm_start)
    if alarm_start >= failure_time:
        return True

    horizon = timedelta(days=horizon_days)
    return not (alarm_start <= failure_time <= alarm_start + horizon)


def count_false_alarms(
    episodes: list[AlarmEpisode],
    failure_time: pd.Timestamp | None,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> int:
    """Count distinct alarm episodes classified as false alarms."""
    return sum(
        1
        for ep in episodes
        if is_false_alarm(ep.start, failure_time, horizon_days)
    )


def precision_at_horizon(
    episodes: list[AlarmEpisode],
    failure_time: pd.Timestamp | None,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> float | None:
    """Fraction of alarms that precede a failure within horizon_days."""
    if not episodes:
        return None
    if failure_time is None:
        return 0.0

    failure_time = to_utc(failure_time)
    horizon = timedelta(days=horizon_days)
    tp = sum(
        1
        for ep in episodes
        if failure_time - horizon <= ep.start < failure_time
    )
    return tp / len(episodes)


def evaluate_turbine(
    turbine_id: str,
    scores: pd.Series,
    threshold: float,
    failure: GearboxFailure | None,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    persistence_samples: int = DEFAULT_PERSISTENCE_SAMPLES,
    cooldown_hours: float = DEFAULT_COOLDOWN_HOURS,
    score_start: pd.Timestamp | None = None,
) -> TurbineEvalResult:
    """Evaluate one turbine's anomaly score series."""
    if score_start is not None:
        scores = scores[scores.index >= score_start]

    episodes = detect_alarm_episodes(
        scores,
        threshold=threshold,
        persistence_samples=persistence_samples,
        cooldown_hours=cooldown_hours,
    )

    failure_time = to_utc(failure.timestamp) if failure else None
    first_alarm, lead_days = (
        compute_lead_time_days(episodes, failure_time)
        if failure_time is not None
        else (None, None)
    )

    false_alarms = count_false_alarms(episodes, failure_time, horizon_days)
    scored_days = (
        (scores.index[-1] - scores.index[0]).total_seconds() / 86400.0
        if len(scores) > 1
        else 0.0
    )

    return TurbineEvalResult(
        turbine_id=turbine_id,
        has_gearbox_failure=failure is not None,
        failure_time=failure_time,
        first_alarm_time=first_alarm,
        lead_time_days=lead_days,
        successful_warning=lead_days is not None and lead_days >= horizon_days,
        false_alarm_episodes=false_alarms,
        scored_days=scored_days,
        precision_at_horizon=precision_at_horizon(
            episodes, failure_time, horizon_days
        ),
    )


def false_alarms_per_turbine_year(results: list[TurbineEvalResult]) -> float:
    """Aggregate false alarm rate across turbines."""
    total_false = sum(r.false_alarm_episodes for r in results)
    total_years = sum(r.scored_days for r in results) / 365.25
    if total_years <= 0:
        return float("nan")
    return total_false / total_years


def _json_value(value):
    """Convert metric values to JSON-serializable form."""
    if value is None:
        return None
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def results_to_dict(results: list[TurbineEvalResult], horizon_days: int) -> dict:
    """Serialize evaluation results for JSON export."""
    rate = false_alarms_per_turbine_year(results)
    return {
        "horizon_days": horizon_days,
        "false_alarms_per_turbine_year": None
        if np.isnan(rate)
        else rate,
        "turbines": [
            {k: _json_value(v) for k, v in asdict(r).items()}
            for r in results
        ],
    }
