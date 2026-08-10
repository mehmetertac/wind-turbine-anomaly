# Evaluation protocol

Rolling-origin, strictly time-ordered evaluation for gearbox anomaly detection. Reused across pure ML and physics-residual hybrid tracks.

See also: [DATA.md](DATA.md) | [AGENT.md](../AGENT.md)

## Principles

1. **Train on healthy history only** — exclude the failure window and all post-failure data from training.
2. **Score forward in time** — apply the model chronologically from the end of training through failure (and beyond for false-alarm counting).
3. **No random splits** — no shuffling, no pooling failure turbines into training.
4. **Probabilistic-first** — report lead time distributions and false-alarm rates, not a single binary flag.

## Training window

For turbine with gearbox failure at time `F`:

```
train_mask = timestamp < (F - buffer_days)
```

Default `buffer_days = 90` (aligned with Hack the Wind / Frontiers 2022 lead-time horizon).

For turbines without logged gearbox failure: train on first 90% of data; score full series for false-alarm rate only.

## Scoring window

Score all timestamps **from end of training** through end of available data. Do not retrain during the test window in Task 1 (single-origin baseline).

## Alarm rule

Implemented in [`src/wind_turbine_anomaly/eval/protocol.py`](../src/wind_turbine_anomaly/eval/protocol.py):

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `threshold` | 99th percentile of **training** anomaly scores | Score above this triggers candidate alarm |
| `persistence_samples` | 6 | Consecutive 10-min samples above threshold (1 hour) |
| `cooldown_hours` | 24 | Minimum gap between distinct alarm episodes |

## Metrics

### Lead time (days)

```
lead_time = failure_time - first_alarm_time
```

Computed only for alarms **before** the logged failure. `NaN` if no qualifying alarm.

### Successful warning

`lead_time >= N` where `N` is the prediction horizon (default 30 days; sweep 7, 14, 30, 60, 90).

### False alarms per turbine-year

For each alarm episode at time `t`:

- **False alarm** if no gearbox failure occurs in `[t, t + N]`.
- Turbines without logged gearbox failure: every episode is a false alarm.

```
false_alarms_per_turbine_year = total_false_alarms / (total_scored_days / 365.25)
```

### Precision@N

Fraction of alarm episodes where a gearbox failure follows within `N` days.

## Task 1 targets

| Turbine | Failure | Expected role |
|---------|---------|---------------|
| T01 | 2016-07-18 pump damage | Lead-time case (literature ~21 days with CUSUM) |
| T06 | 2017-10-17 bearing damage | Flagship wear-out case (literature ~89 days) |
| T07, T11 | None | False-alarm rate only |

## Resampling note

Literature often aggregates 10-min SCADA to **1-hour** averages before detection. Task 1 keeps native 10-min resolution; compare fairly when benchmarking against published results.

## Output

Baseline metrics written to `results/isolation_forest/metrics.json` by the notebook / pipeline.
