# wind-turbine-anomaly

Predict gearbox failures **N days ahead** on [EDP open wind-turbine SCADA](https://www.edp.com/en/innovation/open-data/data). Compare **pure ML anomaly detection** (Isolation Forest → autoencoder → LSTM-AE) against a **physics-residual + ML hybrid** — and show the hybrid wins.

## Problem

Gearbox failures are among the costliest wind-turbine events: crane mobilization, long downtime, and high replacement cost. SCADA records temperatures, power, and speeds every 10 minutes. The question is not “is this point an outlier?” but **how many days of warning** you get before a confirmed failure, and at what **false-alarm rate**.

## Approaches

| Track | Method | Role |
|-------|--------|------|
| Pure ML (Days 1–2) | Isolation Forest, autoencoder, LSTM-AE | Multivariate baseline on raw SCADA |
| Hybrid (Days 2–3) | Physics-informed residual + IF | Model expected gearbox temp; detect regime shifts in residuals via EWMA/rolling features + Isolation Forest |

Current status: **All three pure-ML baselines** and **physics-residual hybrid (Days 1–3)** implemented. Run `python scripts/run_all_ml_baselines.py` for the full benchmark including `physics_hybrid` in `results/metrics.csv`. See [`docs/PHYSICS_THERMAL.md`](docs/PHYSICS_THERMAL.md).

## What this means in maintenance terms

A detection system is useful only if it changes **when** you dispatch a crew — not whether a dot on a chart turns red.

### The metrics that matter

- **Lead time (days):** Days between the first sustained alarm and the logged gearbox failure. More lead time means you can schedule inspection during low-wind windows, order parts, and avoid emergency crane calls.
- **False alarms per turbine-year:** How often the system cries wolf when no failure follows within N days (default N = 30). Each false alarm costs a truck roll, lost production from conservative curtailment, and credibility with site managers.
- **Trade-off:** Lowering the anomaly threshold catches failures earlier but increases false alarms. Maintenance planning is choosing a point on that curve — similar to setting inspection intervals on bearing vibration trends. See `results/plots/threshold_tradeoff.png` after running baselines.

### Headline claim (synthetic EDP — illustrative)

After `python scripts/run_all_ml_baselines.py`, see `results/headline_claim.json`:

> At the 99th-percentile threshold, **physics_hybrid** flags gearbox failures a median of **65 days** in advance at **8.5 false alarms per turbine-year**.

Best pure ML (LSTM-AE) at the same point: **80 days** median lead time at **31 false alarms/turbine-year**. The hybrid trades some lead time for a much lower false-alarm rate — the operating point a site manager would prefer when truck rolls are expensive. **Re-run on real EDP before publication.**

### What N days of warning buys you

| Scenario | Typical cost | Notes |
|----------|--------------|-------|
| Emergency gearbox replacement + crane mobilization | €50k–150k + weeks downtime | Unplanned; crane availability and wind windows drive schedule |
| Planned gearbox inspection (borescope, oil sample, vibration) | €3k–8k | Scheduled during low-wind period; parts can be pre-ordered |
| Gearbox replacement (planned) | ~€100k | Same hardware cost, but downtime is controlled |

**65 days of warning** converts an unplanned crane mobilization into a planned intervention: order the crane for a known low-wind week, pull oil samples first, and confirm the fault before committing to a full replacement.

### False-alarm economics

Each false alarm triggers an inspection that finds nothing wrong. At ~€5k per truck roll:

- **Hybrid @ 8.5 FA/turbine-year** → ~€43k/year fleet cost (4 turbines) if every alarm is acted on
- **Best pure ML @ 31 FA/turbine-year** → ~€155k/year for the same fleet

One avoided catastrophic gearbox failure (~€100k+) pays for roughly **20 false-alarm inspections**. The break-even question is not "zero false alarms" but "few enough false alarms that the avoided failures dominate."

### Choosing an operating point

Site managers pick a point on the threshold trade-off curve (`results/threshold_sweep.csv`):

- **Stricter threshold** (higher percentile): fewer false alarms, later failure warnings — acceptable for remote sites where truck rolls are costly.
- **Looser threshold** (lower percentile): earlier warnings, more false alarms — acceptable for critical assets or sites with easy crane access.

The physics-residual hybrid reduces false alarms especially during **high-load and hot-ambient** operating regimes (see `results/hybrid_vs_ml_regime.json`) — the conditions where raw-temperature ML most often cries wolf.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -e ".[dev]"
pre-commit install

# Download data — see data/README.md
python scripts/generate_synthetic_edp.py --force   # if EDP portal unavailable
python scripts/download_edp.py --check

# Run tests
pytest -q

# Run all baselines (pure ML + physics hybrid + benchmark CSV + plots)
python scripts/run_all_ml_baselines.py

# Or run baseline notebook (requires raw data in data/raw/edp/)
jupyter notebook notebooks/01_eda_and_if_baseline.ipynb
```

## Documentation

- [Data guide](docs/DATA.md) — channels, turbines, gearbox failure events
- [Physics thermal model](docs/PHYSICS_THERMAL.md) — normal-behavior gearbox temperature (Day 1)
- [Evaluation protocol](docs/EVALUATION.md) — rolling-origin, lead time, false-alarm metrics, threshold sweep
- [Week 6 reflection](WEEK_06_REFLECTION.md) — build summary, open questions, field-relevant patterns
- [Data acquisition](data/README.md) — EDP portal and Mendeley fallback
- [Handover](handover.md) — current status, data situation, next steps
- [Agent rules](AGENT.md) — contribution and testing requirements

## License

Apache-2.0 (see [LICENSE](LICENSE)).
