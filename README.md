# wind-turbine-anomaly

Predict gearbox failures **N days ahead** on [EDP open wind-turbine SCADA](https://www.edp.com/en/innovation/open-data/data). Compare **pure ML anomaly detection** (Isolation Forest → autoencoder → LSTM-AE) against a **physics-residual + ML hybrid** — and show the hybrid wins.

## Problem

Gearbox failures are among the costliest wind-turbine events: crane mobilization, long downtime, and high replacement cost. SCADA records temperatures, power, and speeds every 10 minutes. The question is not “is this point an outlier?” but **how many days of warning** you get before a confirmed failure, and at what **false-alarm rate**.

## Approaches

| Track | Method | Role |
|-------|--------|------|
| Pure ML (Days 1–2) | Isolation Forest, autoencoder, LSTM-AE | Multivariate baseline on raw SCADA |
| Hybrid (Days 3–5) | Physics-informed residual + ML | Model expected gearbox temp from power, ambient, load; detect regime shifts in residuals |

Current status: **Isolation Forest baseline** implemented. See [`notebooks/01_eda_and_if_baseline.ipynb`](notebooks/01_eda_and_if_baseline.ipynb).

## What this means in maintenance terms

A detection system is useful only if it changes **when** you dispatch a crew — not whether a dot on a chart turns red.

- **Lead time (days)**: Days between the first sustained alarm and the logged gearbox failure. More lead time means you can schedule inspection during low-wind windows, order parts, and avoid emergency crane calls.
- **False alarms per turbine-year**: How often the system cries wolf when no failure follows within N days. Each false alarm costs a truck roll, lost production from conservative curtailment, and credibility with site managers.
- **Trade-off**: Lowering the anomaly threshold catches failures earlier but increases false alarms. Maintenance planning is choosing a point on that curve — similar to setting inspection intervals on bearing vibration trends.

Example framing (Hack the Wind / industry benchmarks): gearbox replacement ~€100k; planned inspection ~€5k. A 30-day warning with one false alarm per turbine-year is often acceptable; a 60-day warning with ten false alarms is not.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -e ".[dev]"
pre-commit install

# Download data — see data/README.md
python scripts/download_edp.py --check

# Run tests
pytest -q

# Run baseline notebook (requires raw data in data/raw/edp/)
jupyter notebook notebooks/01_eda_and_if_baseline.ipynb
```

## Documentation

- [Data guide](docs/DATA.md) — channels, turbines, gearbox failure events
- [Evaluation protocol](docs/EVALUATION.md) — rolling-origin, lead time, false-alarm metrics
- [Data acquisition](data/README.md) — EDP portal and Mendeley fallback
- [Agent rules](AGENT.md) — contribution and testing requirements

## License

Apache-2.0 (see [LICENSE](LICENSE)).
