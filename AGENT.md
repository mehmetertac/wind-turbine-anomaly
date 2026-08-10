# Agent rules

Guidelines for AI agents and contributors working in this repository.

## Code organization

- No single file should exceed **1,000 lines**. If a module grows beyond that, split it.
- Package layout lives under [`src/wind_turbine_anomaly/`](src/wind_turbine_anomaly/).

## Documentation

- **Update documentation before every push** to the repository.
- Cross-link docs so readers know where to continue:
  - [`README.md`](README.md) — project overview and maintenance narrative
  - [`docs/DATA.md`](docs/DATA.md) — EDP dataset, channels, failure events
  - [`docs/EVALUATION.md`](docs/EVALUATION.md) — rolling-origin evaluation protocol
  - [`data/README.md`](data/README.md) — how to download raw data

## Testing

- Create **at least minimal unit tests** for every change.
- Add integration tests where the project supports them (data loader, end-to-end baseline).
- **Run tests before commit or push.** Pre-commit hook runs `pytest -q` (see [`.pre-commit-config.yaml`](.pre-commit-config.yaml)).

Setup hooks:

```bash
pip install -e ".[dev]"
pre-commit install
```

## Data

- Raw EDP files are **not** committed. Place them in `data/raw/edp/` per [`data/README.md`](data/README.md).

## Evaluation principles

- Strictly **time-ordered** evaluation: train on healthy history, score forward toward failure.
- No leakage from failure periods into training.
- Report **lead time** and **false alarms per turbine-year**, not just binary flags.
