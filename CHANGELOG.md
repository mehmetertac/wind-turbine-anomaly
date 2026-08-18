# Changelog

All notable changes to this project are documented here.

## [0.1.0] — 2026-08-18

First public release: EDP gearbox failure-warning benchmark with pure-ML baselines and physics-residual hybrid.

### Added

- EDP data loader, SCADA cleaning, synthetic EDP generator for pipeline development
- Pure-ML baselines: Isolation Forest, dense autoencoder, LSTM autoencoder
- Physics-residual hybrid: per-turbine thermal model + residual-feature Isolation Forest
- Rolling-origin evaluation protocol (lead time, false alarms/turbine-year, threshold sweep)
- Robustness pass (multi-turbine, seasonal residuals, leakage audit)
- Thermal interpretability (SHAP drivers, T06 annotated failure case study)
- Blog post: [docs/BLOG.md](docs/BLOG.md)
- Benchmark figures in `docs/assets/` (regenerate locally via scripts; `results/` gitignored)

### Benchmark headline (synthetic EDP @ 99th-percentile threshold)

- **Physics-residual hybrid:** 65 days median lead time, 8.5 false alarms/turbine-year
- **Best pure ML (LSTM-AE):** 80 days median lead time, 31 false alarms/turbine-year

Re-run on real EDP CSVs before comparing to published Hack the Wind benchmarks.
