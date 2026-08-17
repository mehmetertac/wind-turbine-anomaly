# Gearbox thermal normal-behavior model (Days 1–3)

Physics-informed residual track. Day 1 builds the normal-behavior thermal model; Days 2–3 run ML on the residual signal and benchmark against pure ML.

## Physical relationship

Steady-state gearbox (oil/bearing) temperature is modeled as:

```
T_gear ≈ T_amb + α·P + β·RPM + γ·P² + ε
```

| Term | SCADA proxy | Rationale |
|------|-------------|-----------|
| T_amb | `Nac_Temp_Avg` | Nacelle interior tracks ambient plus self-heating; no met-mast ambient wired in yet |
| P | `Grd_Prod_Pwr_Avg` | Gearbox losses scale with load/torque |
| RPM | `Rtr_RPM_Avg` | Speed-dependent friction and windage |
| P² | derived | Captures nonlinear I²R / load-squared heating without a complex thermal network |

**Not available:** oil flow rate. EDP SCADA in this project has oil *temperature* only (`Gear_Oil_Temp_Avg`), not flow. Documented as a limitation; cooling effectiveness shifts are partially absorbed by the nacelle proxy.

## Residual definition

```
residual = T_predicted − T_actual
```

- **Negative residual** → actual hotter than expected (degradation / failure signal)
- **Positive residual** → actual colder than expected

During healthy operation, residuals should be small, unbiased, and uncorrelated with drivers (power, RPM, nacelle temp).

## Model implementation

- **Per-turbine** models (each turbine has different baseline offsets)
- **Targets:** `Gear_Oil_Temp_Avg` and `Gear_Bear_Temp_Avg` (separate models)
- **Drivers:** power, rotor RPM, nacelle temperature
- **Operating filter:** rows with power > 50 kW (idle/transient excluded)
- **Training:** healthy period only — strictly before failure minus 90-day buffer (same as ML baselines)
- **Validation split:** time-ordered 80/20 within healthy rows (no shuffle, no leakage)

### Model selection

Two candidates are fit on the training split and compared on held-out healthy validation:

| Candidate | Form | Selection rule |
|-----------|------|----------------|
| `linear` (default) | Ridge on `[P, RPM, Nac, P²]` | Chosen unless GBM beats it by >10% RMSE |
| `gbm` | `GradientBoostingRegressor(max_depth=3, n_estimators=100)` | Used only when validation RMSE improves enough |

On synthetic EDP, **linear** was selected for all turbines (validation RMSE ~0.25°C oil, ~0.30°C bearing).

### Validation checks

Held-out healthy validation must pass:

| Check | Criterion |
|-------|-----------|
| Small | RMSE, MAE, R² reported |
| Unbiased | \|mean(residual)\| < 0.5°C |
| Structureless vs drivers | \|r(residual, driver)\| < 0.1 for each driver |
| Structureless vs time | \|linear trend\| < 0.01°C/day |

Warnings are logged if checks fail; residuals are still emitted for exploration.

## How to run

```bash
# Synthetic data (if EDP portal unavailable)
python scripts/generate_synthetic_edp.py --force

# Fit thermal models, validate, write residuals + plots
python scripts/run_gearbox_thermal.py
```

Options: `--raw-dir`, `--buffer-days`, `--lookback-days`.

## Outputs

Under `results/physics_thermal/` (gitignored):

| File | Content |
|------|---------|
| `{turbine}_oil_residuals.parquet` | actual, predicted, residual (oil) |
| `{turbine}_bear_residuals.parquet` | actual, predicted, residual (bearing) |
| `{turbine}_{oil\|bear}_validation.json` | metrics, model choice, coefficients |
| `summary.json` | aggregate across turbines |

Plots (failure turbines): `results/plots/residual_{T01,T06}_{oil,bear}.png`

## Synthetic validation results

Before known gearbox failures, residuals show clear negative drift (hotter than expected):

| Turbine | Target | 90d early mean | 90d late mean | Drift |
|---------|--------|----------------|---------------|-------|
| T01 | oil | −0.02°C | −2.40°C | −2.38°C |
| T01 | bear | −0.01°C | −2.77°C | −2.76°C |
| T06 | oil | −0.15°C | −3.66°C | −3.51°C |
| T06 | bear | −0.18°C | −4.23°C | −4.04°C |

Healthy turbines (T07, T11) show no comparable pre-failure drift — residuals stay near zero on validation.

This drift is the signal raw-temperature detectors struggle to isolate from operating-point changes (power swings, seasonal ambient).

## Code locations

| Component | Path |
|-----------|------|
| Config constants | `src/wind_turbine_anomaly/config.py` |
| Thermal model | `src/wind_turbine_anomaly/models/gearbox_thermal.py` |
| Residual features | `src/wind_turbine_anomaly/models/residual_features.py` |
| Physics hybrid detector | `src/wind_turbine_anomaly/models/physics_hybrid.py` |
| Hybrid comparison | `src/wind_turbine_anomaly/eval/hybrid_comparison.py` |
| CLI (Day 1) | `scripts/run_gearbox_thermal.py` |
| CLI (hybrid baseline) | `scripts/run_physics_hybrid_baseline.py` |
| Residual plots | `src/wind_turbine_anomaly/eval/plots.py` |
| Tests | `tests/test_gearbox_thermal.py`, `tests/test_physics_hybrid.py` |

## Days 2–3: ML on residuals (`physics_hybrid`)

The hybrid detector strips operating-point variance with the thermal model, then runs **Isolation Forest** on residual-window features:

| Feature group | Content |
|---------------|---------|
| Point degradation | `-residual` for oil and bearing (negative residual = hotter than expected) |
| EWMA | ~6h smoothed degradation signal (control-chart style) |
| Rolling stats | Mean and std at 1h / 6h / 24h windows |

Training and scoring use the same rolling-origin protocol as pure ML (90-day buffer, 99th-percentile threshold, 6-sample persistence). Operating rows only (power > 50 kW).

### How to run

```bash
python scripts/run_physics_hybrid_baseline.py
# Or full benchmark (pure ML + hybrid + comparison):
python scripts/run_all_ml_baselines.py
python scripts/run_hybrid_comparison.py
```

### Outputs

| File | Content |
|------|---------|
| `results/physics_hybrid/metrics.json` | Per-turbine lead time, false alarms |
| `results/physics_hybrid/{turbine}_scores.parquet` | Anomaly score time series |
| `results/metrics.csv` | Master table including `physics_hybrid` row |
| `results/hybrid_vs_ml_summary.json` | Head-to-head lead time and false-alarm deltas |
| `results/hybrid_vs_ml_regime.json` | False alarms tagged by high-load / hot-ambient regime |
| `results/plots/hybrid_vs_ml_comparison.png` | Bar charts: lead time, false alarms, regime breakdown |

### Expected hybrid advantage

Pure ML on raw SCADA fires during benign high-load or hot-ambient periods because temperature rises with power and ambient. The physics model explains that variance; the residual stream isolates degradation drift. On synthetic data, expect **fewer false alarms** (especially in `high_load` / `hot_ambient` regimes) and **earlier or equal lead time** on T01/T06 vs best pure-ML detector.

## Optional next steps

- Met-mast ambient temperature instead of nacelle proxy
- Horizon sweep (7, 14, 30, 60, 90 days) for successful-warning reporting

## N-days-ahead failure prediction framing

The physics-residual track is not an outlier detector on raw temperature — it is a **failure-warning system** measured in days.

### Physical precursor → alarm

1. **Thermal model** predicts expected oil/bearing temperature from power, RPM, and nacelle proxy.
2. **Negative residual drift** (actual hotter than expected) is the degradation signal — see synthetic validation table above (T01/T06 drift of −2 to −4°C before failure).
3. **EWMA + rolling features** smooth the drift into a sustained anomaly score; **Isolation Forest** flags when the residual pattern departs from healthy training.
4. **Alarm → failure gap** = lead time (days). **Successful warning @ N** = lead time ≥ N (default N = 30, aligned with Hack the Wind / Frontiers 2022 buffer).

### Why physics helps the trade-off

Pure ML on raw SCADA fires during benign high-load or hot-ambient periods because temperature rises with power and ambient. The thermal model explains that variance; the residual stream isolates degradation drift. Regime analysis in `results/hybrid_vs_ml_regime.json` quantifies false alarms during `high_load` / `hot_ambient` operating points — the regimes where raw-temperature detectors struggle most.

At a lower threshold (more sensitive), the hybrid should add fewer spurious alarms than pure ML because operating-point variance is already stripped.

### Headline claim and threshold sweep

Run the full benchmark (includes sweep):

```bash
python scripts/run_all_ml_baselines.py
# Or sweep only (requires baselines + train_scores parquets):
python scripts/run_threshold_sweep.py
```

| Output | Content |
|--------|---------|
| `results/headline_claim.json` | Project headline: median lead time + false alarms/turbine-year @ 99th-percentile threshold |
| `results/threshold_sweep.csv` | Full sweep grid for hybrid vs best pure ML |
| `results/plots/threshold_tradeoff.png` | Lead time vs false-alarm trade-off curve (★ = default operating point) |

Example headline (synthetic EDP, regenerate locally):

> At the 99th-percentile threshold, physics_hybrid flags gearbox failures a median of **65 days** in advance at **8.5 false alarms per turbine-year**.

Compare to best pure ML (LSTM-AE on synthetic): ~80 days median lead time at ~31 false alarms/turbine-year — hybrid trades some lead time for a much lower false-alarm rate. **Re-run on real EDP before publication.**
