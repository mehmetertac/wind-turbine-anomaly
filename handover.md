# Handover — wind-turbine-anomaly (Task 1 complete)

**Repo:** https://github.com/mehmetertac/wind-turbine-anomaly  
**Branch:** `main`  
**Last updated:** August 2026  
**Scope completed:** Task 1 — EDP data pipeline, Isolation Forest baseline, evaluation protocol; Task 2 — dense + LSTM autoencoder baselines, benchmark table; Task 3 — physics-residual hybrid (Days 1–3); **Week 6 — threshold sweep, headline claim, maintenance narrative, reflection**

---

## 1. Project goal

Predict **gearbox failures N days ahead** on wind-turbine SCADA. Compare:

1. **Pure ML** — Isolation Forest, dense autoencoder, LSTM-AE (all three implemented)
2. **Physics-residual hybrid** — expected gearbox temp from power/ambient/load; IF on residual-window features (**Days 1–3 done**)

Deliverables over the 12-week plan include this repo, a blog post, and `WEEK_06_REFLECTION.md`.

---

## 2. What is done

| Component | Location | Status |
|-----------|----------|--------|
| EDP data loader | `src/wind_turbine_anomaly/data/load_edp.py` | Done |
| SCADA cleaning + healthy mask | `src/wind_turbine_anomaly/data/clean.py` | Done |
| **Synthetic EDP generator** | `src/wind_turbine_anomaly/data/synthetic_edp.py` | Done |
| Isolation Forest baseline | `src/wind_turbine_anomaly/models/isolation_forest.py` | Done |
| Dense autoencoder baseline | `src/wind_turbine_anomaly/models/dense_autoencoder.py` | Done |
| LSTM autoencoder baseline | `src/wind_turbine_anomaly/models/lstm_autoencoder.py` | Done |
| Unified baseline runner | `src/wind_turbine_anomaly/eval/baseline_runner.py` | Done |
| Benchmark table + plots | `results/metrics.csv`, `results/plots/trajectory_*.png` | Done |
| Evaluation protocol | `src/wind_turbine_anomaly/eval/protocol.py` | Done |
| CLI: data check | `scripts/download_edp.py` | Done |
| CLI: synthetic data | `scripts/generate_synthetic_edp.py` | Done |
| CLI: baseline run | `scripts/run_if_baseline.py` | Done |
| CLI: all ML baselines | `scripts/run_all_ml_baselines.py` | Done |
| **Gearbox thermal model (Day 1)** | `models/gearbox_thermal.py`, `scripts/run_gearbox_thermal.py` | Done |
| **Physics hybrid detector (Days 2–3)** | `models/physics_hybrid.py`, `scripts/run_physics_hybrid_baseline.py` | Done |
| **Hybrid vs ML comparison** | `eval/hybrid_comparison.py`, `scripts/run_hybrid_comparison.py` | Done |
| **Threshold sweep + headline claim** | `eval/threshold_sweep.py`, `scripts/run_threshold_sweep.py` | Done |
| **Week 6 reflection** | `WEEK_06_REFLECTION.md` | Done |
| Notebook | `notebooks/01_eda_and_if_baseline.ipynb` | Done |
| Unit tests | `tests/` (41+ tests) | Passing |
| Pre-commit hook | `.pre-commit-config.yaml` | Runs `pytest -q` |

---

## 3. Data situation

### Real EDP data (preferred for publication)

- **Source:** [EDP Open Data → Wind technology](https://edp.com/en/innovation/data) (Wind Farm 1, 2016–2017)
- **Problem encountered:** Portal returned **403** for automated download; Mendeley record ([10.17632/zjxjnjp3xs](https://data.mendeley.com/datasets/zjxjnjp3xs)) is description-only and redirects to EDP.
- **Required files** (place in `data/raw/edp/`): see [data/README.md](data/README.md)

### Synthetic data (current local default)

Because real EDP was unavailable, a **synthetic generator** produces EDP-shaped CSVs:

```bash
python scripts/generate_synthetic_edp.py --force
```

- ~421k SCADA rows (2016–2017), turbines T01/T06/T07/T11
- Injected gearbox degradation before T01 (Jul 2016) and T06 (Oct 2017) failures
- Marker file: `data/raw/edp/.synthetic_edp`
- **Not for publication metrics** — use only for pipeline development until real EDP is obtained

**Alternatives if EDP stays blocked:** [Zenodo CARE To Compare](https://zenodo.org/records/15846963) (EDP-derived, needs adapter) or [Hill of Towie](https://zenodo.org/records/14870023) (different schema, 2016–2017 zips).

Raw data and `results/` are **gitignored**.

---

## 4. How to run (fresh clone)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -e ".[dev]"
pre-commit install

# Data (pick one)
python scripts/generate_synthetic_edp.py --force   # synthetic
# OR place real EDP CSVs in data/raw/edp/

python scripts/download_edp.py --check
pytest -q
python scripts/run_if_baseline.py
python scripts/run_all_ml_baselines.py   # IF + AEs + physics_hybrid + metrics.csv + comparison + threshold sweep
python scripts/run_threshold_sweep.py  # sweep only (requires baselines + train_scores parquets)
python scripts/run_gearbox_thermal.py    # physics-residual Day 1: thermal model + residuals (exploratory)
python scripts/run_physics_hybrid_baseline.py  # hybrid detector only
python scripts/run_hybrid_comparison.py  # head-to-head analysis (requires all baselines)
jupyter notebook notebooks/01_eda_and_if_baseline.ipynb
```

---

## 5. Latest baseline results (synthetic data)

Run `python scripts/run_all_ml_baselines.py` to regenerate.

**Headline claim** (`results/headline_claim.json`):

> At the 99th-percentile threshold, physics_hybrid flags gearbox failures a median of **65 days** in advance at **8.5 false alarms per turbine-year** (synthetic EDP).

Best pure ML at the same operating point: LSTM-AE — **80 days** median lead time at **31 false alarms/turbine-year**. Hybrid wins on false-alarm rate; LSTM-AE leads on median lead time on this synthetic run.

**Trade-off curve:** `results/plots/threshold_tradeoff.png` — sweep grid in `results/threshold_sweep.csv`.

Consolidated benchmark: **`results/metrics.csv`** — one row per detector with T01/T06 lead times and aggregate false-alarm rate.

Score trajectory plots (T01, T06): `results/plots/trajectory_T01.png`, `trajectory_T06.png`.

**Re-run on real EDP before drawing conclusions.**

---

## 6. Architecture (Task 1)

```
data/raw/edp/*.csv
    → load_edp.py (per-turbine split + gearbox failures)
    → clean.py (features, healthy training mask, 90-day buffer)
    → isolation_forest.py / dense_autoencoder.py / lstm_autoencoder.py
    → protocol.py (threshold, persistence, lead time, false alarms)
    → results/{detector}/metrics.json + {turbine}_scores.parquet + {turbine}_train_scores.parquet
    → results/metrics.csv (benchmark table) + results/plots/trajectory_*.png
    → results/threshold_sweep.csv + results/headline_claim.json + results/plots/threshold_tradeoff.png
```

### Key config (`src/wind_turbine_anomaly/config.py`)

- **Features:** `Gear_Oil_Temp_Avg`, `Gear_Bear_Temp_Avg`, `Grd_Prod_Pwr_Avg`, `Rtr_RPM_Avg`, `Amb_WindSpeed_Avg`, `Amb_WindDir_Relative_Avg`, `Nac_Temp_Avg`
- **Ground truth:** T01 gearbox 2016-07-18, T06 gearbox 2017-10-17
- **Defaults:** 90-day training buffer, 30-day horizon, 99th-percentile threshold, 6-sample (1h) alarm persistence

### Evaluation rules

Documented in [docs/EVALUATION.md](docs/EVALUATION.md):

- Train on healthy history only (strictly before failure − buffer)
- Score forward in time — no random splits, no leakage
- Metrics: lead time, successful warning @ N days, false alarms per turbine-year, precision@N

---

## 7. Documentation map

| Doc | Purpose |
|-----|---------|
| [README.md](README.md) | Overview + maintenance narrative |
| [AGENT.md](AGENT.md) | Agent/contributor rules (tests, doc updates before push) |
| [data/README.md](data/README.md) | Data acquisition (EDP, Mendeley, synthetic) |
| [docs/DATA.md](docs/DATA.md) | Channels, turbines, failure events |
| [docs/PHYSICS_THERMAL.md](docs/PHYSICS_THERMAL.md) | Gearbox thermal model, residuals (Day 1) |
| [docs/EVALUATION.md](docs/EVALUATION.md) | Rolling-origin protocol + threshold sweep |
| [WEEK_06_REFLECTION.md](WEEK_06_REFLECTION.md) | Week 6 build reflection |
| **handover.md** (this file) | Session handover |

---

## 8. What is next (Task 4+)

1. **Obtain real EDP CSVs** — manual browser download after EDP login, or Zenodo CARE adapter
2. **Re-run all baselines** on real data; compare T01/T06 lead times to literature (~21d / ~89d with CUSUM)
3. **Blog post** — "What an EE sees in turbine data that a data scientist misses"
4. **Cost-optimal threshold picker** — translate trade-off curve into €/alarm maintenance economics

Out of scope for Task 1: CWRU/NASA bearing datasets, SHAP (installed but unused), met-mast ambient temp.

---

## 9. Known issues / caveats

- **EDP portal 403** — no automated download; synthetic fallback added
- **T09** absent in newer EDP releases — loader tolerates; evaluate on T01 + T06
- **T01 high false-alarm rate** on synthetic IF run — needs threshold/contamination sweep
- **`results/` gitignored** — metrics must be regenerated locally after clone
- **Pre-commit** requires git repo + `pre-commit install`

---

## 10. Git workflow

```bash
pytest -q                    # or rely on pre-commit hook
git add ...
git commit -m "message"
git push origin main
```

Per [AGENT.md](AGENT.md): update docs before every push; keep files under 1,000 lines; add tests for changes.

---

## 11. Contact / context

- **12-week plan:** Week focused on gearbox anomaly detection with rolling-origin evaluation and maintenance-framed metrics
- **Recruiter narrative:** At the 99th-percentile operating point on synthetic EDP, physics_hybrid delivers **8.5 false alarms/turbine-year** vs **31** for best pure ML (LSTM-AE), with **65 days** median lead time vs **80 days**. Trade-off curve in `results/plots/threshold_tradeoff.png`; regime analysis in `results/hybrid_vs_ml_regime.json` quantifies fewer false alarms during high-load / hot-ambient periods.
