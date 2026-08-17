# Week 6 Reflection — Gearbox failure prediction N days ahead

End-of-week reflection for the wind-turbine-anomaly project. Generic EE/maintenance perspective — no personal site-visit claims.

Cross-links: [README.md](README.md) | [docs/PHYSICS_THERMAL.md](docs/PHYSICS_THERMAL.md) | [docs/EVALUATION.md](docs/EVALUATION.md) | [handover.md](handover.md)

---

## What did I build?

This week closed the loop from "anomaly score" to "maintenance decision." The arc:

1. **EDP data pipeline** — load, clean, and split SCADA by turbine with gearbox failure ground truth (T01 pump damage, T06 bearing damage).
2. **Three pure-ML baselines** — Isolation Forest, dense autoencoder, LSTM-AE on raw multivariate SCADA.
3. **Physics-residual hybrid** — per-turbine thermal normal-behavior model (oil + bearing), residual drift features (EWMA, rolling stats), Isolation Forest on the residual stream.
4. **Unified evaluation protocol** — rolling-origin training (90-day buffer before failure), forward scoring, sustained-alarm rule, lead time in days, false alarms per turbine-year.
5. **Threshold sweep + headline claim** — re-score saved anomaly outputs at multiple percentiles; produce the trade-off curve and a single-sentence project claim.

The central insight: the question is not "is this point an outlier?" but **"how many days of warning do I get, and at what false-alarm rate?"** Everything — thermal model, residual features, threshold selection — serves that question.

### Headline artifact

After running `python scripts/run_all_ml_baselines.py` on synthetic EDP:

> At the 99th-percentile threshold, physics_hybrid flags gearbox failures a median of **65 days** in advance at **8.5 false alarms per turbine-year**.

Best pure ML (LSTM-AE) at the same operating point: **80 days** median lead time at **31 false alarms/turbine-year**. The hybrid wins on operability (fewer truck rolls); LSTM-AE leads on raw lead time on this synthetic run. The trade-off curve in `results/plots/threshold_tradeoff.png` is the project's central deliverable — not a single row in `metrics.csv`.

---

## What's still fuzzy?

### Threshold selection

The default 99th percentile of healthy training scores is a reasonable starting point but arbitrary. The sweep shows sensitivity: at 90th percentile, hybrid lead time rises to ~85 days but false alarms jump to ~54/turbine-year. There is no cost-optimal picker yet — translating €/inspection vs €/failure into a recommended percentile is future work.

### Residual model drift over seasons

The thermal model uses nacelle interior temperature as an ambient proxy. Seasonal bias (winter nacelle cold soak vs summer self-heating) can shift residuals slowly without degradation. A met-mast ambient channel would help; so would periodic model refitting. The current single-origin eval (train once, score forward) does not test seasonal retraining.

### Label quality of failure logs

The logged failure timestamp is when maintenance recorded the event — not necessarily when damage started. T01 (pump damage) and T06 (bearing damage) may have different true onset times and SCADA signatures. Comparing lead times across failure modes is only fair if the label represents comparable physics.

### Single-origin evaluation

Production systems retrain quarterly or after confirmed faults. This project scores forward from one training cut without rolling retrain. Lead times could shorten or false alarms could drift as the model ages.

### Synthetic vs real EDP

All headline numbers above are on synthetic data with injected degradation. They validate the pipeline, not the physics. Real EDP CSVs are blocked behind a 403 on automated download; manual acquisition or a Zenodo adapter is the gating item for publication-quality metrics.

---

## Field-relevant patterns (generic EE perspective)

These patterns are consistent with industry experience and the EDP failure log descriptions — not claims about specific site visits.

### T01 — pump / lubrication (shorter warning window)

Pump damage and lubrication failures often present with a shorter SCADA warning window than bearing wear-out. Temperature may stay near normal until circulation degrades, then rise quickly. On synthetic data, T01 is the harder case: hybrid lead time (~53 days @ 99th percentile) is shorter than T06 (~77 days). An aggressive threshold catches T01 earlier but floods the site with false alarms — exactly the trade-off curve the sweep exposes.

**What an EE checks first:** oil level, filter differential, cooler fouling, recent oil analysis trends. A SCADA alarm without these checks is premature.

### T06 — bearing wear-out (slow thermal drift)

Bearing damage typically shows gradual friction increase → heat generation → negative thermal residual (hotter than expected at the same load). This is where physics-residual + EWMA should shine: the drift is small per day but persistent, visible in the residual stream long before raw temperature crosses a fixed limit.

On synthetic data, T06 is the flagship case — both hybrid and LSTM-AE exceed the 30-day successful-warning threshold.

### High-load false alarms — why raw ML fires in summer

Raw gearbox temperature rises with power and ambient. A multivariate ML detector trained on healthy data learns "normal" correlations but still fires when the turbine runs hard on a hot afternoon — even with no fault. Regime analysis (`results/hybrid_vs_ml_regime.json`) shows pure ML false alarms clustering in `high_load` and `hot_ambient` regimes.

The physics model explains load-dependent heating; the residual isolates what's left. An EE seeing a temperature alarm during peak production first asks: "Is this proportional to load?" The hybrid encodes that question into the detection pipeline.

### Maintenance translation

- **65 days of warning** → schedule crane for a known low-wind week, pull oil sample, confirm fault before committing to €100k replacement.
- **8.5 false alarms/turbine-year** → ~€43k/year inspection cost for a 4-turbine fleet if every alarm is acted on (~€5k/truck roll). One avoided catastrophic failure pays for ~20 false alarms.
- **31 false alarms/turbine-year** (best pure ML) → ~€155k/year for the same fleet — harder to justify unless lead-time gains are large and failures are frequent.

The hybrid's value proposition on synthetic data is operability: comparable warning horizon with far fewer unnecessary truck rolls. Real EDP will determine whether that holds on published benchmarks.

---

## What's next

1. Obtain real EDP CSVs and re-run the full benchmark + sweep.
2. Compare T01/T06 lead times to Hack the Wind literature (~21 d / ~89 d with CUSUM).
3. Blog post: "What an EE sees in turbine data that a data scientist misses."
4. Cost-optimal threshold selection from the trade-off curve.
