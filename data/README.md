# EDP raw data

Place downloaded EDP open wind-farm files in `data/raw/edp/`. **Do not commit raw files** — they are gitignored.

## Synthetic data (no portal required)

When EDP Open Data is unavailable, generate a **development-only** dataset with the same schema and failure dates:

```bash
python scripts/generate_synthetic_edp.py --force
python scripts/download_edp.py --check
python scripts/run_if_baseline.py
```

This writes ~400k SCADA rows (2016–2017, T01/T06/T07/T11) with injected gearbox degradation before logged failures. A marker file `.synthetic_edp` is created in the output directory. **Not for publication metrics** — use real EDP data when available.

## Primary source: EDP Open Data

- Portal: [EDP Open Data](https://www.edp.com/en/innovation/open-data/data)
- License: CC-BY-SA (free registration required)
- Coverage: onshore wind farm, Portugal, **2016–2017**, 10-minute SCADA

### Required files

| Purpose | Typical filename | Alias (also accepted) |
|---------|------------------|----------------------|
| SCADA 2016 | `wind-farm-1-signals-2016.csv` | `Wind-Turbine-SCADA-signals-2016.csv` |
| SCADA 2017 | `wind-farm-1-signals-2017.csv` | `Wind-Turbine-SCADA-signals-2017_0.csv` |
| Failures 2016 | `htw-failures-2016.csv` | `Historical-Failure-Logbook-2016.csv` |
| Failures 2017 | `htw-failures-2017.csv` | `opendata-wind-failures-2017.csv` |

## Fallback: Mendeley Data

- Dataset: [Description of the SCADA dataset of the EDP onshore wind farm](https://data.mendeley.com/datasets/zjxjnjp3xs) (DOI: 10.17632/zjxjnjp3xs)
- Format: Excel (`.xlsx`), ~219 MB total
- Place XLSX files in `data/raw/edp/` and convert:

```bash
python scripts/download_edp.py --from-mendeley
python scripts/download_edp.py --check
```

## Verify installation

```bash
python scripts/download_edp.py --check
python scripts/download_edp.py --instructions
```

## Reference implementations

- [OpenWindSCADA EDP notebook](https://github.com/sltzgs/OpenWindSCADA/blob/main/notebooks/edp_open_data.ipynb) — column overview and loader patterns
- [Frontiers 2022 gearbox CUSUM paper](https://doi.org/10.3389/fenrg.2022.904622) — train/test split and failure dates

## Processed data

Cleaned per-turbine parquet files (when generated) go in `data/processed/` — also gitignored.

See also: [docs/DATA.md](../docs/DATA.md)
