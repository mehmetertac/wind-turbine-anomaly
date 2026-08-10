# Data guide

EDP onshore wind-farm SCADA and failure logs used in this project. See [data/README.md](../data/README.md) for download instructions.

## Turbines

| ID | Notes |
|----|-------|
| T01 | Gearbox failure logged (2016) |
| T06 | Gearbox failure logged (2017) |
| T07 | No gearbox failure in log — used for false-alarm rate |
| T11 | No gearbox failure in log — used for false-alarm rate |

Older releases included **T09** (2 gearbox events). Current mirrors may omit T09; loader tolerates its absence.

Turbines are 2 MW class, three-stage planetary/spur gearbox, 10-min sampling (UTC).

## Gearbox-relevant SCADA channels

Defined in [`src/wind_turbine_anomaly/config.py`](../src/wind_turbine_anomaly/config.py):

| Column | Description |
|--------|-------------|
| `Gear_Oil_Temp_Avg` | 10-min average gearbox oil temperature (°C) |
| `Gear_Bear_Temp_Avg` | 10-min average gearbox bearing (HSS) temperature (°C) |
| `Grd_Prod_Pwr_Avg` | Active power output (kW) |
| `Rtr_RPM_Avg` | Rotor speed (RPM) |
| `Amb_WindSpeed_Avg` | Nacelle anemometer wind speed (m/s) |
| `Amb_WindDir_Relative_Avg` | Relative wind direction (°) |
| `Nac_Temp_Avg` | Nacelle interior temperature (°C) |

Full SCADA files contain **83 columns** (avg/min/max/std variants). Task 1 uses average columns only.

Optional later: met-mast ambient temperature from separate EDP met-mast files.

## Failure log schema

| Column | Description |
|--------|-------------|
| `Timestamp` | Failure event time (UTC) |
| `Turbine_ID` | e.g. `T01` |
| `Component` | `GEARBOX`, `GENERATOR`, etc. |
| `Remarks` | Free-text description |

Filter: `Component == "GEARBOX"` (case-insensitive).

## Ground-truth gearbox events

| Turbine | Timestamp (UTC) | Remarks | Split (literature) |
|---------|-----------------|---------|-------------------|
| T01 | 2016-07-18 02:10 | Gearbox pump damaged | Training period |
| T06 | 2017-10-17 08:38 | Gearbox bearings damaged | Test period |
| T09* | 2016-10-11 08:06 | Gearbox repaired | Training period |
| T09* | 2017-10-18 08:32 | Gearbox noise | Test period |

\*Present only if T09 is included in your download.

Total failure log: **28 events** across all components (2016–2017).

## Loading in Python

```python
from pathlib import Path
from wind_turbine_anomaly.data.load_edp import load_edp_dataset

turbines, gearbox_failures = load_edp_dataset(Path("data/raw/edp"))
```

## References

- EDP Open Data: https://www.edp.com/en/innovation/open-data/data
- Mendeley description: https://doi.org/10.17632/zjxjnjp3xs
- Evaluation protocol: [EVALUATION.md](EVALUATION.md)
