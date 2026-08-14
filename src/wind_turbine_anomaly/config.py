"""Configuration for EDP wind-turbine anomaly detection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw" / "edp"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

# Gearbox-relevant SCADA feature columns (10-min averages)
FEATURE_COLUMNS: list[str] = [
    "Gear_Oil_Temp_Avg",
    "Gear_Bear_Temp_Avg",
    "Grd_Prod_Pwr_Avg",
    "Rtr_RPM_Avg",
    "Amb_WindSpeed_Avg",
    "Amb_WindDir_Relative_Avg",
    "Nac_Temp_Avg",
]

POWER_COLUMN = "Grd_Prod_Pwr_Avg"
MIN_POWER_KW = 0.0  # set > 0 to filter idle periods

# Gearbox thermal normal-behavior model (physics-residual Day 1)
THERMAL_TARGET_COLUMNS: list[str] = [
    "Gear_Oil_Temp_Avg",
    "Gear_Bear_Temp_Avg",
]
THERMAL_DRIVER_COLUMNS: list[str] = [
    POWER_COLUMN,
    "Rtr_RPM_Avg",
    "Nac_Temp_Avg",
]
THERMAL_MIN_POWER_KW = 50.0
THERMAL_TRAIN_FRACTION = 0.8
THERMAL_GBM_RMSE_IMPROVEMENT = 0.10  # pick GBM only if RMSE drops by >10%
THERMAL_RESULTS_DIR = RESULTS_DIR / "physics_thermal"

# Filename aliases: canonical name -> accepted on-disk names
SIGNAL_FILE_ALIASES: dict[str, list[str]] = {
    "signals_2016": [
        "wind-farm-1-signals-2016.csv",
        "Wind-Turbine-SCADA-signals-2016.csv",
    ],
    "signals_2017": [
        "wind-farm-1-signals-2017.csv",
        "Wind-Turbine-SCADA-signals-2017_0.csv",
        "Wind-Turbine-SCADA-signals-2017.csv",
    ],
}

FAILURE_FILE_ALIASES: dict[str, list[str]] = {
    "failures_2016": [
        "htw-failures-2016.csv",
        "Historical-Failure-Logbook-2016.csv",
    ],
    "failures_2017": [
        "htw-failures-2017.csv",
        "opendata-wind-failures-2017.csv",
    ],
}

# Mendeley XLSX -> canonical CSV mapping
MENDELEY_XLSX_MAP: dict[str, str] = {
    "Wind-Turbine-SCADA-signals-2016.xlsx": "wind-farm-1-signals-2016.csv",
    "Wind-Turbine-SCADA-signals-2017.xlsx": "wind-farm-1-signals-2017.csv",
    "Historical-Failure-Logbook-2016.xlsx": "htw-failures-2016.csv",
    "opendata-wind-failures-2017.xlsx": "htw-failures-2017.csv",
}

# Known gearbox failures (UTC) — ground truth for evaluation
GEARBOX_FAILURES: list[dict[str, str]] = [
    {
        "turbine_id": "T01",
        "timestamp": "2016-07-18T02:10:00+00:00",
        "remarks": "Gearbox pump damaged",
    },
    {
        "turbine_id": "T06",
        "timestamp": "2017-10-17T08:38:00+00:00",
        "remarks": "Gearbox bearings damaged",
    },
    # T09 events (optional — turbine may be absent from newer dataset releases)
    {
        "turbine_id": "T09",
        "timestamp": "2016-10-11T08:06:00+00:00",
        "remarks": "Gearbox repaired",
    },
    {
        "turbine_id": "T09",
        "timestamp": "2017-10-18T08:32:00+00:00",
        "remarks": "Gearbox noise",
    },
]

DEFAULT_BUFFER_DAYS = 90
DEFAULT_HORIZON_DAYS = 30
DEFAULT_CONTAMINATION = 0.01
DEFAULT_PERSISTENCE_SAMPLES = 6  # 1 hour at 10-min resolution
DEFAULT_COOLDOWN_HOURS = 24
DEFAULT_THRESHOLD_PERCENTILE = 99.0

# Dense autoencoder hyperparameters
AE_HIDDEN_DIMS: list[int] = [32, 16]
AE_BOTTLENECK_DIM = 8
AE_EPOCHS = 50
AE_BATCH_SIZE = 256
AE_LR = 1e-3
AE_PATIENCE = 5
AE_VAL_FRACTION = 0.1

# LSTM autoencoder hyperparameters
LSTM_WINDOW_SIZE = 144  # 24 h at 10-min resolution
LSTM_HIDDEN_DIM = 32
LSTM_LATENT_DIM = 16
LSTM_MAX_TRAIN_WINDOWS = 32768  # subsample cap for CPU training time
LSTM_EPOCHS = 15


@dataclass(frozen=True)
class GearboxFailure:
    """Logged gearbox failure event."""

    turbine_id: str
    timestamp: datetime
    remarks: str
