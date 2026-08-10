#!/usr/bin/env python3
"""Download or convert EDP open wind-farm SCADA data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "edp"

REQUIRED_CANONICAL = [
    "wind-farm-1-signals-2016.csv",
    "wind-farm-1-signals-2017.csv",
    "htw-failures-2016.csv",
    "htw-failures-2017.csv",
]

MANUAL_STEPS = """
EDP Open Data — manual download
================================
1. Register (free) at https://www.edp.com/en/innovation/open-data/data
2. Download Wind Farm 1 datasets (2016 + 2017):
   - SCADA signals (CSV or XLSX)
   - Failure history / HTW failures (CSV or XLSX)
3. Place files in: {raw_dir}

Accepted filenames (any alias works):
  - wind-farm-1-signals-2016.csv  OR  Wind-Turbine-SCADA-signals-2016.csv
  - wind-farm-1-signals-2017.csv  OR  Wind-Turbine-SCADA-signals-2017_0.csv
  - htw-failures-2016.csv         OR  Historical-Failure-Logbook-2016.csv
  - htw-failures-2017.csv         OR  opendata-wind-failures-2017.csv

Mendeley fallback (XLSX)
========================
1. Download from https://data.mendeley.com/datasets/zjxjnjp3xs
2. Place .xlsx files in: {raw_dir}
3. Run: python scripts/download_edp.py --from-mendeley

Reference: https://github.com/sltzgs/OpenWindSCADA
"""


def check_files(raw_dir: Path) -> bool:
    """Return True if all required datasets are present (any alias)."""
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from wind_turbine_anomaly.config import FAILURE_FILE_ALIASES, SIGNAL_FILE_ALIASES
    from wind_turbine_anomaly.data.load_edp import _resolve_file

    groups = [
        SIGNAL_FILE_ALIASES["signals_2016"],
        SIGNAL_FILE_ALIASES["signals_2017"],
        FAILURE_FILE_ALIASES["failures_2016"],
        FAILURE_FILE_ALIASES["failures_2017"],
    ]
    ok = True
    for aliases in groups:
        try:
            path = _resolve_file(raw_dir, aliases)
            print(f"  OK  {path.name}")
        except FileNotFoundError:
            print(f"  MISSING  one of: {', '.join(aliases)}")
            ok = False
    return ok


def convert_mendeley_xlsx(raw_dir: Path) -> None:
    """Convert Mendeley XLSX files to canonical CSV names."""
    import pandas as pd

    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from wind_turbine_anomaly.config import MENDELEY_XLSX_MAP

    raw_dir.mkdir(parents=True, exist_ok=True)
    converted = 0
    for xlsx_name, csv_name in MENDELEY_XLSX_MAP.items():
        xlsx_path = raw_dir / xlsx_name
        if not xlsx_path.exists():
            # Try glob for partial name match
            matches = list(raw_dir.glob(f"*{xlsx_name.split('.')[0][:20]}*.xlsx"))
            if matches:
                xlsx_path = matches[0]
            else:
                print(f"  SKIP  {xlsx_name} not found")
                continue
        csv_path = raw_dir / csv_name
        print(f"  Converting {xlsx_path.name} -> {csv_name}")
        df = pd.read_excel(xlsx_path)
        df.to_csv(csv_path, index=False)
        converted += 1

    if converted == 0:
        print("No XLSX files converted. Place Mendeley downloads in:", raw_dir)
    else:
        print(f"Converted {converted} file(s).")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Directory for raw EDP files",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check whether required files are present",
    )
    parser.add_argument(
        "--from-mendeley",
        action="store_true",
        help="Convert Mendeley XLSX files in raw-dir to CSV",
    )
    parser.add_argument(
        "--instructions",
        action="store_true",
        help="Print manual download instructions",
    )
    args = parser.parse_args()
    raw_dir = args.raw_dir

    if args.instructions:
        print(MANUAL_STEPS.format(raw_dir=raw_dir))
        return 0

    raw_dir.mkdir(parents=True, exist_ok=True)

    if args.from_mendeley:
        convert_mendeley_xlsx(raw_dir)

    if args.check or not args.from_mendeley:
        print(f"Checking EDP data in {raw_dir}...")
        ok = check_files(raw_dir)
        if ok:
            print("All required files present.")
            return 0
        print(MANUAL_STEPS.format(raw_dir=raw_dir))
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
