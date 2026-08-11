#!/usr/bin/env python3
"""Generate synthetic EDP-shaped SCADA data for local development."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wind_turbine_anomaly.config import DATA_RAW
from wind_turbine_anomaly.data.synthetic_edp import generate_synthetic_edp_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_RAW,
        help="Directory for generated CSV files (default: data/raw/edp)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible generation",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing CSV files in output directory",
    )
    args = parser.parse_args()

    out = args.output_dir
    existing = list(out.glob("*.csv")) if out.exists() else []
    if existing and not args.force:
        print(f"Output directory already has CSV files: {out}")
        print("Use --force to overwrite, or choose another --output-dir.")
        return 1

    paths = generate_synthetic_edp_dataset(out, random_state=args.seed)
    print(f"Generated synthetic EDP dataset in {out.resolve()}")
    for name, path in paths.items():
        rows = sum(1 for _ in open(path, encoding="utf-8")) - 1
        print(f"  {name}: {rows:,} rows")
    print("\nNext steps:")
    print("  python scripts/download_edp.py --check")
    print("  python scripts/run_if_baseline.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
