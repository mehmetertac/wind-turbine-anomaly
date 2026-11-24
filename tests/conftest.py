"""Pytest session hooks for cross-platform smoke tests."""

from __future__ import annotations

import os
import sys


def pytest_configure(config) -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    if sys.platform == "win32":
        try:
            import torch  # noqa: F401 — import before sklearn (Windows OpenMP DLL order)
        except Exception:
            pass