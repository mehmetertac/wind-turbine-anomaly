"""Packaging and reproducibility constraints."""

from __future__ import annotations

from pathlib import Path

import tomllib


def test_requires_python_pin() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    assert data["project"]["requires-python"] == ">=3.11,<3.13"