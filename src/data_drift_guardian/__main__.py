"""Точка входа для ``python -m data_drift_guardian``."""

from __future__ import annotations

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
