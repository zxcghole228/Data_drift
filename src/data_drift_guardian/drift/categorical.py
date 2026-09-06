"""Категориальные проверки. Ответственный: Михаил; план на 10.09."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from ..contracts import CheckResult

if TYPE_CHECKING:
    import pandas as pd


def chi_square(
    reference: pd.Series,
    current: pd.Series,
    config: Mapping[str, Any],
) -> CheckResult:
    """χ² для сравнения частот категорий; текущая версия является заглушкой."""
    raise NotImplementedError("χ² запланирован в docs/plans/mikhail.md")
