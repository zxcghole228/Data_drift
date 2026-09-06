"""Общий API. Ответственный: Павел; первая интеграция запланирована на 10.09."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from .contracts import AnalysisResult

if TYPE_CHECKING:
    import pandas as pd


def analyze(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    config: Mapping[str, Any] | None = None,
) -> AnalysisResult:
    """Сравнить два DataFrame без их изменения.

    В каркасе вычисления не реализованы. Не возвращает фиктивные результаты.
    """
    raise NotImplementedError("Общий pipeline запланирован в docs/plans/pavel.md")
