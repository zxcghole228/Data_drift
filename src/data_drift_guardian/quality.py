"""Пропуски, дубликаты и диапазоны. Ответственный: Павел; план на 09.09."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from .contracts import QualityResult

if TYPE_CHECKING:
    import pandas as pd


def check_quality(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    config: Mapping[str, Any],
) -> QualityResult:
    """Вернуть результаты Data Quality; текущая версия является заглушкой."""
    raise NotImplementedError("Data Quality запланирован в docs/plans/pavel.md")
