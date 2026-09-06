"""Проверка схемы. Ответственный: Павел; см. план на 08.09."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from .contracts import SchemaResult

if TYPE_CHECKING:
    import pandas as pd


def validate_schema(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    config: Mapping[str, Any],
) -> SchemaResult:
    """Проверить колонки и типы без изменения входов; пока заглушка."""
    raise NotImplementedError("Проверка схемы запланирована в docs/plans/pavel.md")
