"""Согласованное представление распределений. Ответственный: Михаил; план на 09.09."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


def build_probabilities(
    reference: pd.Series,
    current: pd.Series,
    config: Mapping[str, Any],
) -> tuple[list[float], list[float], dict[str, Any]]:
    """Вернуть два вектора вероятностей и описание разбиения; пока заглушка."""
    raise NotImplementedError("Подготовка распределений запланирована в docs/plans/mikhail.md")
