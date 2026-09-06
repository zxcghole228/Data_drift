"""Классификация источника данных. Ответственный: Михаил; план на 11.09."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from .contracts import AdversarialResult

if TYPE_CHECKING:
    import pandas as pd


def adversarial_validate(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    config: Mapping[str, Any],
) -> AdversarialResult:
    """Вернуть отложенный AUC и важности признаков; пока заглушка."""
    raise NotImplementedError("Adversarial Validation запланирован в docs/plans/mikhail.md")
