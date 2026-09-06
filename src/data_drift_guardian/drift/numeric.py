"""KS и Wasserstein. Ответственный: Михаил; план на 08.09."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..contracts import CheckResult

if TYPE_CHECKING:
    import pandas as pd


def ks_test(reference: pd.Series, current: pd.Series) -> CheckResult:
    """Двухвыборочный KS; текущая версия является заглушкой."""
    raise NotImplementedError("KS запланирован в docs/plans/mikhail.md")


def wasserstein(reference: pd.Series, current: pd.Series) -> CheckResult:
    """Расстояние Вассерштейна; текущая версия является заглушкой."""
    raise NotImplementedError("Wasserstein запланирован в docs/plans/mikhail.md")
