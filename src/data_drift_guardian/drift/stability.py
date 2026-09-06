"""PSI и Jensen–Shannon Divergence. Ответственный: Михаил; план на 09.09."""

from __future__ import annotations

from collections.abc import Sequence

from ..contracts import CheckResult


def psi(
    reference_probabilities: Sequence[float],
    current_probabilities: Sequence[float],
) -> CheckResult:
    """PSI для подготовленных вероятностей; текущая версия является заглушкой."""
    raise NotImplementedError("PSI запланирован в docs/plans/mikhail.md")


def js_divergence(
    reference_probabilities: Sequence[float],
    current_probabilities: Sequence[float],
) -> CheckResult:
    """JS divergence с согласованной базой логарифма; пока заглушка."""
    raise NotImplementedError("JS запланирована в docs/plans/mikhail.md")
