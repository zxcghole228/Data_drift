"""Правила drift-алертов. Ответственный: Михаил; план на 10.09."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .contracts import Alert, FeatureResult


def build_drift_alerts(
    features: Mapping[str, FeatureResult],
    config: Mapping[str, Any],
) -> list[Alert]:
    """Собрать объяснимые алерты по применимым правилам; пока заглушка."""
    raise NotImplementedError("Правила drift-алертов запланированы в docs/plans/mikhail.md")
