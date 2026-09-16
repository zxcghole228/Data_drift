"""Тесты единого resolver для distance-порогов."""

from __future__ import annotations

from typing import cast

import pytest

from data_drift_guardian.contracts import DriftConfig
from data_drift_guardian.thresholds import resolve_distance_threshold


def _config() -> DriftConfig:
    return cast(
        DriftConfig,
        {
            "distance_thresholds": {
                "wasserstein": 5.0,
                "psi": 0.2,
                "js": None,
            },
            "feature_thresholds": {
                "age": {"wasserstein": 2.0, "psi": None},
            },
        },
    )


def test_feature_override_has_priority() -> None:
    assert resolve_distance_threshold("age", "wasserstein", _config()) == (
        2.0,
        "feature",
    )


def test_explicit_feature_null_disables_global_threshold() -> None:
    assert resolve_distance_threshold("age", "psi", _config()) == (
        None,
        "feature",
    )


def test_global_threshold_is_fallback() -> None:
    assert resolve_distance_threshold("income", "wasserstein", _config()) == (
        5.0,
        "global",
    )


@pytest.mark.parametrize("feature", ["age", "income"])
def test_missing_threshold_is_explicit(feature: str) -> None:
    assert resolve_distance_threshold(feature, "js", _config()) == (
        None,
        "not_configured",
    )
