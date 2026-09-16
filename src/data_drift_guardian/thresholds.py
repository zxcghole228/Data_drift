"""Единое разрешение distance-порогов для drift-метрик."""

from __future__ import annotations

from .contracts import DistanceMethod, DriftConfig, ThresholdSource


def resolve_distance_threshold(
    feature: str,
    method: DistanceMethod,
    config: DriftConfig,
) -> tuple[float | None, ThresholdSource]:
    """Вернуть эффективный порог и источник по единому правилу приоритета.

    Явный feature-level ``null`` отключает решение для конкретного признака и
    поэтому имеет приоритет над общим порогом метода.
    """

    feature_overrides = config["feature_thresholds"].get(feature, {})
    if method in feature_overrides:
        return feature_overrides[method], "feature"

    global_threshold = config["distance_thresholds"][method]
    if global_threshold is not None:
        return global_threshold, "global"
    return None, "not_configured"
