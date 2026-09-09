"""Population Stability Index и Jensen–Shannon divergence."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Real
from typing import Any

import numpy as np
from scipy.spatial import distance

from ..contracts import CheckResult, Status

DEFAULT_PSI_SMOOTHING = 1e-6
DEFAULT_JS_BASE = 2.0


def _empty_result(
    name: str,
    status: Status,
    reason: str,
    details: dict[str, Any] | None = None,
) -> CheckResult:
    return {
        "name": name,
        "status": status,
        "value": None,
        "threshold": None,
        "p_value": None,
        "adjusted_p_value": None,
        "alert": None,
        "reason": reason,
        "details": {} if details is None else details,
    }


def _prepare_probability_vectors(
    name: str,
    reference_probabilities: Sequence[float],
    current_probabilities: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | CheckResult:
    try:
        raw_reference = np.asarray(reference_probabilities)
        raw_current = np.asarray(current_probabilities)
    except (TypeError, ValueError, OverflowError) as exc:
        return _empty_result(
            name,
            "error",
            f"Probability vectors could not be read: {type(exc).__name__}: {exc}",
        )

    details: dict[str, Any] = {
        "n_cells_reference": int(raw_reference.size),
        "n_cells_current": int(raw_current.size),
    }
    if raw_reference.ndim != 1 or raw_current.ndim != 1:
        return _empty_result(
            name,
            "error",
            "Probability vectors must be one-dimensional",
            details,
        )
    if raw_reference.size == 0 or raw_current.size == 0:
        return _empty_result(
            name,
            "skipped",
            "Probability vectors must not be empty",
            details,
        )
    if raw_reference.size != raw_current.size:
        return _empty_result(
            name,
            "error",
            "Probability vectors must have equal lengths",
            details,
        )
    for label, raw_values in (
        ("reference", raw_reference),
        ("current", raw_current),
    ):
        if raw_values.dtype.kind in {"b", "c", "S", "U"}:
            return _empty_result(
                name,
                "error",
                f"{label} probabilities must contain real numbers, not "
                f"{raw_values.dtype}",
                details,
            )
        if raw_values.dtype.kind == "O" and any(
            isinstance(item, bool) or not isinstance(item, Real) for item in raw_values
        ):
            return _empty_result(
                name,
                "error",
                f"{label} probabilities must contain only real numbers",
                details,
            )

    try:
        reference = raw_reference.astype(np.float64, copy=False)
        current = raw_current.astype(np.float64, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        return _empty_result(
            name,
            "error",
            "Probability vectors must contain real numbers: "
            f"{type(exc).__name__}: {exc}",
            details,
        )

    if not np.isfinite(reference).all() or not np.isfinite(current).all():
        return _empty_result(
            name,
            "error",
            "Probability vectors must contain only finite values",
            details,
        )
    if np.any(reference < 0.0) or np.any(current < 0.0):
        return _empty_result(
            name,
            "error",
            "Probability vectors must not contain negative values",
            details,
        )

    reference_max = float(reference.max())
    current_max = float(current.max())
    if reference_max <= 0.0 or current_max <= 0.0:
        return _empty_result(
            name,
            "skipped",
            "Each probability vector must have a positive sum",
            details,
        )

    with np.errstate(over="ignore", invalid="ignore"):
        reference_sum = float(reference.sum())
        current_sum = float(current.sum())
    details["reference_sum_before_normalization"] = (
        reference_sum if np.isfinite(reference_sum) else None
    )
    details["current_sum_before_normalization"] = (
        current_sum if np.isfinite(current_sum) else None
    )
    details["normalization_rescaled"] = bool(
        not np.isfinite(reference_sum) or not np.isfinite(current_sum)
    )

    scaled_reference = reference / reference_max
    scaled_current = current / current_max
    normalized_reference = scaled_reference / scaled_reference.sum()
    normalized_current = scaled_current / scaled_current.sum()
    details["reference_probabilities"] = [
        float(value) for value in normalized_reference
    ]
    details["current_probabilities"] = [float(value) for value in normalized_current]
    return normalized_reference, normalized_current, details


def _validate_real_parameter(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        # Invalid metric configuration is ValueError by the shared contract.
        raise ValueError(f"{name} must be a real number")  # noqa: TRY004
    normalized = float(value)
    if not np.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _successful_result(
    name: str,
    value: float,
    details: dict[str, Any],
) -> CheckResult:
    if not np.isfinite(value):
        return _empty_result(
            name,
            "error",
            f"{name} returned a non-finite result",
            details,
        )
    return {
        "name": name,
        "status": "ok",
        "value": float(value),
        "threshold": None,
        "p_value": None,
        "adjusted_p_value": None,
        "alert": None,
        "reason": None,
        "details": details,
    }


def psi(
    reference_probabilities: Sequence[float],
    current_probabilities: Sequence[float],
    *,
    smoothing: float = DEFAULT_PSI_SMOOTHING,
) -> CheckResult:
    """Вычислить PSI после аддитивного сглаживания и нормировки."""
    smoothing = _validate_real_parameter("smoothing", smoothing)
    if not 0.0 < smoothing < 1.0:
        raise ValueError("smoothing must be strictly between 0 and 1")
    if smoothing < np.finfo(np.float64).tiny:
        raise ValueError("smoothing is too small for float64 calculations")

    prepared = _prepare_probability_vectors(
        "psi",
        reference_probabilities,
        current_probabilities,
    )
    if isinstance(prepared, dict):
        return prepared
    reference, current, details = prepared

    n_cells = reference.size
    smoothed_reference = (reference + smoothing) / (1.0 + smoothing * n_cells)
    smoothed_current = (current + smoothing) / (1.0 + smoothing * n_cells)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        contributions = (smoothed_current - smoothed_reference) * np.log(
            smoothed_current / smoothed_reference
        )
    if not np.isfinite(contributions).all():
        return _empty_result(
            "psi",
            "error",
            "psi returned non-finite cell contributions",
            details,
        )
    value = float(contributions.sum())
    if -1e-15 < value < 0.0:
        value = 0.0

    details.update(
        {
            "log_base": "e",
            "smoothing": smoothing,
            "smoothing_strategy": "additive_then_renormalize",
            "smoothed_reference_probabilities": [
                float(item) for item in smoothed_reference
            ],
            "smoothed_current_probabilities": [
                float(item) for item in smoothed_current
            ],
            "contributions": [float(item) for item in contributions],
        }
    )
    return _successful_result("psi", value, details)


def js_divergence(
    reference_probabilities: Sequence[float],
    current_probabilities: Sequence[float],
    *,
    base: float = DEFAULT_JS_BASE,
) -> CheckResult:
    """Вычислить Jensen–Shannon divergence вместо расстояния."""
    base = _validate_real_parameter("base", base)
    if base <= 0.0 or base == 1.0:
        raise ValueError("base must be positive and different from 1")

    prepared = _prepare_probability_vectors(
        "js",
        reference_probabilities,
        current_probabilities,
    )
    if isinstance(prepared, dict):
        return prepared
    reference, current, details = prepared

    js_distance = float(distance.jensenshannon(reference, current, base=base))
    if not np.isfinite(js_distance):
        return _empty_result(
            "js",
            "error",
            "js returned a non-finite distance",
            details,
        )
    divergence = js_distance**2
    if -1e-15 < divergence < 0.0:
        divergence = 0.0
    details.update(
        {
            "log_base": base,
            "scipy_distance": js_distance,
            "returned_quantity": "divergence",
        }
    )
    return _successful_result("js", divergence, details)
