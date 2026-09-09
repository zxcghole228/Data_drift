from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.spatial import distance

from data_drift_guardian.drift.stability import js_divergence, psi


@pytest.mark.parametrize("calculation", [psi, js_divergence])
def test_identical_distributions_return_zero(calculation) -> None:
    result = calculation([0.2, 0.3, 0.5], [0.2, 0.3, 0.5])

    assert result["status"] == "ok"
    assert result["value"] == pytest.approx(0.0, abs=1e-15)
    assert result["threshold"] is None
    assert result["p_value"] is None
    assert result["alert"] is None
    assert result["reason"] is None


def test_psi_matches_documented_formula() -> None:
    reference = np.array([0.4, 0.6])
    current = np.array([0.5, 0.5])
    smoothing = 1e-6

    result = psi(reference, current, smoothing=smoothing)

    smoothed_reference = (reference + smoothing) / (1 + 2 * smoothing)
    smoothed_current = (current + smoothing) / (1 + 2 * smoothing)
    expected = np.sum(
        (smoothed_current - smoothed_reference)
        * np.log(smoothed_current / smoothed_reference)
    )
    assert result["status"] == "ok"
    assert result["value"] == pytest.approx(expected)
    assert sum(result["details"]["contributions"]) == pytest.approx(expected)


def test_zero_cells_are_finite_after_psi_smoothing() -> None:
    result = psi([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])

    assert result["status"] == "ok"
    assert result["value"] is not None
    assert np.isfinite(result["value"])
    assert result["value"] > 0.0
    assert result["details"]["smoothing_strategy"] == ("additive_then_renormalize")


def test_js_returns_divergence_not_scipy_distance() -> None:
    reference = [1.0, 0.0]
    current = [0.5, 0.5]

    result = js_divergence(reference, current, base=2.0)
    expected_distance = distance.jensenshannon(reference, current, base=2.0)

    assert result["status"] == "ok"
    assert result["value"] == pytest.approx(expected_distance**2)
    assert result["details"]["scipy_distance"] == pytest.approx(expected_distance)
    assert result["details"]["returned_quantity"] == "divergence"


def test_disjoint_distributions_have_maximum_base_two_js_divergence() -> None:
    result = js_divergence([1.0, 0.0], [0.0, 1.0], base=2.0)

    assert result["status"] == "ok"
    assert result["value"] == pytest.approx(1.0)
    assert 0.0 <= result["value"] <= 1.0


@pytest.mark.parametrize("calculation", [psi, js_divergence])
def test_vectors_are_normalized_before_calculation(calculation) -> None:
    result = calculation([2.0, 2.0], [5.0, 5.0])

    assert result["status"] == "ok"
    assert result["value"] == pytest.approx(0.0, abs=1e-15)
    assert result["details"]["reference_sum_before_normalization"] == 4.0
    assert result["details"]["current_sum_before_normalization"] == 10.0


@pytest.mark.parametrize("calculation", [psi, js_divergence])
@pytest.mark.parametrize(
    ("reference", "current", "status", "message"),
    [
        ([], [], "skipped", "must not be empty"),
        ([0.0, 0.0], [0.5, 0.5], "skipped", "positive sum"),
        ([0.5, 0.5], [1.0], "error", "equal lengths"),
        ([0.5, -0.5], [0.5, 0.5], "error", "negative"),
        ([0.5, np.nan], [0.5, 0.5], "error", "finite"),
        ([0.5, np.inf], [0.5, 0.5], "error", "finite"),
        (["0.5", "0.5"], [0.5, 0.5], "error", "real numbers"),
        ([True, False], [0.5, 0.5], "error", "real numbers"),
        ([0.5 + 0.1j, 0.5], [0.5, 0.5], "error", "real numbers"),
        ([[0.5, 0.5]], [[0.5, 0.5]], "error", "one-dimensional"),
    ],
)
def test_invalid_probability_vectors_are_explicit(
    calculation,
    reference,
    current,
    status: str,
    message: str,
) -> None:
    result = calculation(reference, current)

    assert result["status"] == status
    assert result["value"] is None
    assert result["alert"] is None
    assert result["reason"] is not None
    assert message in result["reason"]


@pytest.mark.parametrize("smoothing", [0.0, 1.0, -1e-6, np.inf, True])
def test_invalid_psi_smoothing_raises_value_error(smoothing) -> None:
    with pytest.raises(ValueError):
        psi([0.5, 0.5], [0.4, 0.6], smoothing=smoothing)


@pytest.mark.parametrize("base", [0.0, 1.0, -2.0, np.inf, True])
def test_invalid_js_base_raises_value_error(base) -> None:
    with pytest.raises(ValueError):
        js_divergence([0.5, 0.5], [0.4, 0.6], base=base)


@pytest.mark.parametrize("calculation", [psi, js_divergence])
def test_result_is_json_safe(calculation) -> None:
    result = calculation([0.0, 1.0], [1.0, 0.0])

    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("calculation", [psi, js_divergence])
def test_large_finite_weights_are_normalized_without_overflow(calculation) -> None:
    result = calculation([1e308, 1e308], [1e308, 1e308])

    assert result["status"] == "ok"
    assert result["value"] == pytest.approx(0.0, abs=1e-15)
    assert result["details"]["normalization_rescaled"] is True
    assert result["details"]["reference_sum_before_normalization"] is None
    json.dumps(result, allow_nan=False)
