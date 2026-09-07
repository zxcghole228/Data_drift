from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from data_drift_guardian.drift.numeric import ks_test, wasserstein


@pytest.mark.parametrize("calculation", [ks_test, wasserstein])
def test_identical_values_have_zero_distance(calculation) -> None:
    values = pd.Series([1.0, 2.0, 3.0, 4.0])

    result = calculation(values, values.copy())

    assert result["status"] == "ok"
    assert result["value"] == pytest.approx(0.0)
    assert result["threshold"] is None
    assert result["alert"] is None
    assert result["reason"] is None
    if calculation is ks_test:
        assert result["p_value"] == pytest.approx(1.0)
    else:
        assert result["p_value"] is None


def test_obvious_shift_is_detected_numerically() -> None:
    reference = pd.Series(np.arange(100, dtype=float))
    current = pd.Series(np.arange(100, dtype=float) + 1_000.0)

    ks_result = ks_test(reference, current)
    wasserstein_result = wasserstein(reference, current)

    assert ks_result["status"] == "ok"
    assert ks_result["value"] == pytest.approx(1.0)
    assert ks_result["p_value"] is not None
    assert ks_result["p_value"] < 0.05
    assert wasserstein_result["status"] == "ok"
    assert wasserstein_result["value"] == pytest.approx(1_000.0)


def test_non_finite_values_are_removed_and_counted() -> None:
    reference = pd.Series([0.0, 1.0, np.nan, np.inf, -np.inf])
    current = pd.Series([1.0, 2.0, 3.0, np.nan])

    ks_result = ks_test(reference, current)
    wasserstein_result = wasserstein(reference, current)

    expected_ks = stats.ks_2samp([0.0, 1.0], [1.0, 2.0, 3.0])
    expected_wasserstein = stats.wasserstein_distance(
        [0.0, 1.0], [1.0, 2.0, 3.0]
    )

    assert ks_result["value"] == pytest.approx(expected_ks.statistic)
    assert ks_result["p_value"] == pytest.approx(expected_ks.pvalue)
    assert wasserstein_result["value"] == pytest.approx(expected_wasserstein)
    assert ks_result["details"] == {
        "n_reference_original": 5,
        "n_current_original": 4,
        "n_reference_valid": 2,
        "n_current_valid": 3,
        "n_reference_missing": 1,
        "n_current_missing": 1,
        "n_reference_infinite": 2,
        "n_current_infinite": 0,
        "reference_dtype": "float64",
        "current_dtype": "float64",
        "alternative": "two-sided",
        "method": "auto",
        "statistic_location": pytest.approx(1.0),
        "statistic_sign": 1,
    }


@pytest.mark.parametrize("calculation", [ks_test, wasserstein])
def test_nullable_numeric_dtype_is_supported(calculation) -> None:
    reference = pd.Series([1, pd.NA, 2, 3], dtype="Int64")
    current = pd.Series([2.0, 3.0, pd.NA, 4.0], dtype="Float64")

    result = calculation(reference, current)

    assert result["status"] == "ok"
    assert result["details"]["n_reference_valid"] == 3
    assert result["details"]["n_current_valid"] == 3
    assert result["details"]["n_reference_missing"] == 1
    assert result["details"]["n_current_missing"] == 1
    assert result["details"]["reference_dtype"] == "Int64"
    assert result["details"]["current_dtype"] == "Float64"


@pytest.mark.parametrize("calculation", [ks_test, wasserstein])
@pytest.mark.parametrize(
    ("reference", "current", "expected_valid"),
    [
        (pd.Series([], dtype="float64"), pd.Series([1.0, 2.0]), (0, 2)),
        (pd.Series([np.nan, np.nan]), pd.Series([1.0, 2.0]), (0, 2)),
        (pd.Series([1.0]), pd.Series([2.0, 3.0]), (1, 2)),
    ],
)
def test_insufficient_samples_are_skipped(
    calculation,
    reference: pd.Series,
    current: pd.Series,
    expected_valid: tuple[int, int],
) -> None:
    result = calculation(reference, current)

    assert result["status"] == "skipped"
    assert result["value"] is None
    assert result["p_value"] is None
    assert result["alert"] is None
    assert result["reason"] is not None
    assert "At least 2 finite values" in result["reason"]
    assert result["details"]["n_reference_valid"] == expected_valid[0]
    assert result["details"]["n_current_valid"] == expected_valid[1]


def test_constant_features_and_unequal_sizes_match_scipy() -> None:
    constant_reference = pd.Series([5.0] * 4)
    constant_current = pd.Series([5.0] * 7)
    unequal_reference = pd.Series([0.0, 1.0, 3.0])
    unequal_current = pd.Series([1.0, 2.0, 4.0, 8.0, 9.0])

    constant_ks = ks_test(constant_reference, constant_current)
    constant_wasserstein = wasserstein(constant_reference, constant_current)
    unequal_ks = ks_test(unequal_reference, unequal_current)
    unequal_wasserstein = wasserstein(unequal_reference, unequal_current)

    expected_ks = stats.ks_2samp(unequal_reference, unequal_current)
    expected_wasserstein = stats.wasserstein_distance(
        unequal_reference, unequal_current
    )

    assert constant_ks["value"] == pytest.approx(0.0)
    assert constant_ks["p_value"] == pytest.approx(1.0)
    assert constant_wasserstein["value"] == pytest.approx(0.0)
    assert unequal_ks["value"] == pytest.approx(expected_ks.statistic)
    assert unequal_ks["p_value"] == pytest.approx(expected_ks.pvalue)
    assert unequal_wasserstein["value"] == pytest.approx(expected_wasserstein)


@pytest.mark.parametrize("calculation", [ks_test, wasserstein])
@pytest.mark.parametrize(
    "unsupported",
    [
        pd.Series(["1", "2"]),
        pd.Series([True, False]),
        pd.Series([1 + 1j, 2 + 2j]),
    ],
)
def test_unsupported_dtype_is_skipped(calculation, unsupported: pd.Series) -> None:
    result = calculation(unsupported, pd.Series([1.0, 2.0]))

    assert result["status"] == "skipped"
    assert result["value"] is None
    assert result["alert"] is None
    assert result["reason"] is not None
    assert "unsupported dtype" in result["reason"]
    assert result["details"]["n_reference_valid"] == 0
    assert result["details"]["n_current_valid"] == 2
    assert result["details"]["n_reference_infinite"] is None


@pytest.mark.parametrize("calculation", [ks_test, wasserstein])
def test_invalid_container_type_raises_type_error(calculation) -> None:
    with pytest.raises(TypeError, match="pandas.Series"):
        calculation([1.0, 2.0], pd.Series([1.0, 2.0]))


@pytest.mark.parametrize("calculation", [ks_test, wasserstein])
def test_inputs_are_not_modified_and_result_is_json_safe(calculation) -> None:
    reference = pd.Series([1.0, np.nan, np.inf, 2.0])
    current = pd.Series([2.0, 3.0, 4.0])
    original_reference = reference.copy(deep=True)
    original_current = current.copy(deep=True)

    result = calculation(reference, current)

    pd.testing.assert_series_equal(reference, original_reference)
    pd.testing.assert_series_equal(current, original_current)
    json.dumps(result, allow_nan=False)
