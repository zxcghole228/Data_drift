"""Tests for the categorical Pearson chi-square drift check."""

from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2_contingency

from data_drift_guardian.drift.categorical import chi_square


def _series(counts: dict[object, int], *, missing: int = 0) -> pd.Series:
    values: list[object] = []
    for category, count in counts.items():
        values.extend([category] * count)
    values.extend([None] * missing)
    return pd.Series(values, dtype="object")


def test_known_table_matches_scipy_without_yates_correction() -> None:
    reference = _series({"a": 10, "b": 10, "c": 20})
    current = _series({"a": 20, "b": 20, "c": 20})

    result = chi_square(reference, current)
    expected = chi2_contingency([[10, 10, 20], [20, 20, 20]], correction=False)

    assert result["status"] == "ok"
    assert result["value"] == pytest.approx(expected.statistic)
    assert result["p_value"] == pytest.approx(expected.pvalue)
    assert result["details"]["degrees_of_freedom"] == 2
    assert result["alert"] is None


def test_equal_proportions_have_zero_statistic() -> None:
    result = chi_square(
        _series({"north": 20, "south": 30}),
        _series({"north": 40, "south": 60}),
    )

    assert result["status"] == "ok"
    assert result["value"] == pytest.approx(0.0)
    assert result["p_value"] == pytest.approx(1.0)


def test_clear_proportion_shift_has_small_p_value() -> None:
    result = chi_square(
        _series({"north": 90, "south": 10}),
        _series({"north": 10, "south": 90}),
    )

    assert result["status"] == "ok"
    assert result["p_value"] is not None
    assert result["p_value"] < 1e-20


def test_missing_values_are_excluded_and_reported() -> None:
    result = chi_square(
        _series({"a": 20, "b": 20}, missing=3),
        _series({"a": 20, "b": 20}, missing=4),
    )

    assert result["status"] == "ok"
    assert result["details"]["n_reference_valid"] == 40
    assert result["details"]["n_current_valid"] == 40
    assert result["details"]["n_reference_missing"] == 3
    assert result["details"]["n_current_missing"] == 4


def test_rare_categories_are_pooled_by_combined_count() -> None:
    result = chi_square(
        _series({"a": 50, "b": 50, "r1": 3, "r2": 3}),
        _series({"a": 50, "b": 50, "r1": 3, "r2": 3}),
    )

    assert result["status"] == "ok"
    assert result["details"]["categories"][-1] == "<pooled rare categories>"
    assert set(result["details"]["pooled_categories"]) == {
        "str:'r1'",
        "str:'r2'",
    }
    assert result["details"]["observed_counts"]["reference"][-1] == 6


def test_sparse_table_is_skipped_instead_of_returning_unreliable_p_value() -> None:
    result = chi_square(
        _series({"a": 30, "b": 30}),
        _series({"a": 30, "b": 30, "new": 20}),
    )

    assert result["status"] == "skipped"
    assert result["p_value"] is None
    assert result["alert"] is None
    assert "наблюдаемая частота" in str(result["reason"])
    assert result["details"]["minimum_observed_count"] == 0


def test_all_missing_sample_is_skipped() -> None:
    result = chi_square(
        pd.Series([None, pd.NA], dtype="object"),
        _series({"a": 20, "b": 20}),
    )

    assert result["status"] == "skipped"
    assert result["value"] is None
    assert "хотя бы одно" in str(result["reason"])


def test_single_effective_category_is_skipped() -> None:
    result = chi_square(_series({"same": 20}), _series({"same": 20}))

    assert result["status"] == "skipped"
    assert "меньше двух категорий" in str(result["reason"])


def test_type_aware_categories_do_not_merge_string_and_integer() -> None:
    result = chi_square(
        _series({1: 20, "1": 20}),
        _series({1: 20, "1": 20}),
    )

    assert result["status"] == "ok"
    assert result["details"]["categories"] == ["int:1", "str:'1'"]


def test_custom_frequency_limits_are_applied() -> None:
    result = chi_square(
        _series({"a": 3, "b": 3}),
        _series({"a": 3, "b": 3}),
        {
            "rare_category_min_total": 2,
            "min_observed_count": 3,
            "min_expected_count": 3.0,
        },
    )

    assert result["status"] == "ok"


@pytest.mark.parametrize(
    "config",
    [
        {"rare_category_min_total": 0},
        {"min_observed_count": True},
        {"min_expected_count": np.inf},
        {"unknown": 1},
    ],
)
def test_invalid_config_raises_value_error(config: dict[str, object]) -> None:
    series = _series({"a": 10, "b": 10})
    with pytest.raises(ValueError):
        chi_square(series, series, config)


def test_invalid_inputs_raise_clear_type_errors() -> None:
    categorical = _series({"a": 10, "b": 10})
    with pytest.raises(TypeError, match="pandas.Series"):
        chi_square(["a"], categorical)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="категориальные"):
        chi_square(pd.Series([1, 2]), pd.Series([1, 2]))
    with pytest.raises(TypeError, match="вложенные"):
        chi_square(
            pd.Series([[1], [2]], dtype="object"),
            pd.Series([[1], [2]], dtype="object"),
        )


def test_inputs_are_unchanged_and_result_is_json_safe() -> None:
    reference = _series({"a": 20, "b": 20}, missing=1)
    current = _series({"a": 25, "b": 15}, missing=2)
    original_reference = copy.deepcopy(reference)
    original_current = copy.deepcopy(current)

    result = chi_square(reference, current)

    pd.testing.assert_series_equal(reference, original_reference)
    pd.testing.assert_series_equal(current, original_current)
    json.dumps(result, allow_nan=False)
