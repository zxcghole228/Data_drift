from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from data_drift_guardian.drift.binning import build_probabilities


def test_numeric_bins_are_reference_quantiles_with_infinite_tails() -> None:
    reference = pd.Series([0.0, 1.0, 2.0, 3.0])
    current = pd.Series([-100.0, 0.0, 1.0, 100.0])

    reference_probabilities, current_probabilities, details = build_probabilities(
        reference,
        current,
        {"kind": "numeric", "n_bins": 2},
    )

    assert reference_probabilities == pytest.approx([0.5, 0.5])
    assert current_probabilities == pytest.approx([0.75, 0.25])
    assert details["bin_edges"] == ["-inf", 1.5, "+inf"]
    assert details["reference_counts"] == [2, 2]
    assert details["current_counts"] == [3, 1]
    assert sum(details["current_counts"]) == len(current)


def test_numeric_cleaning_counts_missing_and_infinite_values() -> None:
    reference = pd.Series([0.0, 1.0, np.nan, np.inf, -np.inf])
    current = pd.Series([0.0, 2.0, 3.0, np.nan])

    reference_probabilities, current_probabilities, details = build_probabilities(
        reference,
        current,
        {"kind": "numeric", "n_bins": 2},
    )

    assert sum(reference_probabilities) == pytest.approx(1.0)
    assert sum(current_probabilities) == pytest.approx(1.0)
    assert details["n_reference_original"] == 5
    assert details["n_reference_valid"] == 2
    assert details["n_reference_missing"] == 1
    assert details["n_reference_infinite"] == 2
    assert details["n_current_valid"] == 3


def test_constant_reference_uses_below_equal_above_cells() -> None:
    reference = pd.Series([5.0, 5.0, 5.0])
    current = pd.Series([4.0, 5.0, 6.0])

    reference_probabilities, current_probabilities, details = build_probabilities(
        reference,
        current,
        {"kind": "numeric", "n_bins": 10},
    )

    assert reference_probabilities == pytest.approx([0.0, 1.0, 0.0])
    assert current_probabilities == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    assert details["binning_strategy"] == "constant_reference"
    assert details["constant_reference_value"] == 5.0
    assert details["actual_n_bins"] == 3


def test_categorical_union_preserves_new_current_category() -> None:
    reference = pd.Series(["a", "a", "b", None], dtype="object")
    current = pd.Series(["b", "c", "c", pd.NA], dtype="object")

    reference_probabilities, current_probabilities, details = build_probabilities(
        reference,
        current,
        {"kind": "categorical"},
    )

    assert reference_probabilities == pytest.approx([2 / 3, 1 / 3, 0.0])
    assert current_probabilities == pytest.approx([0.0, 1 / 3, 2 / 3])
    assert [item["value"] for item in details["categories"]] == ["a", "b", "c"]
    assert [item["value"] for item in details["new_categories_in_current"]] == ["c"]
    assert details["n_reference_missing"] == 1
    assert details["n_current_missing"] == 1


def test_nullable_numeric_dtypes_are_supported() -> None:
    reference = pd.Series([1, 2, pd.NA, 3], dtype="Int64")
    current = pd.Series([1.0, pd.NA, 4.0], dtype="Float64")

    _, _, details = build_probabilities(
        reference,
        current,
        {"kind": "numeric", "n_bins": 3},
    )

    assert details["reference_dtype"] == "Int64"
    assert details["current_dtype"] == "Float64"
    assert details["n_reference_valid"] == 3
    assert details["n_current_valid"] == 2


@pytest.mark.parametrize(
    ("reference", "current", "config", "error_type", "message"),
    [
        (
            pd.Series([np.nan], dtype="float64"),
            pd.Series([1.0]),
            {"kind": "numeric"},
            ValueError,
            "no finite values",
        ),
        (
            pd.Series([None], dtype="object"),
            pd.Series(["a"], dtype="object"),
            {"kind": "categorical"},
            ValueError,
            "no non-missing",
        ),
        (
            pd.Series(["1", "2"]),
            pd.Series(["1", "3"]),
            {"kind": "numeric"},
            TypeError,
            "real numeric dtype",
        ),
        (
            pd.Series([1.0, 2.0]),
            pd.Series([1.0, 3.0]),
            {"kind": "categorical"},
            TypeError,
            "categorical-compatible dtype",
        ),
    ],
)
def test_unusable_series_raise_clear_error(
    reference: pd.Series,
    current: pd.Series,
    config: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        build_probabilities(reference, current, config)


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({}, "kind"),
        ({"kind": "unknown"}, "kind"),
        ({"kind": "numeric", "n_bins": True}, "integer"),
        ({"kind": "numeric", "n_bins": 1}, "at least 2"),
    ],
)
def test_invalid_binning_config_raises_value_error(
    config: dict[str, object],
    message: str,
) -> None:
    values = pd.Series([1.0, 2.0])
    with pytest.raises(ValueError, match=message):
        build_probabilities(values, values, config)


def test_inputs_are_unchanged_and_output_is_json_safe() -> None:
    reference = pd.Series([1.0, np.nan, 2.0, np.inf])
    current = pd.Series([-10.0, 1.0, 3.0, 20.0])
    original_reference = reference.copy(deep=True)
    original_current = current.copy(deep=True)

    result = build_probabilities(
        reference,
        current,
        {"kind": "numeric", "n_bins": 3},
    )

    pd.testing.assert_series_equal(reference, original_reference)
    pd.testing.assert_series_equal(current, original_current)
    json.dumps(result, allow_nan=False)
