"""Тесты Data Quality для Reference и Current."""

from __future__ import annotations

import copy
import json
from typing import Any

import numpy as np
import pandas as pd
import pytest
from data_drift_guardian.quality import check_quality


@pytest.fixture
def valid_config() -> dict[str, Any]:
    return {
        "contract_version": "0.1",
        "random_seed": 42,
        "schema": {
            "allow_extra_columns": False,
            "features": {
                "age": {
                    "kind": "numeric",
                    "nullable": True,
                    "min": 18,
                    "max": 100,
                },
                "income": {
                    "kind": "numeric",
                    "nullable": True,
                    "min": 0,
                    "max": 1_000_000,
                },
                "region": {"kind": "categorical", "nullable": True},
            },
        },
        "quality": {
            "max_missing_fraction": 0.10,
            "max_missing_increase_pp": 5.0,
            "max_duplicate_fraction": 0.01,
        },
        "drift": {
            "numeric_methods": ["ks", "wasserstein", "psi", "js"],
            "categorical_methods": ["chi2", "psi", "js"],
            "alpha": 0.05,
            "multiple_testing": "bh",
            "n_bins": 10,
            "psi_smoothing": 0.000001,
            "js_base": 2,
            "distance_thresholds": {
                "wasserstein": None,
                "psi": None,
                "js": None,
            },
        },
        "adversarial": {
            "enabled": False,
            "n_splits": 3,
            "roc_auc_threshold": None,
            "exclude_columns": [],
        },
    }


@pytest.fixture
def valid_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    reference = pd.DataFrame(
        {
            "age": pd.Series([18, 30, 55, 100], dtype="Int64"),
            "income": pd.Series([100.0, 200.0, 300.0, 400.0], dtype="Float64"),
            "region": pd.Series(["north", "south", "east", "west"], dtype="string"),
        }
    )
    current = pd.DataFrame(
        {
            "age": pd.Series([20.0, 40.0, 60.0], dtype="float64"),
            "income": pd.Series([110, 210, 310], dtype="int64"),
            "region": pd.Series(["north", "east", "south"], dtype="category"),
        }
    )
    return reference, current


def _check(result: dict[str, Any], feature: str, name: str) -> dict[str, Any]:
    return next(
        check for check in result["feature_checks"][feature] if check["name"] == name
    )


def _missingness_frame(missing_count: int, rows: int = 100) -> pd.DataFrame:
    income = np.arange(rows, dtype=float)
    income[:missing_count] = np.nan
    return pd.DataFrame(
        {
            "age": 18 + np.arange(rows) % 83,
            "income": income,
            "region": pd.Series([f"region-{index % 3}" for index in range(rows)]),
        }
    )


def test_healthy_data_returns_ok_json_safe_result_without_mutation(
    valid_config: dict[str, Any],
    valid_frames: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    reference, current = valid_frames
    original_reference = reference.copy(deep=True)
    original_current = current.copy(deep=True)

    result = check_quality(reference, current, valid_config)

    assert result["status"] == "ok"
    assert result["reason"] is None
    assert [check["name"] for check in result["dataset_checks"]] == [
        "reference_duplicate_fraction",
        "current_duplicate_fraction",
    ]
    assert list(result["feature_checks"]) == ["age", "income", "region"]
    assert all(
        check["alert"] is False
        for checks in result["feature_checks"].values()
        for check in checks
        if check["status"] != "skipped"
    )
    pd.testing.assert_frame_equal(reference, original_reference)
    pd.testing.assert_frame_equal(current, original_current)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    ("current_missing", "expected_increase", "expected_alert", "expected_status"),
    [
        (7, 5.0, False, "ok"),
        (8, 6.0, True, "warning"),
    ],
)
def test_missing_increase_uses_percentage_points_and_respects_boundary(
    valid_config: dict[str, Any],
    current_missing: int,
    expected_increase: float,
    expected_alert: bool,
    expected_status: str,
) -> None:
    reference = _missingness_frame(2)
    current = _missingness_frame(current_missing)

    result = check_quality(reference, current, valid_config)
    increase = _check(result, "income", "missing_increase_pp")

    assert increase["value"] == pytest.approx(expected_increase)
    assert increase["threshold"] == 5.0
    assert increase["alert"] is expected_alert
    assert increase["status"] == expected_status
    assert result["status"] == expected_status


def test_missing_fraction_equal_to_maximum_is_not_an_alert(
    valid_config: dict[str, Any],
) -> None:
    reference = _missingness_frame(10)
    current = _missingness_frame(10)

    result = check_quality(reference, current, valid_config)

    reference_missing = _check(result, "income", "reference_missing_fraction")
    current_missing = _check(result, "income", "current_missing_fraction")
    assert reference_missing["value"] == 0.10
    assert reference_missing["alert"] is False
    assert current_missing["value"] == 0.10
    assert current_missing["alert"] is False
    assert result["status"] == "ok"


def test_fully_missing_numeric_column_reports_missing_and_skips_ranges(
    valid_config: dict[str, Any],
    valid_frames: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    reference, current = valid_frames
    current["income"] = pd.Series([pd.NA] * len(current), dtype="Float64")

    result = check_quality(reference, current, valid_config)

    missing = _check(result, "income", "current_missing_fraction")
    minimum = _check(result, "income", "current_min")
    maximum = _check(result, "income", "current_max")
    assert missing["value"] == 1.0
    assert missing["details"]["missing_count"] == len(current)
    assert missing["status"] == "warning"
    assert minimum["status"] == "skipped"
    assert minimum["alert"] is None
    assert maximum["status"] == "skipped"
    assert result["status"] == "warning"


def test_non_nullable_feature_with_missing_value_is_critical(
    valid_config: dict[str, Any],
    valid_frames: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    reference, current = valid_frames
    valid_config["schema"]["features"]["income"]["nullable"] = False
    current["income"] = pd.Series([110.0, pd.NA, 310.0], dtype="Float64")

    result = check_quality(reference, current, valid_config)
    missing = _check(result, "income", "current_missing_fraction")

    assert missing["threshold"] == 0.0
    assert missing["alert"] is True
    assert missing["status"] == "critical"
    assert result["status"] == "critical"


def test_duplicate_fraction_counts_only_repeated_rows_after_first(
    valid_config: dict[str, Any],
) -> None:
    reference = pd.DataFrame(
        {
            "age": [20, 20, 20, 30],
            "income": [100.0, 100.0, 100.0, 200.0],
            "region": ["north", "north", "north", "south"],
        }
    )
    current = reference.drop_duplicates().reset_index(drop=True)

    result = check_quality(reference, current, valid_config)
    duplicate = result["dataset_checks"][0]

    assert duplicate["name"] == "reference_duplicate_fraction"
    assert duplicate["details"]["duplicate_count"] == 2
    assert duplicate["details"]["row_count"] == 4
    assert duplicate["value"] == 0.5
    assert duplicate["status"] == "warning"
    assert result["status"] == "warning"


def test_values_exactly_on_min_max_boundaries_are_valid(
    valid_config: dict[str, Any],
    valid_frames: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    reference, current = valid_frames

    result = check_quality(reference, current, valid_config)

    assert _check(result, "age", "reference_min")["value"] == 18.0
    assert _check(result, "age", "reference_min")["alert"] is False
    assert _check(result, "age", "reference_max")["value"] == 100.0
    assert _check(result, "age", "reference_max")["alert"] is False


def test_values_outside_min_max_are_counted_and_critical(
    valid_config: dict[str, Any],
    valid_frames: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    reference, current = valid_frames
    current["age"] = [17.0, 50.0, 101.0]

    result = check_quality(reference, current, valid_config)
    minimum = _check(result, "age", "current_min")
    maximum = _check(result, "age", "current_max")

    assert minimum["value"] == 17.0
    assert minimum["threshold"] == 18
    assert minimum["details"]["violation_count"] == 1
    assert minimum["status"] == "critical"
    assert maximum["value"] == 101.0
    assert maximum["threshold"] == 100
    assert maximum["details"]["violation_count"] == 1
    assert maximum["status"] == "critical"
    assert result["status"] == "critical"


def test_infinities_are_separate_from_missing_and_ignored_by_ranges(
    valid_config: dict[str, Any],
    valid_frames: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    reference, current = valid_frames
    reference["age"] = [18.0, np.inf, -np.inf, 100.0]

    result = check_quality(reference, current, valid_config)
    missing = _check(result, "age", "reference_missing_fraction")
    infinite = _check(result, "age", "reference_infinite_count")
    minimum = _check(result, "age", "reference_min")
    maximum = _check(result, "age", "reference_max")

    assert missing["value"] == 0.0
    assert infinite["value"] == 2
    assert infinite["details"]["positive_infinity_count"] == 1
    assert infinite["details"]["negative_infinity_count"] == 1
    assert infinite["status"] == "critical"
    assert minimum["value"] == 18.0
    assert minimum["alert"] is False
    assert maximum["value"] == 100.0
    assert maximum["alert"] is False


@pytest.mark.parametrize("dataset", ["reference", "current"])
def test_empty_dataframe_returns_structured_error(
    valid_config: dict[str, Any],
    valid_frames: tuple[pd.DataFrame, pd.DataFrame],
    dataset: str,
) -> None:
    reference, current = valid_frames
    if dataset == "reference":
        reference = reference.iloc[0:0]
    else:
        current = current.iloc[0:0]

    result = check_quality(reference, current, valid_config)

    assert result == {
        "status": "error",
        "dataset_checks": [],
        "feature_checks": {},
        "reason": f"Пустые таблицы: {dataset}",
    }


def test_schema_incompatible_feature_is_not_quality_checked(
    valid_config: dict[str, Any],
    valid_frames: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    reference, current = valid_frames
    current["age"] = pd.Series(["20", "40", "60"], dtype="string")

    result = check_quality(reference, current, valid_config)

    assert list(result["feature_checks"]) == ["income", "region"]
    assert result["status"] == "ok"


def test_invalid_dataframe_type_raises_type_error(
    valid_config: dict[str, Any],
    valid_frames: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    _, current = valid_frames

    with pytest.raises(TypeError, match="pandas.DataFrame"):
        check_quality([], current, valid_config)  # type: ignore[arg-type]


def test_invalid_configuration_raises_value_error(
    valid_config: dict[str, Any],
    valid_frames: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    reference, current = valid_frames
    invalid_config = copy.deepcopy(valid_config)
    invalid_config["quality"]["unknown"] = 1

    with pytest.raises(ValueError, match="неизвестные ключи"):
        check_quality(reference, current, invalid_config)
