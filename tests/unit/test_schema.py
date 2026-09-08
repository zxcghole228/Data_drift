"""Тесты проверки схемы Reference и Current."""

from __future__ import annotations

import copy
import json
from typing import Any

import pandas as pd
import pytest

from data_drift_guardian.schema import validate_schema


@pytest.fixture
def schema_config() -> dict[str, Any]:
    return {
        "allow_extra_columns": False,
        "features": {
            "age": {"kind": "numeric", "nullable": True},
            "income": {"kind": "numeric", "nullable": True},
            "region": {"kind": "categorical", "nullable": True},
        },
    }


@pytest.fixture
def valid_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    reference = pd.DataFrame(
        {
            "age": pd.Series([20, 30, 40], dtype="int64"),
            "income": pd.Series([100, pd.NA, 300], dtype="Int64"),
            "region": pd.Series(["north", "south", "west"], dtype="string"),
        }
    )
    current = pd.DataFrame(
        {
            "age": pd.Series([21.0, 31.0], dtype="float64"),
            "income": pd.Series([110.0, pd.NA], dtype="Float64"),
            "region": pd.Series(["north", "east"], dtype="category"),
        }
    )
    return reference, current


def test_matching_schema_accepts_physical_dtype_differences_and_is_json_safe(
    valid_frames: tuple[pd.DataFrame, pd.DataFrame],
    schema_config: dict[str, Any],
) -> None:
    reference, current = valid_frames
    original_reference = reference.copy(deep=True)
    original_current = current.copy(deep=True)

    result = validate_schema(reference, current, schema_config)  # type: ignore[arg-type]

    assert result == {
        "status": "ok",
        "missing_columns": {"reference": [], "current": []},
        "extra_columns": {"reference": [], "current": []},
        "duplicate_columns": {"reference": [], "current": []},
        "type_mismatches": [],
        "valid_features": ["age", "income", "region"],
        "reason": None,
    }
    pd.testing.assert_frame_equal(reference, original_reference)
    pd.testing.assert_frame_equal(current, original_current)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("dataset_name", ["reference", "current"])
def test_missing_column_is_reported_only_for_affected_dataset(
    valid_frames: tuple[pd.DataFrame, pd.DataFrame],
    schema_config: dict[str, Any],
    dataset_name: str,
) -> None:
    reference, current = valid_frames
    if dataset_name == "reference":
        reference = reference.drop(columns="region")
    else:
        current = current.drop(columns="region")

    result = validate_schema(reference, current, schema_config)  # type: ignore[arg-type]

    other_name = "current" if dataset_name == "reference" else "reference"
    assert result["status"] == "error"
    assert result["missing_columns"][dataset_name] == ["region"]
    assert result["missing_columns"][other_name] == []
    assert result["valid_features"] == ["age", "income"]


@pytest.mark.parametrize(
    ("allow_extra", "expected_status"),
    [(True, "ok"), (False, "error")],
)
def test_extra_column_respects_configuration(
    valid_frames: tuple[pd.DataFrame, pd.DataFrame],
    schema_config: dict[str, Any],
    allow_extra: bool,
    expected_status: str,
) -> None:
    reference, current = valid_frames
    current = current.assign(extra=[1, 2])
    schema_config["allow_extra_columns"] = allow_extra

    result = validate_schema(reference, current, schema_config)  # type: ignore[arg-type]

    assert result["status"] == expected_status
    assert result["extra_columns"] == {"reference": [], "current": ["extra"]}
    assert result["valid_features"] == ["age", "income", "region"]


def test_wrong_semantic_type_excludes_only_affected_feature(
    valid_frames: tuple[pd.DataFrame, pd.DataFrame],
    schema_config: dict[str, Any],
) -> None:
    reference, current = valid_frames
    current["age"] = pd.Series(["21", "31"], dtype="string")

    result = validate_schema(reference, current, schema_config)  # type: ignore[arg-type]

    assert result["status"] == "error"
    assert result["type_mismatches"] == [
        {
            "column": "age",
            "dataset": "current",
            "expected": "numeric",
            "actual": "string",
        }
    ]
    assert result["valid_features"] == ["income", "region"]


def test_bool_is_compatible_with_categorical() -> None:
    config = {
        "allow_extra_columns": False,
        "features": {"active": {"kind": "categorical", "nullable": True}},
    }
    reference = pd.DataFrame({"active": pd.Series([True, False], dtype="bool")})
    current = pd.DataFrame({"active": pd.Series([True, pd.NA], dtype="boolean")})

    result = validate_schema(reference, current, config)  # type: ignore[arg-type]

    assert result["status"] == "ok"
    assert result["valid_features"] == ["active"]


@pytest.mark.parametrize("dataset_name", ["reference", "current"])
def test_empty_dataframe_is_an_error_and_disables_all_features(
    valid_frames: tuple[pd.DataFrame, pd.DataFrame],
    schema_config: dict[str, Any],
    dataset_name: str,
) -> None:
    reference, current = valid_frames
    if dataset_name == "reference":
        reference = reference.iloc[0:0]
    else:
        current = current.iloc[0:0]

    result = validate_schema(reference, current, schema_config)  # type: ignore[arg-type]

    assert result["status"] == "error"
    assert result["valid_features"] == []
    assert result["reason"] is not None
    assert f"пустые таблицы: {dataset_name}" in result["reason"]


@pytest.mark.parametrize("dataset_name", ["reference", "current"])
def test_duplicate_expected_column_is_reported_before_dtype_access(
    valid_frames: tuple[pd.DataFrame, pd.DataFrame],
    schema_config: dict[str, Any],
    dataset_name: str,
) -> None:
    reference, current = valid_frames
    source = reference if dataset_name == "reference" else current
    duplicated = pd.concat([source, source[["age"]]], axis=1)
    if dataset_name == "reference":
        reference = duplicated
    else:
        current = duplicated

    result = validate_schema(reference, current, schema_config)  # type: ignore[arg-type]

    assert result["status"] == "error"
    assert result["duplicate_columns"][dataset_name] == ["age"]
    assert result["valid_features"] == ["income", "region"]


def test_datetime_and_complex_are_not_numeric(
    schema_config: dict[str, Any],
) -> None:
    reference = pd.DataFrame(
        {
            "age": pd.Series([1 + 1j, 2 + 2j], dtype="complex128"),
            "income": pd.Series([100.0, 200.0]),
            "region": pd.Series(["north", "south"], dtype="string"),
        }
    )
    current = pd.DataFrame(
        {
            "age": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "income": pd.Series([110.0, 210.0]),
            "region": pd.Series(["north", "south"], dtype="string"),
        }
    )

    result = validate_schema(reference, current, schema_config)  # type: ignore[arg-type]

    assert result["status"] == "error"
    assert [item["dataset"] for item in result["type_mismatches"]] == [
        "reference",
        "current",
    ]
    assert result["valid_features"] == ["income", "region"]


def test_nested_object_values_are_not_treated_as_categories() -> None:
    config = {
        "allow_extra_columns": False,
        "features": {"payload": {"kind": "categorical", "nullable": True}},
    }
    reference = pd.DataFrame(
        {"payload": pd.Series([[1, 2], [3]], dtype="object")}
    )
    current = pd.DataFrame(
        {"payload": pd.Series([{"key": 1}, {"key": 2}], dtype="object")}
    )

    result = validate_schema(reference, current, config)  # type: ignore[arg-type]

    assert result["status"] == "error"
    assert len(result["type_mismatches"]) == 2
    assert result["valid_features"] == []
    assert all(
        "вложенные значения" in mismatch["actual"]
        for mismatch in result["type_mismatches"]
    )


def test_invalid_dataframe_type_raises_type_error(
    valid_frames: tuple[pd.DataFrame, pd.DataFrame],
    schema_config: dict[str, Any],
) -> None:
    _, current = valid_frames

    with pytest.raises(TypeError, match="pandas.DataFrame"):
        validate_schema([], current, schema_config)  # type: ignore[arg-type]


def test_invalid_schema_config_raises_value_error(
    valid_frames: tuple[pd.DataFrame, pd.DataFrame],
    schema_config: dict[str, Any],
) -> None:
    reference, current = valid_frames
    invalid_config = copy.deepcopy(schema_config)
    invalid_config["unknown"] = True

    with pytest.raises(ValueError, match="неизвестные ключи"):
        validate_schema(reference, current, invalid_config)  # type: ignore[arg-type]
