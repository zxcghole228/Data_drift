"""Проверка колонок и семантических типов табличных данных."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_complex_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_scalar,
    is_string_dtype,
)

from .config import validate_schema_config
from .contracts import FeatureType, SchemaConfig, SchemaResult, TypeMismatch


DatasetName = Literal["reference", "current"]


def _require_dataframes(reference: object, current: object) -> None:
    invalid: list[str] = []
    if not isinstance(reference, pd.DataFrame):
        invalid.append(f"reference={type(reference).__name__}")
    if not isinstance(current, pd.DataFrame):
        invalid.append(f"current={type(current).__name__}")
    if invalid:
        raise TypeError(
            "reference и current должны иметь тип pandas.DataFrame; "
            f"получено: {', '.join(invalid)}"
        )


def _unique_strings(values: Iterable[object]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        label = str(value)
        if label not in seen:
            seen.add(label)
            result.append(label)
    return result


def _duplicate_labels(frame: pd.DataFrame) -> list[object]:
    duplicates = frame.columns[frame.columns.duplicated(keep=False)]
    return list(dict.fromkeys(duplicates.tolist()))


def _contains_nested_values(series: pd.Series) -> bool:
    return any(not is_scalar(value) for value in series.array)


def _is_compatible(series: pd.Series, expected: FeatureType) -> tuple[bool, str]:
    dtype = series.dtype
    actual = str(dtype)

    if expected == "numeric":
        compatible = (
            is_numeric_dtype(dtype)
            and not is_bool_dtype(dtype)
            and not is_complex_dtype(dtype)
        )
        return compatible, actual

    if is_bool_dtype(dtype) or isinstance(dtype, pd.CategoricalDtype):
        return True, actual
    if is_object_dtype(dtype):
        if _contains_nested_values(series):
            return False, f"{actual} (содержит вложенные значения)"
        return True, actual
    return is_string_dtype(dtype), actual


def validate_schema(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    config: SchemaConfig,
) -> SchemaResult:
    """Проверить структуру двух DataFrame без изменения входных данных.

    Проверка касается наличия колонок и совместимости dtype с семантическими
    типами. Пропуски и ограничения ``nullable`` обрабатывает Data Quality.
    Некорректная конфигурация вызывает ``ValueError``; проблемы допустимых
    DataFrame возвращаются в структурированном ``SchemaResult``.
    """

    _require_dataframes(reference, current)
    validated_config = validate_schema_config(config)

    expected_features = list(validated_config["features"])
    expected_set = set(expected_features)
    datasets: tuple[tuple[DatasetName, pd.DataFrame], ...] = (
        ("reference", reference),
        ("current", current),
    )

    duplicate_labels = {
        name: _duplicate_labels(frame) for name, frame in datasets
    }
    duplicate_columns = {
        name: _unique_strings(duplicate_labels[name]) for name, _ in datasets
    }
    missing_columns = {
        name: [feature for feature in expected_features if feature not in frame.columns]
        for name, frame in datasets
    }
    extra_columns = {
        name: _unique_strings(
            column for column in frame.columns if column not in expected_set
        )
        for name, frame in datasets
    }

    type_mismatches: list[TypeMismatch] = []
    incompatible_features: set[str] = set()
    duplicated_expected_features: set[str] = set()

    for dataset_name, frame in datasets:
        duplicated = set(duplicate_labels[dataset_name])
        duplicated_expected_features.update(expected_set & duplicated)

        for feature_name, feature_config in validated_config["features"].items():
            if feature_name not in frame.columns or feature_name in duplicated:
                continue

            column = frame[feature_name]
            compatible, actual = _is_compatible(column, feature_config["kind"])
            if compatible:
                continue

            type_mismatches.append(
                {
                    "column": feature_name,
                    "dataset": dataset_name,
                    "expected": feature_config["kind"],
                    "actual": actual,
                }
            )
            incompatible_features.add(feature_name)

    empty_datasets = [name for name, frame in datasets if frame.empty]
    unavailable_features = (
        set(missing_columns["reference"])
        | set(missing_columns["current"])
        | duplicated_expected_features
        | incompatible_features
    )
    valid_features = [
        feature for feature in expected_features if feature not in unavailable_features
    ]
    if empty_datasets:
        valid_features = []

    problems: list[str] = []
    if empty_datasets:
        problems.append(f"пустые таблицы: {', '.join(empty_datasets)}")
    missing_count = sum(len(columns) for columns in missing_columns.values())
    if missing_count:
        problems.append(f"отсутствующие обязательные колонки: {missing_count}")
    if not validated_config["allow_extra_columns"]:
        extra_count = sum(len(columns) for columns in extra_columns.values())
        if extra_count:
            problems.append(f"запрещённые лишние колонки: {extra_count}")
    duplicate_count = sum(len(columns) for columns in duplicate_columns.values())
    if duplicate_count:
        problems.append(f"повторяющиеся имена колонок: {duplicate_count}")
    if type_mismatches:
        problems.append(f"несовместимые типы: {len(type_mismatches)}")

    return {
        "status": "error" if problems else "ok",
        "missing_columns": missing_columns,
        "extra_columns": extra_columns,
        "duplicate_columns": duplicate_columns,
        "type_mismatches": type_mismatches,
        "valid_features": valid_features,
        "reason": (
            "Проверка схемы выявила нарушения: " + "; ".join(problems)
            if problems
            else None
        ),
    }
