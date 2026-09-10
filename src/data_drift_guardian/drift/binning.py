"""Общее дискретное представление признаков."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_complex_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)

from ..contracts import FeatureType

DEFAULT_N_BINS = 10


def _require_series(reference: object, current: object) -> None:
    invalid: list[str] = []
    if not isinstance(reference, pd.Series):
        invalid.append(f"reference={type(reference).__name__}")
    if not isinstance(current, pd.Series):
        invalid.append(f"current={type(current).__name__}")
    if invalid:
        raise TypeError(
            "reference and current must be pandas.Series objects; received "
            + ", ".join(invalid)
        )


def _is_real_numeric(series: pd.Series) -> bool:
    return (
        is_numeric_dtype(series.dtype)
        and not is_bool_dtype(series.dtype)
        and not is_complex_dtype(series.dtype)
    )


def _is_categorical(series: pd.Series) -> bool:
    return (
        is_bool_dtype(series.dtype)
        or is_object_dtype(series.dtype)
        or is_string_dtype(series.dtype)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )


def _clean_numeric(
    series: pd.Series,
    label: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not _is_real_numeric(series):
        raise TypeError(
            f"{label} must have a real numeric dtype, received {series.dtype}"
        )

    values = series.to_numpy(dtype=np.float64, na_value=np.nan)
    missing = np.isnan(values)
    infinite = np.isinf(values)
    clean = values[~(missing | infinite)]
    if clean.size == 0:
        raise ValueError(f"{label} has no finite values")

    details = {
        f"n_{label}_original": int(values.size),
        f"n_{label}_valid": int(clean.size),
        f"n_{label}_missing": int(missing.sum()),
        f"n_{label}_infinite": int(infinite.sum()),
        f"{label}_dtype": str(series.dtype),
    }
    return clean, details


def _probabilities(counts: np.ndarray) -> list[float]:
    total = int(counts.sum())
    if total <= 0:
        raise ValueError("A probability distribution requires at least one value")
    return [float(value) for value in counts / total]


def _build_numeric_probabilities(
    reference: pd.Series,
    current: pd.Series,
    n_bins: int,
) -> tuple[list[float], list[float], dict[str, Any]]:
    reference_values, reference_details = _clean_numeric(reference, "reference")
    current_values, current_details = _clean_numeric(current, "current")

    reference_constant = float(reference_values[0])
    if np.all(reference_values == reference_constant):
        reference_counts = np.array(
            [
                np.count_nonzero(reference_values < reference_constant),
                np.count_nonzero(reference_values == reference_constant),
                np.count_nonzero(reference_values > reference_constant),
            ],
            dtype=np.int64,
        )
        current_counts = np.array(
            [
                np.count_nonzero(current_values < reference_constant),
                np.count_nonzero(current_values == reference_constant),
                np.count_nonzero(current_values > reference_constant),
            ],
            dtype=np.int64,
        )
        details: dict[str, Any] = {
            "kind": "numeric",
            "binning_strategy": "constant_reference",
            "requested_n_bins": int(n_bins),
            "actual_n_bins": 3,
            "constant_reference_value": reference_constant,
            "bin_labels": [
                "below_reference_value",
                "equal_reference_value",
                "above_reference_value",
            ],
        }
    else:
        quantile_levels = np.linspace(0.0, 1.0, n_bins + 1)
        quantile_edges = np.quantile(
            reference_values,
            quantile_levels,
            method="linear",
        )
        interior_edges = np.unique(quantile_edges[1:-1])
        bin_edges = np.concatenate(([-np.inf], interior_edges, [np.inf]))
        reference_counts, _ = np.histogram(reference_values, bins=bin_edges)
        current_counts, _ = np.histogram(current_values, bins=bin_edges)

        details = {
            "kind": "numeric",
            "binning_strategy": "reference_quantiles",
            "quantile_method": "linear",
            "requested_n_bins": int(n_bins),
            "actual_n_bins": int(bin_edges.size - 1),
            "bin_edges": [
                "-inf",
                *[float(value) for value in interior_edges],
                "+inf",
            ],
        }

    details.update(reference_details)
    details.update(current_details)
    details["reference_counts"] = [int(value) for value in reference_counts]
    details["current_counts"] = [int(value) for value in current_counts]
    return (
        _probabilities(reference_counts),
        _probabilities(current_counts),
        details,
    )


def _to_python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _category_key(value: Any) -> tuple[str, str]:
    value = _to_python_scalar(value)
    try:
        hash(value)
    except TypeError as exc:
        raise TypeError(
            f"Categorical values must be hashable, received {type(value).__name__}"
        ) from exc
    type_name = f"{type(value).__module__}.{type(value).__qualname__}"
    return type_name, repr(value)


def _json_safe_category(value: Any) -> str | int | float | bool:
    value = _to_python_scalar(value)
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and np.isfinite(value):
        return value
    return str(value)


def _build_categorical_probabilities(
    reference: pd.Series,
    current: pd.Series,
) -> tuple[list[float], list[float], dict[str, Any]]:
    for label, series in (("reference", reference), ("current", current)):
        if not _is_categorical(series):
            raise TypeError(
                f"{label} must have a categorical-compatible dtype, "
                f"received {series.dtype}"
            )

    reference_values = reference[~reference.isna()].tolist()
    current_values = current[~current.isna()].tolist()
    if not reference_values:
        raise ValueError("reference has no non-missing categorical values")
    if not current_values:
        raise ValueError("current has no non-missing categorical values")

    ordered_keys: list[tuple[str, str]] = []
    category_records: dict[tuple[str, str], dict[str, Any]] = {}
    reference_counts_by_key: dict[tuple[str, str], int] = {}
    current_counts_by_key: dict[tuple[str, str], int] = {}

    for label, values, counts in (
        ("reference", reference_values, reference_counts_by_key),
        ("current", current_values, current_counts_by_key),
    ):
        for value in values:
            key = _category_key(value)
            if key not in category_records:
                ordered_keys.append(key)
                category_records[key] = {
                    "value": _json_safe_category(value),
                    "type": key[0],
                    "first_seen_in": label,
                }
            counts[key] = counts.get(key, 0) + 1

    reference_counts = np.array(
        [reference_counts_by_key.get(key, 0) for key in ordered_keys],
        dtype=np.int64,
    )
    current_counts = np.array(
        [current_counts_by_key.get(key, 0) for key in ordered_keys],
        dtype=np.int64,
    )
    new_category_keys = [
        key for key in ordered_keys if key not in reference_counts_by_key
    ]

    details = {
        "kind": "categorical",
        "binning_strategy": "ordered_union",
        "actual_n_bins": len(ordered_keys),
        "categories": [category_records[key] for key in ordered_keys],
        "new_categories_in_current": [
            category_records[key] for key in new_category_keys
        ],
        "n_reference_original": len(reference),
        "n_current_original": len(current),
        "n_reference_valid": len(reference_values),
        "n_current_valid": len(current_values),
        "n_reference_missing": int(reference.isna().sum()),
        "n_current_missing": int(current.isna().sum()),
        "reference_dtype": str(reference.dtype),
        "current_dtype": str(current.dtype),
        "reference_counts": [int(value) for value in reference_counts],
        "current_counts": [int(value) for value in current_counts],
    }
    return (
        _probabilities(reference_counts),
        _probabilities(current_counts),
        details,
    )


def build_probabilities(
    reference: pd.Series,
    current: pd.Series,
    config: Mapping[str, Any],
) -> tuple[list[float], list[float], dict[str, Any]]:
    """Построить сопоставимые вероятности для одного признака.

    ``config`` — подготовленная конфигурация признака с обязательным ключом
    ``kind`` (``numeric`` или ``categorical``). Для числового признака ключ
    ``n_bins`` необязателен и по умолчанию равен 10. Числовые границы
    строятся только по Reference, а категориальные ячейки — по
    упорядоченному объединению категорий Reference и Current.
    """
    _require_series(reference, current)
    if not isinstance(config, Mapping):
        raise TypeError(f"config must be a mapping, received {type(config).__name__}")

    kind = config.get("kind")
    if kind not in ("numeric", "categorical"):
        raise ValueError("config['kind'] must be 'numeric' or 'categorical'")
    feature_type: FeatureType = kind

    if feature_type == "categorical":
        return _build_categorical_probabilities(reference, current)

    raw_n_bins = config.get("n_bins", DEFAULT_N_BINS)
    if isinstance(raw_n_bins, bool) or not isinstance(raw_n_bins, Integral):
        # Invalid configuration is ValueError by the shared API contract.
        raise ValueError("config['n_bins'] must be an integer")  # noqa: TRY004
    n_bins = int(raw_n_bins)
    if n_bins < 2:
        raise ValueError("config['n_bins'] must be at least 2")
    return _build_numeric_probabilities(reference, current, n_bins)
