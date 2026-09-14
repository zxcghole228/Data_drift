"""Pearson chi-square test for categorical distribution drift."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_object_dtype,
    is_string_dtype,
)
from scipy.stats import chi2_contingency

from ..contracts import CheckResult, Status

DEFAULT_RARE_CATEGORY_MIN_TOTAL = 10
DEFAULT_MIN_OBSERVED_COUNT = 5
DEFAULT_MIN_EXPECTED_COUNT = 5.0

_CONFIG_KEYS = frozenset(
    {
        "rare_category_min_total",
        "min_observed_count",
        "min_expected_count",
    }
)
_POOLED_KEY = ("__internal__", "pooled_rare_categories")


def _result(
    status: Status,
    reason: str | None,
    *,
    value: float | None = None,
    p_value: float | None = None,
    details: dict[str, Any] | None = None,
) -> CheckResult:
    return {
        "name": "chi2",
        "status": status,
        "value": value,
        "threshold": None,
        "p_value": p_value,
        "adjusted_p_value": None,
        "alert": None,
        "reason": reason,
        "details": {} if details is None else details,
    }


def _require_series(reference: object, current: object) -> None:
    invalid: list[str] = []
    if not isinstance(reference, pd.Series):
        invalid.append(f"reference={type(reference).__name__}")
    if not isinstance(current, pd.Series):
        invalid.append(f"current={type(current).__name__}")
    if invalid:
        raise TypeError(
            "reference и current должны иметь тип pandas.Series; "
            f"получено: {', '.join(invalid)}"
        )


def _is_categorical(series: pd.Series) -> bool:
    return bool(
        is_object_dtype(series.dtype)
        or is_string_dtype(series.dtype)
        or isinstance(series.dtype, pd.CategoricalDtype)
        or is_bool_dtype(series.dtype)
    )


def _contains_nested_values(series: pd.Series) -> bool:
    nested_types = (list, dict, set, np.ndarray)
    return any(isinstance(value, nested_types) for value in series.dropna().tolist())


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} должен иметь тип int")
    normalized = int(value)
    if normalized < 1:
        raise ValueError(f"{name} должен быть не меньше 1")
    return normalized


def _positive_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} должен быть конечным положительным числом")
    normalized = float(value)
    if not np.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} должен быть конечным положительным числом")
    return normalized


def _normalize_config(config: Mapping[str, Any]) -> tuple[int, int, float]:
    if not isinstance(config, Mapping):
        raise TypeError("config должен быть отображением ключ-значение")
    unknown = sorted(str(key) for key in set(config) - _CONFIG_KEYS)
    if unknown:
        raise ValueError(f"Неизвестные параметры chi2: {', '.join(unknown)}")

    rare_min_total = _positive_int(
        config.get("rare_category_min_total", DEFAULT_RARE_CATEGORY_MIN_TOTAL),
        "rare_category_min_total",
    )
    min_observed = _positive_int(
        config.get("min_observed_count", DEFAULT_MIN_OBSERVED_COUNT),
        "min_observed_count",
    )
    min_expected = _positive_float(
        config.get("min_expected_count", DEFAULT_MIN_EXPECTED_COUNT),
        "min_expected_count",
    )
    return rare_min_total, min_observed, min_expected


def _category_key(value: Any) -> tuple[str, str]:
    python_value = value.item() if isinstance(value, np.generic) else value
    return type(python_value).__name__, repr(python_value)


def _category_label(key: tuple[str, str]) -> str:
    if key == _POOLED_KEY:
        return "<pooled rare categories>"
    return f"{key[0]}:{key[1]}"


def _counts(series: pd.Series) -> Counter[tuple[str, str]]:
    return Counter(_category_key(value) for value in series.dropna().tolist())


def _ordered_union(
    reference: pd.Series,
    current: pd.Series,
) -> list[tuple[str, str]]:
    ordered: dict[tuple[str, str], None] = {}
    for value in [*reference.dropna().tolist(), *current.dropna().tolist()]:
        ordered.setdefault(_category_key(value), None)
    return list(ordered)


def _pool_counts(
    counts: Counter[tuple[str, str]],
    categories: list[tuple[str, str]],
    rare_categories: set[tuple[str, str]],
) -> list[int]:
    result = [
        counts[category] for category in categories if category not in rare_categories
    ]
    if rare_categories:
        result.append(sum(counts[category] for category in rare_categories))
    return result


def chi_square(
    reference: pd.Series,
    current: pd.Series,
    config: Mapping[str, Any] | None = None,
) -> CheckResult:
    """Compare categorical frequencies with Pearson's chi-square test.

    Missing values are excluded because missingness is handled by Data Quality.
    Categories with a combined count below ``rare_category_min_total`` are
    pooled into one deterministic cell. The asymptotic p-value is returned only
    when every resulting observed and expected cell satisfies the configured
    minimum; otherwise the check is explicitly skipped.
    """

    _require_series(reference, current)
    settings = {} if config is None else config
    rare_min_total, min_observed, min_expected = _normalize_config(settings)

    if not _is_categorical(reference) or not _is_categorical(current):
        raise TypeError("chi2 поддерживает только категориальные pandas dtype")
    if _contains_nested_values(reference) or _contains_nested_values(current):
        raise TypeError("chi2 не поддерживает вложенные значения категорий")

    reference_counts = _counts(reference)
    current_counts = _counts(current)
    n_reference_valid = sum(reference_counts.values())
    n_current_valid = sum(current_counts.values())
    base_details: dict[str, Any] = {
        "n_reference_total": len(reference),
        "n_current_total": len(current),
        "n_reference_valid": int(n_reference_valid),
        "n_current_valid": int(n_current_valid),
        "n_reference_missing": int(reference.isna().sum()),
        "n_current_missing": int(current.isna().sum()),
        "reference_dtype": str(reference.dtype),
        "current_dtype": str(current.dtype),
        "correction": False,
        "rare_category_policy": {
            "method": "pool_by_combined_count",
            "rare_category_min_total": rare_min_total,
            "min_observed_count": min_observed,
            "min_expected_count": min_expected,
        },
    }

    if n_reference_valid == 0 or n_current_valid == 0:
        return _result(
            "skipped",
            "Для chi2 требуется хотя бы одно непустое значение в каждой выборке",
            details=base_details,
        )

    categories = _ordered_union(reference, current)
    combined_counts = reference_counts + current_counts
    rare_categories = {
        category
        for category in categories
        if combined_counts[category] < rare_min_total
    }
    effective_categories = [
        category for category in categories if category not in rare_categories
    ]
    if rare_categories:
        effective_categories.append(_POOLED_KEY)

    labels = [_category_label(category) for category in effective_categories]
    observed = np.asarray(
        [
            _pool_counts(reference_counts, categories, rare_categories),
            _pool_counts(current_counts, categories, rare_categories),
        ],
        dtype=np.int64,
    )
    details = {
        **base_details,
        "categories": labels,
        "pooled_categories": [
            _category_label(category)
            for category in categories
            if category in rare_categories
        ],
        "observed_counts": {
            "reference": observed[0].tolist(),
            "current": observed[1].tolist(),
        },
    }

    if observed.shape[1] < 2:
        return _result(
            "skipped",
            "После объединения редких значений осталось меньше двух категорий",
            details=details,
        )

    result = chi2_contingency(observed, correction=False)
    expected = np.asarray(result.expected_freq, dtype=np.float64)
    minimum_observed = int(observed.min())
    minimum_expected = float(expected.min())
    details.update(
        {
            "expected_counts": {
                "reference": expected[0].tolist(),
                "current": expected[1].tolist(),
            },
            "degrees_of_freedom": int(result.dof),
            "minimum_observed_count": minimum_observed,
            "minimum_expected_count": minimum_expected,
        }
    )

    problems: list[str] = []
    if minimum_observed < min_observed:
        problems.append(
            f"минимальная наблюдаемая частота {minimum_observed} < {min_observed}"
        )
    if minimum_expected < min_expected:
        problems.append(
            f"минимальная ожидаемая частота {minimum_expected:.6g} < {min_expected:g}"
        )
    if problems:
        return _result(
            "skipped",
            "Асимптотический chi2 неприменим: " + "; ".join(problems),
            details=details,
        )

    statistic = float(result.statistic)
    p_value = float(result.pvalue)
    if not np.isfinite(statistic) or not np.isfinite(p_value):
        return _result(
            "error",
            "SciPy вернул нечисловой или бесконечный результат chi2",
            details=details,
        )
    return _result(
        "ok",
        None,
        value=statistic,
        p_value=p_value,
        details=details,
    )
