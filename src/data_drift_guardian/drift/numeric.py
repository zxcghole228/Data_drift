"""Статистические методы для одномерных числовых признаков."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype
from scipy import stats

from ..contracts import CheckResult, Status

MIN_SAMPLE_SIZE = 2


def _empty_result(name: str, status: Status, reason: str) -> CheckResult:
    """Создать результат для проверки, которую нельзя вычислить."""
    return {
        "name": name,
        "status": status,
        "value": None,
        "threshold": None,
        "p_value": None,
        "adjusted_p_value": None,
        "alert": None,
        "reason": reason,
        "details": {},
    }


def _require_series(reference: object, current: object) -> None:
    invalid: list[str] = []
    if not isinstance(reference, pd.Series):
        invalid.append(f"reference={type(reference).__name__}")
    if not isinstance(current, pd.Series):
        invalid.append(f"current={type(current).__name__}")
    if invalid:
        actual = ", ".join(invalid)
        raise TypeError(
            "reference and current must be pandas.Series objects; "
            f"received {actual}"
        )


def _unsupported_dtype(series: pd.Series) -> bool:
    """Вернуть True для dtype без корректного вещественного порядка."""
    return (
        not is_numeric_dtype(series.dtype)
        or is_bool_dtype(series.dtype)
        or is_complex_dtype(series.dtype)
    )


def _prepare_samples(
    reference: pd.Series,
    current: pd.Series,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | CheckResult:
    unsupported = [
        f"{label}={series.dtype}"
        for label, series in (("reference", reference), ("current", current))
        if _unsupported_dtype(series)
    ]
    if unsupported:
        result = _empty_result(
            name="",
            status="skipped",
            reason=(
                "Real numeric pandas Series are required; unsupported dtype(s): "
                + ", ".join(unsupported)
            ),
        )
        details: dict[str, Any] = {
            "n_reference_original": int(len(reference)),
            "n_current_original": int(len(current)),
            "reference_dtype": str(reference.dtype),
            "current_dtype": str(current.dtype),
        }
        for label, series in (("reference", reference), ("current", current)):
            details[f"n_{label}_missing"] = int(series.isna().sum())
            if _unsupported_dtype(series):
                details[f"n_{label}_valid"] = 0
                details[f"n_{label}_infinite"] = None
                continue

            values = series.to_numpy(dtype=np.float64, na_value=np.nan)
            details[f"n_{label}_valid"] = int(np.isfinite(values).sum())
            details[f"n_{label}_infinite"] = int(np.isinf(values).sum())
        result["details"] = details
        return result

    reference_values = reference.to_numpy(dtype=np.float64, na_value=np.nan)
    current_values = current.to_numpy(dtype=np.float64, na_value=np.nan)

    reference_missing = np.isnan(reference_values)
    current_missing = np.isnan(current_values)
    reference_infinite = np.isinf(reference_values)
    current_infinite = np.isinf(current_values)

    clean_reference = reference_values[
        ~(reference_missing | reference_infinite)
    ]
    clean_current = current_values[~(current_missing | current_infinite)]

    details: dict[str, Any] = {
        "n_reference_original": int(reference_values.size),
        "n_current_original": int(current_values.size),
        "n_reference_valid": int(clean_reference.size),
        "n_current_valid": int(clean_current.size),
        "n_reference_missing": int(reference_missing.sum()),
        "n_current_missing": int(current_missing.sum()),
        "n_reference_infinite": int(reference_infinite.sum()),
        "n_current_infinite": int(current_infinite.sum()),
        "reference_dtype": str(reference.dtype),
        "current_dtype": str(current.dtype),
    }

    if (
        clean_reference.size < MIN_SAMPLE_SIZE
        or clean_current.size < MIN_SAMPLE_SIZE
    ):
        result = _empty_result(
            name="",
            status="skipped",
            reason=(
                "At least "
                f"{MIN_SAMPLE_SIZE} finite values are required in each sample; "
                f"received reference={clean_reference.size}, "
                f"current={clean_current.size}"
            ),
        )
        result["details"] = details
        return result

    return clean_reference, clean_current, details


def _calculate(
    name: str,
    reference: pd.Series,
    current: pd.Series,
    calculation: Callable[
        [np.ndarray, np.ndarray],
        tuple[float, float | None, dict[str, Any]],
    ],
) -> CheckResult:
    _require_series(reference, current)

    try:
        prepared = _prepare_samples(reference, current)
    except (TypeError, ValueError, OverflowError) as exc:
        return _empty_result(
            name=name,
            status="error",
            reason=f"Failed to prepare numeric samples: {type(exc).__name__}: {exc}",
        )

    if isinstance(prepared, dict):
        prepared["name"] = name
        return prepared

    clean_reference, clean_current, details = prepared
    try:
        value, p_value, method_details = calculation(
            clean_reference,
            clean_current,
        )
    except (TypeError, ValueError, FloatingPointError, OverflowError) as exc:
        result = _empty_result(
            name=name,
            status="error",
            reason=f"Failed to calculate {name}: {type(exc).__name__}: {exc}",
        )
        result["details"] = details
        return result

    if not np.isfinite(value) or (
        p_value is not None and not np.isfinite(p_value)
    ):
        result = _empty_result(
            name=name,
            status="error",
            reason=f"{name} returned a non-finite result",
        )
        result["details"] = details
        return result

    details.update(method_details)
    return {
        "name": name,
        "status": "ok",
        "value": float(value),
        "threshold": None,
        "p_value": None if p_value is None else float(p_value),
        "adjusted_p_value": None,
        "alert": None,
        "reason": None,
        "details": details,
    }


def ks_test(reference: pd.Series, current: pd.Series) -> CheckResult:
    """Вычислить двухвыборочный двусторонний критерий Колмогорова-Смирнова.

    Функция вычисляет статистику и исходный p-value. Порог ``alpha``, поправка
    на множественные проверки и итоговый alert применяются уровнем pipeline.
    """

    def calculate(
        clean_reference: np.ndarray,
        clean_current: np.ndarray,
    ) -> tuple[float, float, dict[str, Any]]:
        result = stats.ks_2samp(
            clean_reference,
            clean_current,
            alternative="two-sided",
            method="auto",
        )
        return (
            float(result.statistic),
            float(result.pvalue),
            {
                "alternative": "two-sided",
                "method": "auto",
                "statistic_location": float(result.statistic_location),
                "statistic_sign": int(result.statistic_sign),
            },
        )

    return _calculate("ks", reference, current, calculate)


def wasserstein(reference: pd.Series, current: pd.Series) -> CheckResult:
    """Вычислить одномерное расстояние Вассерштейна первого порядка.

    Расстояние выражено в тех же единицах, что и значения признака. Без
    настроенного порога функция не делает вывода об alert.
    """

    def calculate(
        clean_reference: np.ndarray,
        clean_current: np.ndarray,
    ) -> tuple[float, None, dict[str, str]]:
        distance = stats.wasserstein_distance(clean_reference, clean_current)
        return float(distance), None, {"unit": "same_as_feature"}

    return _calculate("wasserstein", reference, current, calculate)
