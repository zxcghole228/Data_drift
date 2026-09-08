"""Проверки пропусков, дубликатов, диапазонов и бесконечностей."""

from __future__ import annotations

from collections.abc import Mapping
from math import isclose
from typing import Any, Literal

import numpy as np
import pandas as pd

from .config import validate_config
from .contracts import (
    AnalysisConfig,
    CheckResult,
    FeatureConfig,
    QualityResult,
    Status,
)
from .schema import validate_schema

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


def _exceeds(value: float, threshold: float) -> bool:
    """Сравнить с верхним порогом без ложного срабатывания на его границе."""

    return value > threshold and not isclose(
        value,
        threshold,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


def _result(
    *,
    name: str,
    status: Status,
    value: float | None,
    threshold: float | None,
    alert: bool | None,
    reason: str | None = None,
    details: dict[str, Any] | None = None,
) -> CheckResult:
    return {
        "name": name,
        "status": status,
        "value": value,
        "threshold": threshold,
        "p_value": None,
        "adjusted_p_value": None,
        "alert": alert,
        "reason": reason,
        "details": details or {},
    }


def _duplicate_check(
    frame: pd.DataFrame,
    dataset: DatasetName,
    threshold: float,
) -> CheckResult:
    row_count = len(frame)
    try:
        duplicate_count = int(frame.duplicated(keep="first").sum())
    except (TypeError, ValueError) as exc:
        return _result(
            name=f"{dataset}_duplicate_fraction",
            status="error",
            value=None,
            threshold=threshold,
            alert=None,
            reason=(
                f"Не удалось проверить дубликаты в {dataset}: "
                f"{type(exc).__name__}: {exc}"
            ),
            details={
                "dataset": dataset,
                "row_count": row_count,
                "definition": "повторные полные строки после первого вхождения",
            },
        )

    fraction = duplicate_count / row_count
    alert = _exceeds(fraction, threshold)
    return _result(
        name=f"{dataset}_duplicate_fraction",
        status="warning" if alert else "ok",
        value=float(fraction),
        threshold=threshold,
        alert=alert,
        reason=(
            f"Доля дубликатов в {dataset} превышает допустимый порог" if alert else None
        ),
        details={
            "dataset": dataset,
            "row_count": row_count,
            "duplicate_count": duplicate_count,
            "definition": "повторные полные строки после первого вхождения",
        },
    )


def _missing_fraction_check(
    series: pd.Series,
    *,
    dataset: DatasetName,
    feature: str,
    feature_config: FeatureConfig,
    configured_threshold: float,
) -> tuple[CheckResult, int, float]:
    row_count = len(series)
    missing_count = int(series.isna().sum())
    fraction = missing_count / row_count
    nullable = feature_config["nullable"]
    threshold = configured_threshold if nullable else 0.0
    alert = _exceeds(fraction, threshold)

    if alert and nullable:
        reason = f"Доля пропусков {feature!r} в {dataset} превышает допустимый порог"
        status: Status = "warning"
    elif alert:
        reason = f"Признак {feature!r} в {dataset} не допускает пропуски"
        status = "critical"
    else:
        reason = None
        status = "ok"

    check = _result(
        name=f"{dataset}_missing_fraction",
        status=status,
        value=float(fraction),
        threshold=threshold,
        alert=alert,
        reason=reason,
        details={
            "dataset": dataset,
            "feature": feature,
            "dtype": str(series.dtype),
            "row_count": row_count,
            "missing_count": missing_count,
            "nullable": nullable,
            "configured_max_missing_fraction": configured_threshold,
        },
    )
    return check, missing_count, float(fraction)


def _missing_increase_check(
    *,
    feature: str,
    reference_count: int,
    current_count: int,
    reference_rows: int,
    current_rows: int,
    reference_fraction: float,
    current_fraction: float,
    threshold: float,
) -> CheckResult:
    increase_pp = round(
        (current_fraction - reference_fraction) * 100.0,
        12,
    )
    alert = _exceeds(increase_pp, threshold)
    return _result(
        name="missing_increase_pp",
        status="warning" if alert else "ok",
        value=float(increase_pp),
        threshold=threshold,
        alert=alert,
        reason=(
            f"Рост доли пропусков {feature!r} превышает допустимый порог"
            if alert
            else None
        ),
        details={
            "feature": feature,
            "reference_rows": reference_rows,
            "current_rows": current_rows,
            "reference_missing_count": reference_count,
            "current_missing_count": current_count,
            "reference_missing_fraction": reference_fraction,
            "current_missing_fraction": current_fraction,
        },
    )


def _numeric_values(series: pd.Series) -> np.ndarray:
    return series.to_numpy(dtype=np.float64, na_value=np.nan)


def _infinity_check(
    values: np.ndarray,
    *,
    dataset: DatasetName,
    feature: str,
    dtype: str,
    row_count: int,
) -> CheckResult:
    positive_count = int(np.isposinf(values).sum())
    negative_count = int(np.isneginf(values).sum())
    infinite_count = positive_count + negative_count
    alert = infinite_count > 0
    return _result(
        name=f"{dataset}_infinite_count",
        status="critical" if alert else "ok",
        value=infinite_count,
        threshold=0,
        alert=alert,
        reason=(
            f"В признаке {feature!r} в {dataset} обнаружены бесконечные значения"
            if alert
            else None
        ),
        details={
            "dataset": dataset,
            "feature": feature,
            "dtype": dtype,
            "row_count": row_count,
            "positive_infinity_count": positive_count,
            "negative_infinity_count": negative_count,
            "infinite_fraction": float(infinite_count / row_count),
        },
    )


def _range_check(
    values: np.ndarray,
    *,
    dataset: DatasetName,
    feature: str,
    dtype: str,
    boundary_name: Literal["min", "max"],
    boundary: float,
) -> CheckResult:
    finite = values[np.isfinite(values)]
    check_name = f"{dataset}_{boundary_name}"
    if finite.size == 0:
        return _result(
            name=check_name,
            status="skipped",
            value=None,
            threshold=boundary,
            alert=None,
            reason=f"В {dataset} нет конечных значений для проверки {boundary_name}",
            details={
                "dataset": dataset,
                "feature": feature,
                "dtype": dtype,
                "finite_count": 0,
                "violation_count": None,
            },
        )

    if boundary_name == "min":
        observed = float(finite.min())
        violation_count = int((finite < boundary).sum())
    else:
        observed = float(finite.max())
        violation_count = int((finite > boundary).sum())

    alert = violation_count > 0
    return _result(
        name=check_name,
        status="critical" if alert else "ok",
        value=observed,
        threshold=boundary,
        alert=alert,
        reason=(
            f"В признаке {feature!r} в {dataset} обнаружены значения "
            f"за границей {boundary_name}={boundary}"
            if alert
            else None
        ),
        details={
            "dataset": dataset,
            "feature": feature,
            "dtype": dtype,
            "finite_count": int(finite.size),
            "violation_count": violation_count,
        },
    )


def _feature_checks(
    reference: pd.Series,
    current: pd.Series,
    *,
    feature: str,
    feature_config: FeatureConfig,
    config: AnalysisConfig,
) -> list[CheckResult]:
    missing_threshold = config["quality"]["max_missing_fraction"]
    reference_missing, reference_count, reference_fraction = _missing_fraction_check(
        reference,
        dataset="reference",
        feature=feature,
        feature_config=feature_config,
        configured_threshold=missing_threshold,
    )
    current_missing, current_count, current_fraction = _missing_fraction_check(
        current,
        dataset="current",
        feature=feature,
        feature_config=feature_config,
        configured_threshold=missing_threshold,
    )

    checks = [
        reference_missing,
        current_missing,
        _missing_increase_check(
            feature=feature,
            reference_count=reference_count,
            current_count=current_count,
            reference_rows=len(reference),
            current_rows=len(current),
            reference_fraction=reference_fraction,
            current_fraction=current_fraction,
            threshold=config["quality"]["max_missing_increase_pp"],
        ),
    ]

    if feature_config["kind"] != "numeric":
        return checks

    series_by_dataset = {
        "reference": reference,
        "current": current,
    }
    for dataset, series in series_by_dataset.items():
        values = _numeric_values(series)
        checks.append(
            _infinity_check(
                values,
                dataset=dataset,
                feature=feature,
                dtype=str(series.dtype),
                row_count=len(values),
            )
        )
        if "min" in feature_config:
            checks.append(
                _range_check(
                    values,
                    dataset=dataset,
                    feature=feature,
                    dtype=str(series.dtype),
                    boundary_name="min",
                    boundary=feature_config["min"],
                )
            )
        if "max" in feature_config:
            checks.append(
                _range_check(
                    values,
                    dataset=dataset,
                    feature=feature,
                    dtype=str(series.dtype),
                    boundary_name="max",
                    boundary=feature_config["max"],
                )
            )

    return checks


def _summarize_status(checks: list[CheckResult]) -> tuple[Status, str | None]:
    counts = {
        status: sum(check["status"] == status for check in checks)
        for status in ("error", "critical", "warning", "skipped")
    }
    if counts["error"]:
        return "error", f"Не выполнено проверок качества: {counts['error']}"
    if counts["critical"]:
        return (
            "critical",
            (
                "Выявлены критические нарушения качества: "
                f"{counts['critical']}; предупреждения: {counts['warning']}"
            ),
        )
    if counts["warning"]:
        return "warning", f"Сработало предупреждений качества: {counts['warning']}"
    if checks and counts["skipped"] == len(checks):
        return "skipped", "Все проверки качества были пропущены"
    return "ok", None


def check_quality(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    config: Mapping[str, Any],
) -> QualityResult:
    """Проверить качество двух таблиц без изменения входных данных.

    Проблемы конфигурации вызывают ``ValueError``, неверные типы аргументов —
    ``TypeError``. Проблемы содержимого допустимых DataFrame возвращаются как
    структурированный ``QualityResult``.
    """

    _require_dataframes(reference, current)
    validated_config = validate_config(config)
    schema_result = validate_schema(
        reference,
        current,
        validated_config["schema"],
    )

    empty_datasets = [
        name
        for name, frame in (("reference", reference), ("current", current))
        if frame.empty
    ]
    if empty_datasets:
        return {
            "status": "error",
            "dataset_checks": [],
            "feature_checks": {},
            "reason": f"Пустые таблицы: {', '.join(empty_datasets)}",
        }

    duplicate_threshold = validated_config["quality"]["max_duplicate_fraction"]
    dataset_checks = [
        _duplicate_check(reference, "reference", duplicate_threshold),
        _duplicate_check(current, "current", duplicate_threshold),
    ]

    feature_checks = {
        feature: _feature_checks(
            reference[feature],
            current[feature],
            feature=feature,
            feature_config=validated_config["schema"]["features"][feature],
            config=validated_config,
        )
        for feature in schema_result["valid_features"]
    }

    all_checks = dataset_checks + [
        check
        for checks_for_feature in feature_checks.values()
        for check in checks_for_feature
    ]
    status, reason = _summarize_status(all_checks)
    return {
        "status": status,
        "dataset_checks": dataset_checks,
        "feature_checks": feature_checks,
        "reason": reason,
    }
