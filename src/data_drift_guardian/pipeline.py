"""Оркестрация проверки схемы, качества данных и доступных drift-метрик."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from math import isclose
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from .config import CONTRACT_VERSION, validate_config
from .contracts import (
    AdversarialResult,
    Alert,
    AnalysisConfig,
    AnalysisResult,
    CheckResult,
    DriftConfig,
    DriftResult,
    FeatureConfig,
    FeatureResult,
    FeatureType,
    QualityResult,
    SchemaResult,
    Status,
)
from .drift import build_probabilities, js_divergence, ks_test, psi, wasserstein
from .quality import check_quality
from .schema import validate_schema

_STATUS_PRIORITY: tuple[Status, ...] = (
    "error",
    "critical",
    "warning",
    "ok",
    "skipped",
)


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


def _infer_feature_type(series: pd.Series) -> FeatureType:
    """Вывести семантический тип без изменения или преобразования Series."""

    if is_numeric_dtype(series.dtype) and not is_bool_dtype(series.dtype):
        return "numeric"
    return "categorical"


def _default_config(reference: pd.DataFrame) -> AnalysisConfig:
    """Построить документированную конфигурацию для ``config=None``.

    Имена и семантические типы признаков выводятся только из Reference.
    Диапазоны значений и пороги drift-метрик из данных не выводятся.
    """

    features: dict[str, FeatureConfig] = {}
    for position, raw_name in enumerate(reference.columns):
        name = str(raw_name)
        if name in features:
            continue
        features[name] = {
            "kind": _infer_feature_type(reference.iloc[:, position]),
            "nullable": True,
        }

    raw_config: dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "random_seed": 42,
        "schema": {
            "allow_extra_columns": False,
            "features": features,
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
    return validate_config(raw_config)


def _check_result(
    name: str,
    status: Status,
    reason: str,
    *,
    details: dict[str, Any] | None = None,
) -> CheckResult:
    return {
        "name": name,
        "status": status,
        "value": None,
        "threshold": None,
        "p_value": None,
        "adjusted_p_value": None,
        "alert": None,
        "reason": reason,
        "details": {} if details is None else details,
    }


def _highest_status(statuses: list[Status]) -> Status:
    if not statuses:
        return "skipped"
    for status in _STATUS_PRIORITY:
        if status in statuses:
            return status
    return "skipped"


def _exceeds(value: float, threshold: float) -> bool:
    return value > threshold and not isclose(
        value,
        threshold,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


def _apply_distance_threshold(
    result: CheckResult,
    threshold: float | None,
) -> CheckResult:
    """Применить верхний порог к успешно рассчитанной distance-метрике."""

    result["details"]["decision_quantity"] = "value"
    if result["status"] != "ok" or result["value"] is None:
        return result
    if threshold is None:
        result["details"]["decision"] = "threshold_not_configured"
        return result

    value = float(result["value"])
    alert = _exceeds(value, threshold)
    result["threshold"] = threshold
    result["alert"] = alert
    result["status"] = "warning" if alert else "ok"
    result["reason"] = (
        f"Метрика {result['name']} превышает порог {threshold}" if alert else None
    )
    result["details"]["decision"] = "upper_threshold"
    return result


def _apply_p_value_rule(
    result: CheckResult,
    drift_config: DriftConfig,
) -> CheckResult:
    """Применить alpha без поправки или явно отложить решение для BH."""

    result["details"]["decision_quantity"] = "p_value"
    result["details"]["multiple_testing"] = drift_config["multiple_testing"]
    if result["status"] != "ok" or result["p_value"] is None:
        return result

    alpha = drift_config["alpha"]
    result["threshold"] = alpha
    if drift_config["multiple_testing"] == "bh":
        result["details"]["decision"] = "deferred_until_bh_correction"
        return result

    alert = result["p_value"] < alpha and not isclose(
        result["p_value"],
        alpha,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )
    result["alert"] = alert
    result["status"] = "warning" if alert else "ok"
    result["reason"] = (
        f"p-value метода {result['name']} меньше alpha={alpha}" if alert else None
    )
    result["details"]["decision"] = "raw_p_value"
    return result


def _valid_count(series: pd.Series, feature_type: FeatureType) -> int:
    if feature_type == "categorical":
        return int(series.notna().sum())
    values = series.to_numpy(dtype=np.float64, na_value=np.nan)
    return int(np.isfinite(values).sum())


def _stability_results(
    reference: pd.Series,
    current: pd.Series,
    *,
    feature_config: FeatureConfig,
    drift_config: DriftConfig,
    methods: list[str],
    n_reference_valid: int,
    n_current_valid: int,
) -> dict[str, CheckResult]:
    requested = [method for method in methods if method in {"psi", "js"}]
    if not requested:
        return {}

    if n_reference_valid == 0 or n_current_valid == 0:
        reason = (
            "Для построения распределений требуется хотя бы одно пригодное "
            "значение в каждой выборке"
        )
        return {
            method: _check_result(
                method,
                "skipped",
                reason,
                details={
                    "n_reference_valid": n_reference_valid,
                    "n_current_valid": n_current_valid,
                },
            )
            for method in requested
        }

    binning_config: dict[str, Any] = {"kind": feature_config["kind"]}
    if feature_config["kind"] == "numeric":
        binning_config["n_bins"] = drift_config["n_bins"]

    try:
        reference_probabilities, current_probabilities, binning_details = (
            build_probabilities(reference, current, binning_config)
        )
    except (TypeError, ValueError, OverflowError) as exc:
        reason = (
            f"Не удалось построить общее распределение: {type(exc).__name__}: {exc}"
        )
        return {method: _check_result(method, "error", reason) for method in requested}

    results: dict[str, CheckResult] = {}
    for method in requested:
        try:
            if method == "psi":
                result = psi(
                    reference_probabilities,
                    current_probabilities,
                    smoothing=drift_config["psi_smoothing"],
                )
            else:
                result = js_divergence(
                    reference_probabilities,
                    current_probabilities,
                    base=drift_config["js_base"],
                )
        except (TypeError, ValueError, FloatingPointError, OverflowError) as exc:
            result = _check_result(
                method,
                "error",
                f"Не удалось вычислить {method}: {type(exc).__name__}: {exc}",
            )

        result["details"]["binning"] = deepcopy(binning_details)
        threshold = drift_config["distance_thresholds"][method]
        results[method] = _apply_distance_threshold(result, threshold)
    return results


def _summarize_feature(
    checks: dict[str, CheckResult],
) -> tuple[Status, bool | None, str | None]:
    statuses = [check["status"] for check in checks.values()]
    status = _highest_status(statuses)

    if any(check["alert"] is True for check in checks.values()):
        alert: bool | None = True
    elif any(check["alert"] is False for check in checks.values()):
        alert = False
    else:
        alert = None

    failed = [name for name, check in checks.items() if check["status"] == "error"]
    skipped = [name for name, check in checks.items() if check["status"] == "skipped"]
    if failed:
        reason = "Ошибки drift-проверок: " + ", ".join(failed)
    elif checks and len(skipped) == len(checks):
        reason = "Все drift-проверки признака пропущены: " + ", ".join(skipped)
    elif skipped:
        reason = "Часть drift-проверок пропущена: " + ", ".join(skipped)
    elif not checks:
        reason = "Для признака не настроены drift-методы"
    else:
        reason = None
    return status, alert, reason


def _analyze_feature(
    reference: pd.Series,
    current: pd.Series,
    *,
    feature_config: FeatureConfig,
    drift_config: DriftConfig,
) -> FeatureResult:
    feature_type = feature_config["kind"]
    methods: list[str] = list(
        drift_config[
            "numeric_methods" if feature_type == "numeric" else "categorical_methods"
        ]
    )
    n_reference_valid = _valid_count(reference, feature_type)
    n_current_valid = _valid_count(current, feature_type)
    stability = _stability_results(
        reference,
        current,
        feature_config=feature_config,
        drift_config=drift_config,
        methods=methods,
        n_reference_valid=n_reference_valid,
        n_current_valid=n_current_valid,
    )

    checks: dict[str, CheckResult] = {}
    for method in methods:
        if method == "ks":
            checks[method] = _apply_p_value_rule(
                ks_test(reference, current),
                drift_config,
            )
        elif method == "wasserstein":
            checks[method] = _apply_distance_threshold(
                wasserstein(reference, current),
                drift_config["distance_thresholds"]["wasserstein"],
            )
        elif method in stability:
            checks[method] = stability[method]
        elif method == "chi2":
            checks[method] = _check_result(
                "chi2",
                "skipped",
                "Категориальный χ² ещё не реализован",
            )

    status, alert, reason = _summarize_feature(checks)
    return {
        "feature_type": feature_type,
        "status": status,
        "n_reference_valid": n_reference_valid,
        "n_current_valid": n_current_valid,
        "checks": checks,
        "alert": alert,
        "reason": reason,
    }


def _schema_skip_reason(feature: str, schema: SchemaResult) -> str:
    problems: list[str] = []
    for dataset in ("reference", "current"):
        if feature in schema["missing_columns"][dataset]:
            problems.append(f"колонка отсутствует в {dataset}")
        if feature in schema["duplicate_columns"][dataset]:
            problems.append(f"имя колонки повторяется в {dataset}")
    for mismatch in schema["type_mismatches"]:
        if mismatch["column"] == feature:
            problems.append(
                f"тип {mismatch['actual']} в {mismatch['dataset']} несовместим "
                f"с {mismatch['expected']}"
            )
    if not problems:
        problems.append("признак недоступен после проверки схемы")
    return f"Признак {feature!r} пропущен: " + "; ".join(problems)


def _drift_result(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    config: AnalysisConfig,
    schema: SchemaResult,
) -> DriftResult:
    valid_features = set(schema["valid_features"])
    features: dict[str, FeatureResult] = {}

    for feature, feature_config in config["schema"]["features"].items():
        if feature not in valid_features:
            features[feature] = {
                "feature_type": feature_config["kind"],
                "status": "skipped",
                "n_reference_valid": 0,
                "n_current_valid": 0,
                "checks": {},
                "alert": None,
                "reason": _schema_skip_reason(feature, schema),
            }
            continue
        features[feature] = _analyze_feature(
            reference[feature],
            current[feature],
            feature_config=feature_config,
            drift_config=config["drift"],
        )

    status = _highest_status([feature["status"] for feature in features.values()])
    failed_count = sum(feature["status"] == "error" for feature in features.values())
    skipped_count = sum(feature["status"] == "skipped" for feature in features.values())
    if failed_count:
        reason = f"Ошибки drift-анализа для признаков: {failed_count}"
    elif features and skipped_count == len(features):
        reason = "Все признаки пропущены в drift-анализе"
    elif skipped_count:
        reason = f"Полностью пропущено признаков: {skipped_count}"
    else:
        partial = sum(
            any(check["status"] == "skipped" for check in feature["checks"].values())
            for feature in features.values()
        )
        reason = (
            f"Есть частично пропущенные методы у признаков: {partial}"
            if partial
            else None
        )

    return {"status": status, "features": features, "reason": reason}


def _adversarial_result(config: AnalysisConfig) -> AdversarialResult:
    enabled = config["adversarial"]["enabled"]
    return {
        "status": "skipped",
        "roc_auc": None,
        "fold_auc": [],
        "feature_importance": {},
        "importance_type": None,
        "alert": None,
        "reason": (
            "Adversarial Validation ещё не реализован"
            if enabled
            else "Adversarial Validation отключён в конфигурации"
        ),
    }


def _schema_alerts(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    schema: SchemaResult,
    config: AnalysisConfig,
) -> list[Alert]:
    alerts: list[Alert] = []
    for dataset, frame in (("reference", reference), ("current", current)):
        if frame.empty:
            alerts.append(
                {
                    "source": "schema",
                    "feature": None,
                    "check": "empty_dataset",
                    "severity": "critical",
                    "message": f"Таблица {dataset} пуста",
                }
            )
        for feature in schema["missing_columns"][dataset]:
            alerts.append(
                {
                    "source": "schema",
                    "feature": feature,
                    "check": "missing_column",
                    "severity": "critical",
                    "message": f"Колонка {feature!r} отсутствует в {dataset}",
                }
            )
        if not config["schema"]["allow_extra_columns"]:
            for feature in schema["extra_columns"][dataset]:
                alerts.append(
                    {
                        "source": "schema",
                        "feature": feature,
                        "check": "extra_column",
                        "severity": "critical",
                        "message": f"Лишняя колонка {feature!r} обнаружена в {dataset}",
                    }
                )
        for feature in schema["duplicate_columns"][dataset]:
            alerts.append(
                {
                    "source": "schema",
                    "feature": feature,
                    "check": "duplicate_column",
                    "severity": "critical",
                    "message": f"Имя колонки {feature!r} повторяется в {dataset}",
                }
            )

    for mismatch in schema["type_mismatches"]:
        alerts.append(
            {
                "source": "schema",
                "feature": mismatch["column"],
                "check": "type_mismatch",
                "severity": "critical",
                "message": (
                    f"Тип {mismatch['actual']} признака {mismatch['column']!r} "
                    f"в {mismatch['dataset']} несовместим с {mismatch['expected']}"
                ),
            }
        )
    return alerts


def _alert_from_check(
    check: CheckResult,
    *,
    source: str,
    feature: str | None,
) -> Alert:
    return {
        "source": source,  # type: ignore[typeddict-item]
        "feature": feature,
        "check": check["name"],
        "severity": "critical" if check["status"] == "critical" else "warning",
        "message": check["reason"] or f"Сработало правило {check['name']}",
    }


def _quality_alerts(quality: QualityResult) -> list[Alert]:
    alerts = [
        _alert_from_check(check, source="quality", feature=None)
        for check in quality["dataset_checks"]
        if check["alert"] is True
    ]
    for feature, checks in quality["feature_checks"].items():
        alerts.extend(
            _alert_from_check(check, source="quality", feature=feature)
            for check in checks
            if check["alert"] is True
        )
    return alerts


def _drift_alerts(drift: DriftResult) -> list[Alert]:
    alerts: list[Alert] = []
    for feature, feature_result in drift["features"].items():
        alerts.extend(
            _alert_from_check(check, source="drift", feature=feature)
            for check in feature_result["checks"].values()
            if check["alert"] is True
        )
    return alerts


def _is_analyzed(feature: FeatureResult) -> bool:
    return any(
        check["status"] in {"ok", "warning", "critical"}
        for check in feature["checks"].values()
    )


def analyze(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    config: Mapping[str, Any] | None = None,
) -> AnalysisResult:
    """Сравнить два DataFrame и вернуть единый JSON-безопасный результат.

    Некорректная конфигурация вызывает ``ValueError``, неверные типы входных
    объектов — ``TypeError``. Ошибки содержимого и неприменимые вычисления
    сохраняются в отдельных разделах результата и не отменяют анализ других
    совместимых признаков.
    """

    _require_dataframes(reference, current)
    if config is None:
        if len(reference.columns) == 0:
            raise ValueError(
                "Невозможно вывести конфигурацию: Reference не содержит колонок"
            )
        validated_config = _default_config(reference)
    else:
        validated_config = validate_config(config)

    schema = validate_schema(
        reference,
        current,
        validated_config["schema"],
    )
    quality = check_quality(reference, current, validated_config)
    drift = _drift_result(reference, current, validated_config, schema)
    adversarial = _adversarial_result(validated_config)

    alerts = [
        *_schema_alerts(reference, current, schema, validated_config),
        *_quality_alerts(quality),
        *_drift_alerts(drift),
    ]
    feature_results = list(drift["features"].values())
    result: AnalysisResult = {
        "contract_version": validated_config["contract_version"],
        "metadata": {
            "reference_rows": len(reference),
            "current_rows": len(current),
            "random_seed": validated_config["random_seed"],
        },
        "schema": schema,
        "quality": quality,
        "drift": drift,
        "adversarial": adversarial,
        "alerts": alerts,
        "summary": {
            "status": _highest_status(
                [
                    schema["status"],
                    quality["status"],
                    drift["status"],
                    adversarial["status"],
                ]
            ),
            "has_alerts": bool(alerts),
            "n_alerts": len(alerts),
            "analyzed_features": sum(
                _is_analyzed(feature) for feature in feature_results
            ),
            "skipped_features": sum(
                feature["status"] == "skipped" for feature in feature_results
            ),
        },
    }

    try:
        json.dumps(result, allow_nan=False)
    except (TypeError, ValueError) as exc:  # pragma: no cover - invariant guard
        raise RuntimeError(
            "Внутренняя ошибка: AnalysisResult не является JSON-безопасным"
        ) from exc
    return result
