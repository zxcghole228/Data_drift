"""Multiple-testing decisions and explainable drift alerts."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from math import isclose, isfinite
from numbers import Real
from typing import Any, cast

from .contracts import Alert, CheckResult, FeatureResult, Status

_P_VALUE_METHODS = frozenset({"ks", "chi2"})
_FAMILY_DESCRIPTION = "all successful KS and chi2 checks in one analyze call"
_STATUS_PRIORITY: tuple[Status, ...] = (
    "error",
    "critical",
    "warning",
    "ok",
    "skipped",
)


def _settings(config: Mapping[str, Any]) -> tuple[str, float]:
    if not isinstance(config, Mapping):
        raise TypeError("config должен быть отображением ключ-значение")
    method = config.get("multiple_testing")
    alpha = config.get("alpha")
    if method not in {"none", "bh"}:
        raise ValueError("multiple_testing должен быть равен 'none' или 'bh'")
    if isinstance(alpha, bool) or not isinstance(alpha, Real):
        raise ValueError("alpha должен быть числом из интервала (0, 1)")
    normalized_alpha = float(alpha)
    if not isfinite(normalized_alpha) or not 0.0 < normalized_alpha < 1.0:
        raise ValueError("alpha должен быть числом из интервала (0, 1)")
    return cast(str, method), normalized_alpha


def _highest_status(statuses: list[Status]) -> Status:
    if not statuses:
        return "skipped"
    for status in _STATUS_PRIORITY:
        if status in statuses:
            return status
    return "skipped"


def _refresh_feature(feature: FeatureResult) -> None:
    checks = feature["checks"]
    feature["status"] = _highest_status([check["status"] for check in checks.values()])
    if any(check["alert"] is True for check in checks.values()):
        feature["alert"] = True
    elif any(check["alert"] is False for check in checks.values()):
        feature["alert"] = False
    else:
        feature["alert"] = None

    failed = [name for name, check in checks.items() if check["status"] == "error"]
    skipped = [name for name, check in checks.items() if check["status"] == "skipped"]
    if failed:
        feature["reason"] = "Ошибки drift-проверок: " + ", ".join(failed)
    elif checks and len(skipped) == len(checks):
        feature["reason"] = "Все drift-проверки признака пропущены: " + ", ".join(
            skipped
        )
    elif skipped:
        feature["reason"] = "Часть drift-проверок пропущена: " + ", ".join(skipped)
    elif not checks:
        feature["reason"] = "Для признака не настроены drift-методы"
    else:
        feature["reason"] = None


def _bh_adjusted(p_values: list[float]) -> list[float]:
    """Return monotone Benjamini-Hochberg adjusted p-values."""

    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [1.0] * len(p_values)
    running_minimum = 1.0
    family_size = len(p_values)
    for zero_based_rank in range(family_size - 1, -1, -1):
        index = order[zero_based_rank]
        rank = zero_based_rank + 1
        candidate = min(1.0, p_values[index] * family_size / rank)
        running_minimum = min(running_minimum, candidate)
        adjusted[index] = running_minimum
    return adjusted


def apply_multiple_testing(
    features: Mapping[str, FeatureResult],
    config: Mapping[str, Any],
) -> dict[str, FeatureResult]:
    """Apply the configured p-value decision rule without mutating input.

    The BH family is defined once per ``analyze`` call and contains every
    successful KS and chi-square p-value. Skipped or failed checks do not enter
    the family. Raw p-values are kept unchanged.
    """

    if not isinstance(features, Mapping):
        raise TypeError("features должен быть отображением результатов признаков")
    method, alpha = _settings(config)
    copied = cast(dict[str, FeatureResult], deepcopy(dict(features)))
    if method == "none":
        return copied

    candidates: list[tuple[str, str, CheckResult]] = []
    p_values: list[float] = []
    for feature_name, feature in copied.items():
        for check_name, check in feature["checks"].items():
            p_value = check["p_value"]
            if (
                check_name in _P_VALUE_METHODS
                and check["status"] == "ok"
                and p_value is not None
            ):
                normalized = float(p_value)
                if not isfinite(normalized) or not 0.0 <= normalized <= 1.0:
                    check["status"] = "error"
                    check["alert"] = None
                    check["reason"] = "p-value вне допустимого диапазона [0, 1]"
                    continue
                candidates.append((feature_name, check_name, check))
                p_values.append(normalized)

    adjusted = _bh_adjusted(p_values)
    family_size = len(candidates)
    ranked_indices = sorted(range(family_size), key=p_values.__getitem__)
    ranks = {index: rank for rank, index in enumerate(ranked_indices, start=1)}
    for index, ((feature_name, check_name, check), adjusted_p_value) in enumerate(
        zip(candidates, adjusted, strict=True)
    ):
        alert = adjusted_p_value < alpha and not isclose(
            adjusted_p_value,
            alpha,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        check["adjusted_p_value"] = float(adjusted_p_value)
        check["threshold"] = alpha
        check["alert"] = alert
        check["status"] = "warning" if alert else "ok"
        check["reason"] = (
            f"Скорректированный p-value метода {check_name} меньше alpha={alpha}"
            if alert
            else None
        )
        check["details"].update(
            {
                "decision": "benjamini_hochberg_adjusted_p_value",
                "decision_quantity": "adjusted_p_value",
                "multiple_testing": "bh",
                "multiple_testing_family": _FAMILY_DESCRIPTION,
                "multiple_testing_family_size": family_size,
                "multiple_testing_rank": ranks[index],
                "multiple_testing_feature": feature_name,
            }
        )

    for feature in copied.values():
        _refresh_feature(feature)
    return copied


def _number(value: float | None) -> str:
    return "—" if value is None else f"{float(value):.6g}"


def build_drift_alerts(
    features: Mapping[str, FeatureResult],
    config: Mapping[str, Any],
) -> list[Alert]:
    """Build one alert per triggered drift rule with its decision evidence."""

    if not isinstance(features, Mapping):
        raise TypeError("features должен быть отображением результатов признаков")
    _settings(config)
    alerts: list[Alert] = []
    for feature_name, feature in features.items():
        for check_name, check in feature["checks"].items():
            if check["alert"] is not True:
                continue
            decision_quantity = check["details"].get("decision_quantity")
            if decision_quantity == "adjusted_p_value":
                evidence = (
                    f"adjusted_p_value={_number(check['adjusted_p_value'])}, "
                    f"raw_p_value={_number(check['p_value'])}, "
                    f"alpha={_number(check['threshold'])}"
                )
            elif decision_quantity == "p_value":
                evidence = (
                    f"p_value={_number(check['p_value'])}, "
                    f"alpha={_number(check['threshold'])}"
                )
            else:
                evidence = (
                    f"value={_number(check['value'])}, "
                    f"threshold={_number(check['threshold'])}"
                )
            explanation = check["reason"] or "Сработало настроенное правило"
            alerts.append(
                {
                    "source": "drift",
                    "feature": feature_name,
                    "check": check_name,
                    "severity": (
                        "critical" if check["status"] == "critical" else "warning"
                    ),
                    "message": f"{check_name}: {evidence}. {explanation}",
                }
            )
    return alerts
