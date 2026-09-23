"""Чтение и строгая валидация конфигурации online-мониторинга."""

from __future__ import annotations

import re
from collections.abc import Mapping
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import Any, cast

import yaml

from ..config import SUPPORTED_CONFIG_EXTENSIONS, validate_config
from .contracts import (
    MonitoringConfig,
    OnlineAlertConfig,
    OnlineConfig,
    OnlinePerformanceConfig,
    OnlineWindowConfig,
    WindowStrategy,
)

_ANALYSIS_ROOT_KEYS = frozenset(
    {"contract_version", "random_seed", "schema", "quality", "drift", "adversarial"}
)
_ROOT_KEYS = _ANALYSIS_ROOT_KEYS | {"online"}
_ONLINE_KEYS = frozenset(
    {
        "enabled",
        "state_path",
        "max_request_bytes",
        "window",
        "alerts",
        "performance",
    }
)
_WINDOW_KEYS = frozenset({"strategy", "size", "min_batch_rows"})
_ALERT_KEYS = frozenset(
    {"jsonl_path", "webhook_url_env", "timeout_seconds", "max_attempts"}
)
_PERFORMANCE_KEYS = frozenset(
    {
        "enabled",
        "min_feedback_rows",
        "baseline_accuracy",
        "max_accuracy_drop",
        "baseline_roc_auc",
        "max_roc_auc_drop",
    }
)
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _as_mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} должен быть отображением ключ-значение")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"Все ключи {context} должны быть строками")
    return cast(Mapping[str, Any], value)


def _check_exact_keys(
    value: Mapping[str, Any],
    expected: frozenset[str],
    context: str,
) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    problems: list[str] = []
    if missing:
        problems.append(f"отсутствуют обязательные ключи: {', '.join(missing)}")
    if unknown:
        problems.append(f"неизвестные ключи: {', '.join(unknown)}")
    if problems:
        raise ValueError(f"Некорректная секция {context}: {'; '.join(problems)}")


def _as_bool(value: object, context: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{context} должен иметь тип bool")
    return value


def _as_int(value: object, context: str, *, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(
            f"{context} должен быть целым числом не меньше {minimum}"
        )
    return value


def _as_positive_float(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{context} должен быть положительным конечным числом")
    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"{context} должен быть положительным конечным числом")
    return result


def _as_optional_fraction(value: object, context: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{context} должен быть числом в диапазоне [0, 1] или null")
    result = float(value)
    if not isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{context} должен быть числом в диапазоне [0, 1] или null")
    return result


def _as_nonempty_string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} должен быть непустой строкой")
    return value


def _as_optional_string(value: object, context: str) -> str | None:
    if value is None:
        return None
    return _as_nonempty_string(value, context)


def _validate_window(value: object) -> OnlineWindowConfig:
    section = _as_mapping(value, "online.window")
    _check_exact_keys(section, _WINDOW_KEYS, "online.window")

    strategy = section["strategy"]
    if strategy != "tumbling":
        raise ValueError("online.window.strategy должен быть равен 'tumbling'")

    size = _as_int(section["size"], "online.window.size", minimum=2)
    min_batch_rows = _as_int(
        section["min_batch_rows"],
        "online.window.min_batch_rows",
        minimum=2,
    )
    if min_batch_rows > size:
        raise ValueError(
            "online.window.min_batch_rows не должен превышать online.window.size"
        )

    return {
        "strategy": cast(WindowStrategy, strategy),
        "size": size,
        "min_batch_rows": min_batch_rows,
    }


def _validate_alerts(value: object) -> OnlineAlertConfig:
    section = _as_mapping(value, "online.alerts")
    _check_exact_keys(section, _ALERT_KEYS, "online.alerts")

    webhook_url_env = _as_optional_string(
        section["webhook_url_env"], "online.alerts.webhook_url_env"
    )
    if webhook_url_env is not None and _ENV_NAME.fullmatch(webhook_url_env) is None:
        raise ValueError(
            "online.alerts.webhook_url_env должен быть корректным именем "
            "переменной окружения"
        )

    return {
        "jsonl_path": _as_optional_string(
            section["jsonl_path"], "online.alerts.jsonl_path"
        ),
        "webhook_url_env": webhook_url_env,
        "timeout_seconds": _as_positive_float(
            section["timeout_seconds"], "online.alerts.timeout_seconds"
        ),
        "max_attempts": _as_int(
            section["max_attempts"], "online.alerts.max_attempts", minimum=1
        ),
    }


def _validate_performance(value: object) -> OnlinePerformanceConfig:
    section = _as_mapping(value, "online.performance")
    _check_exact_keys(section, _PERFORMANCE_KEYS, "online.performance")

    baseline_accuracy = _as_optional_fraction(
        section["baseline_accuracy"], "online.performance.baseline_accuracy"
    )
    max_accuracy_drop = _as_optional_fraction(
        section["max_accuracy_drop"], "online.performance.max_accuracy_drop"
    )
    baseline_roc_auc = _as_optional_fraction(
        section["baseline_roc_auc"], "online.performance.baseline_roc_auc"
    )
    max_roc_auc_drop = _as_optional_fraction(
        section["max_roc_auc_drop"], "online.performance.max_roc_auc_drop"
    )

    for metric, baseline, drop in (
        ("accuracy", baseline_accuracy, max_accuracy_drop),
        ("roc_auc", baseline_roc_auc, max_roc_auc_drop),
    ):
        if (baseline is None) != (drop is None):
            raise ValueError(
                f"online.performance: baseline и max drop для {metric} "
                "должны задаваться вместе"
            )

    return {
        "enabled": _as_bool(section["enabled"], "online.performance.enabled"),
        "min_feedback_rows": _as_int(
            section["min_feedback_rows"],
            "online.performance.min_feedback_rows",
            minimum=2,
        ),
        "baseline_accuracy": baseline_accuracy,
        "max_accuracy_drop": max_accuracy_drop,
        "baseline_roc_auc": baseline_roc_auc,
        "max_roc_auc_drop": max_roc_auc_drop,
    }


def validate_monitoring_config(config: Mapping[str, Any]) -> MonitoringConfig:
    """Проверить общую analysis+online конфигурацию и вернуть независимую копию."""

    root = _as_mapping(config, "корневой конфигурации мониторинга")
    _check_exact_keys(root, _ROOT_KEYS, "корневая конфигурация мониторинга")

    analysis_source = {key: root[key] for key in _ANALYSIS_ROOT_KEYS}
    analysis = validate_config(analysis_source)

    online_section = _as_mapping(root["online"], "online")
    _check_exact_keys(online_section, _ONLINE_KEYS, "online")
    state_path = _as_nonempty_string(online_section["state_path"], "online.state_path")
    alerts = _validate_alerts(online_section["alerts"])
    if alerts["jsonl_path"] == state_path:
        raise ValueError("online.state_path и online.alerts.jsonl_path должны различаться")

    online: OnlineConfig = {
        "enabled": _as_bool(online_section["enabled"], "online.enabled"),
        "state_path": state_path,
        "max_request_bytes": _as_int(
            online_section["max_request_bytes"],
            "online.max_request_bytes",
            minimum=1,
        ),
        "window": _validate_window(online_section["window"]),
        "alerts": alerts,
        "performance": _validate_performance(online_section["performance"]),
    }
    return {"analysis": analysis, "online": online}


def load_monitoring_config(path: str | Path) -> MonitoringConfig:
    """Прочитать YAML с analysis- и online-секциями и строго его проверить."""

    if not isinstance(path, (str, Path)):
        raise TypeError(
            "path должен иметь тип str или pathlib.Path, "
            f"получен {type(path).__name__}"
        )

    config_path = Path(path).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(
            f"Файл конфигурации мониторинга не найден: {config_path}"
        )
    if config_path.is_dir():
        raise IsADirectoryError(
            "Ожидался файл конфигурации мониторинга, получена директория: "
            f"{config_path}"
        )
    if not config_path.is_file():
        raise ValueError(
            "Путь не указывает на обычный файл конфигурации мониторинга: "
            f"{config_path}"
        )

    extension = config_path.suffix.lower()
    if extension not in SUPPORTED_CONFIG_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_CONFIG_EXTENSIONS))
        actual = extension or "<без расширения>"
        raise ValueError(
            f"Неподдерживаемый формат конфигурации {actual}: {config_path}. "
            f"Поддерживаются: {supported}"
        )

    try:
        with config_path.open(encoding="utf-8") as stream:
            raw_config = yaml.safe_load(stream)
    except (UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(
            f"Не удалось прочитать YAML-конфигурацию {config_path}: {exc}"
        ) from exc

    return validate_monitoring_config(
        _as_mapping(raw_config, "корневой конфигурации мониторинга")
    )
