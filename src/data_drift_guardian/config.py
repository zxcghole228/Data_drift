"""Чтение и runtime-валидация конфигурации анализа."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import Any, cast

import yaml

from .contracts import (
    AdversarialConfig,
    AnalysisConfig,
    DistanceThresholds,
    DriftConfig,
    FeatureConfig,
    FeatureType,
    MultipleTestingMethod,
    QualityConfig,
    SchemaConfig,
)


SUPPORTED_CONFIG_EXTENSIONS = frozenset({".yaml", ".yml"})
CONTRACT_VERSION = "0.1"

_ROOT_KEYS = frozenset(
    {"contract_version", "random_seed", "schema", "quality", "drift", "adversarial"}
)
_SCHEMA_KEYS = frozenset({"allow_extra_columns", "features"})
_FEATURE_REQUIRED_KEYS = frozenset({"kind", "nullable"})
_FEATURE_OPTIONAL_KEYS = frozenset({"min", "max"})
_QUALITY_KEYS = frozenset(
    {"max_missing_fraction", "max_missing_increase_pp", "max_duplicate_fraction"}
)
_DRIFT_KEYS = frozenset(
    {
        "numeric_methods",
        "categorical_methods",
        "alpha",
        "multiple_testing",
        "n_bins",
        "psi_smoothing",
        "js_base",
        "distance_thresholds",
    }
)
_DISTANCE_THRESHOLD_KEYS = frozenset({"wasserstein", "psi", "js"})
_ADVERSARIAL_KEYS = frozenset(
    {"enabled", "n_splits", "roc_auc_threshold", "exclude_columns"}
)

_FEATURE_TYPES = frozenset({"numeric", "categorical"})
_NUMERIC_METHODS = frozenset({"ks", "wasserstein", "psi", "js"})
_CATEGORICAL_METHODS = frozenset({"chi2", "psi", "js"})
_MULTIPLE_TESTING_METHODS = frozenset({"none", "bh"})


def _as_mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} должен быть отображением ключ-значение")

    non_string_keys = [repr(key) for key in value if not isinstance(key, str)]
    if non_string_keys:
        raise ValueError(
            f"Все ключи {context} должны быть строками; "
            f"получены: {', '.join(non_string_keys)}"
        )
    return cast(Mapping[str, Any], value)


def _check_keys(
    value: Mapping[str, Any],
    *,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
    context: str,
) -> None:
    actual = set(value)
    missing = sorted(required - actual)
    unknown = sorted(actual - required - optional)

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


def _as_int(value: object, context: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise ValueError(f"{context} должен иметь тип int")
    if minimum is not None and value < minimum:
        raise ValueError(f"{context} должен быть не меньше {minimum}; получено {value}")
    return value


def _as_number(value: object, context: str) -> float | int:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{context} должен быть конечным числом")

    normalized = float(value)
    if not isfinite(normalized):
        raise ValueError(f"{context} должен быть конечным числом")
    return value if type(value) is int else normalized


def _as_fraction(value: object, context: str) -> float:
    normalized = float(_as_number(value, context))
    if not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{context} должен находиться в диапазоне [0, 1]")
    return normalized


def _as_method_list(
    value: object,
    *,
    allowed: frozenset[str],
    context: str,
) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{context} должен быть списком")
    if any(not isinstance(item, str) or item not in allowed for item in value):
        raise ValueError(
            f"{context} содержит неизвестный метод; допустимы: "
            f"{', '.join(sorted(allowed))}"
        )
    if len(value) != len(set(value)):
        raise ValueError(f"{context} не должен содержать повторяющиеся методы")
    return list(value)


def validate_schema_config(config: Mapping[str, Any]) -> SchemaConfig:
    """Проверить секцию ``schema`` и вернуть независимую нормализованную копию."""

    section = _as_mapping(config, "schema")
    _check_keys(section, required=_SCHEMA_KEYS, context="schema")

    allow_extra_columns = _as_bool(
        section["allow_extra_columns"], "schema.allow_extra_columns"
    )
    raw_features = _as_mapping(section["features"], "schema.features")
    if not raw_features:
        raise ValueError("schema.features не должен быть пустым")

    features: dict[str, FeatureConfig] = {}
    for name, raw_feature in raw_features.items():
        if not name.strip():
            raise ValueError("Имя признака в schema.features не должно быть пустым")

        context = f"schema.features.{name}"
        feature = _as_mapping(raw_feature, context)
        _check_keys(
            feature,
            required=_FEATURE_REQUIRED_KEYS,
            optional=_FEATURE_OPTIONAL_KEYS,
            context=context,
        )

        kind = feature["kind"]
        if not isinstance(kind, str) or kind not in _FEATURE_TYPES:
            raise ValueError(
                f"{context}.kind должен быть одним из: "
                f"{', '.join(sorted(_FEATURE_TYPES))}"
            )
        nullable = _as_bool(feature["nullable"], f"{context}.nullable")

        normalized_feature: FeatureConfig = {
            "kind": cast(FeatureType, kind),
            "nullable": nullable,
        }
        if kind == "categorical" and ({"min", "max"} & set(feature)):
            raise ValueError(
                f"{context}: границы min/max допустимы только для numeric-признаков"
            )

        if "min" in feature:
            normalized_feature["min"] = _as_number(feature["min"], f"{context}.min")
        if "max" in feature:
            normalized_feature["max"] = _as_number(feature["max"], f"{context}.max")
        if (
            "min" in normalized_feature
            and "max" in normalized_feature
            and normalized_feature["min"] > normalized_feature["max"]
        ):
            raise ValueError(f"{context}.min не должен превышать {context}.max")

        features[name] = normalized_feature

    return {"allow_extra_columns": allow_extra_columns, "features": features}


def _validate_quality_config(config: object) -> QualityConfig:
    section = _as_mapping(config, "quality")
    _check_keys(section, required=_QUALITY_KEYS, context="quality")

    missing_increase = float(
        _as_number(
            section["max_missing_increase_pp"], "quality.max_missing_increase_pp"
        )
    )
    if not 0.0 <= missing_increase <= 100.0:
        raise ValueError(
            "quality.max_missing_increase_pp должен находиться в диапазоне [0, 100]"
        )

    return {
        "max_missing_fraction": _as_fraction(
            section["max_missing_fraction"], "quality.max_missing_fraction"
        ),
        "max_missing_increase_pp": missing_increase,
        "max_duplicate_fraction": _as_fraction(
            section["max_duplicate_fraction"], "quality.max_duplicate_fraction"
        ),
    }


def _validate_distance_thresholds(config: object) -> DistanceThresholds:
    section = _as_mapping(config, "drift.distance_thresholds")
    _check_keys(
        section,
        required=_DISTANCE_THRESHOLD_KEYS,
        context="drift.distance_thresholds",
    )

    result: DistanceThresholds = {"wasserstein": None, "psi": None, "js": None}
    for name in ("wasserstein", "psi", "js"):
        value = section[name]
        if value is None:
            continue
        normalized = float(_as_number(value, f"drift.distance_thresholds.{name}"))
        if normalized < 0.0:
            raise ValueError(
                f"drift.distance_thresholds.{name} не должен быть отрицательным"
            )
        result[name] = normalized  # type: ignore[literal-required]
    return result


def _validate_drift_config(config: object) -> DriftConfig:
    section = _as_mapping(config, "drift")
    _check_keys(section, required=_DRIFT_KEYS, context="drift")

    alpha = float(_as_number(section["alpha"], "drift.alpha"))
    if not 0.0 < alpha < 1.0:
        raise ValueError("drift.alpha должен находиться в диапазоне (0, 1)")

    multiple_testing = section["multiple_testing"]
    if (
        not isinstance(multiple_testing, str)
        or multiple_testing not in _MULTIPLE_TESTING_METHODS
    ):
        raise ValueError(
            "drift.multiple_testing должен быть одним из: "
            + ", ".join(sorted(_MULTIPLE_TESTING_METHODS))
        )

    psi_smoothing = float(
        _as_number(section["psi_smoothing"], "drift.psi_smoothing")
    )
    if psi_smoothing <= 0.0:
        raise ValueError("drift.psi_smoothing должен быть больше 0")

    js_base = float(_as_number(section["js_base"], "drift.js_base"))
    if js_base <= 0.0 or js_base == 1.0:
        raise ValueError("drift.js_base должен быть больше 0 и не равен 1")

    return {
        "numeric_methods": cast(
            Any,
            _as_method_list(
                section["numeric_methods"],
                allowed=_NUMERIC_METHODS,
                context="drift.numeric_methods",
            ),
        ),
        "categorical_methods": cast(
            Any,
            _as_method_list(
                section["categorical_methods"],
                allowed=_CATEGORICAL_METHODS,
                context="drift.categorical_methods",
            ),
        ),
        "alpha": alpha,
        "multiple_testing": cast(MultipleTestingMethod, multiple_testing),
        "n_bins": _as_int(section["n_bins"], "drift.n_bins", minimum=2),
        "psi_smoothing": psi_smoothing,
        "js_base": js_base,
        "distance_thresholds": _validate_distance_thresholds(
            section["distance_thresholds"]
        ),
    }


def _validate_adversarial_config(config: object) -> AdversarialConfig:
    section = _as_mapping(config, "adversarial")
    _check_keys(section, required=_ADVERSARIAL_KEYS, context="adversarial")

    raw_threshold = section["roc_auc_threshold"]
    threshold: float | None
    if raw_threshold is None:
        threshold = None
    else:
        threshold = float(
            _as_number(raw_threshold, "adversarial.roc_auc_threshold")
        )
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "adversarial.roc_auc_threshold должен находиться в диапазоне [0, 1]"
            )

    raw_excluded = section["exclude_columns"]
    if not isinstance(raw_excluded, list) or any(
        not isinstance(name, str) or not name.strip() for name in raw_excluded
    ):
        raise ValueError(
            "adversarial.exclude_columns должен быть списком непустых строк"
        )
    if len(raw_excluded) != len(set(raw_excluded)):
        raise ValueError(
            "adversarial.exclude_columns не должен содержать повторяющиеся имена"
        )

    return {
        "enabled": _as_bool(section["enabled"], "adversarial.enabled"),
        "n_splits": _as_int(
            section["n_splits"], "adversarial.n_splits", minimum=2
        ),
        "roc_auc_threshold": threshold,
        "exclude_columns": list(raw_excluded),
    }


def validate_config(config: Mapping[str, Any]) -> AnalysisConfig:
    """Проверить полную конфигурацию и вернуть JSON-безопасную копию."""

    root = _as_mapping(config, "корневой конфигурации")
    _check_keys(root, required=_ROOT_KEYS, context="корневая конфигурация")

    if root["contract_version"] != CONTRACT_VERSION:
        raise ValueError(
            f"contract_version должен быть равен {CONTRACT_VERSION!r}; "
            f"получено {root['contract_version']!r}"
        )

    return {
        "contract_version": CONTRACT_VERSION,
        "random_seed": _as_int(root["random_seed"], "random_seed", minimum=0),
        "schema": validate_schema_config(
            _as_mapping(root["schema"], "schema")
        ),
        "quality": _validate_quality_config(root["quality"]),
        "drift": _validate_drift_config(root["drift"]),
        "adversarial": _validate_adversarial_config(root["adversarial"]),
    }


def load_config(path: str | Path) -> AnalysisConfig:
    """Прочитать YAML-конфигурацию и выполнить её runtime-валидацию."""

    if not isinstance(path, (str, Path)):
        raise TypeError(
            "path должен иметь тип str или pathlib.Path, "
            f"получен {type(path).__name__}"
        )

    config_path = Path(path).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Файл конфигурации не найден: {config_path}")
    if config_path.is_dir():
        raise IsADirectoryError(
            f"Ожидался файл конфигурации, получена директория: {config_path}"
        )
    if not config_path.is_file():
        raise ValueError(
            f"Путь не указывает на обычный файл конфигурации: {config_path}"
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

    return validate_config(_as_mapping(raw_config, "корневой конфигурации"))
