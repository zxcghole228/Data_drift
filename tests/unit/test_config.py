"""Тесты чтения и runtime-валидации конфигурации."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from data_drift_guardian.config import load_config, validate_config


@pytest.fixture
def valid_config() -> dict[str, Any]:
    return {
        "contract_version": "0.1",
        "random_seed": 42,
        "schema": {
            "allow_extra_columns": False,
            "features": {
                "age": {"kind": "numeric", "nullable": True, "min": 18, "max": 100},
                "income": {"kind": "numeric", "nullable": True, "min": 0},
                "region": {"kind": "categorical", "nullable": True},
            },
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


def test_validate_config_returns_independent_json_safe_copy(
    valid_config: dict[str, Any],
) -> None:
    original = copy.deepcopy(valid_config)

    result = validate_config(valid_config)

    assert result == original
    assert result is not valid_config
    result["schema"]["features"]["age"]["min"] = 21
    assert valid_config == original
    json.dumps(result, allow_nan=False)


def test_load_config_reads_yaml_from_path(
    tmp_path: Path,
    valid_config: dict[str, Any],
) -> None:
    config_path = tmp_path / "analysis.YAML"
    config_path.write_text(
        yaml.safe_dump(valid_config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    result = load_config(config_path)

    assert result == validate_config(valid_config)


def test_load_config_raises_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Файл конфигурации не найден"):
        load_config(tmp_path / "missing.yaml")


def test_load_config_raises_for_invalid_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "broken.yaml"
    config_path.write_text("schema: [", encoding="utf-8")

    with pytest.raises(ValueError, match="Не удалось прочитать YAML-конфигурацию"):
        load_config(config_path)


Mutation = Callable[[dict[str, Any]], None]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda config: config.__setitem__("unknown", 1), "неизвестные ключи"),
        (lambda config: config.pop("schema"), "обязательные ключи"),
        (
            lambda config: config.__setitem__("contract_version", "9.9"),
            "contract_version",
        ),
        (lambda config: config.__setitem__("random_seed", True), "random_seed"),
        (
            lambda config: config["schema"].__setitem__("unknown", 1),
            "неизвестные ключи",
        ),
        (
            lambda config: config["schema"]["features"]["age"].__setitem__(
                "kind", "datetime"
            ),
            "kind",
        ),
        (
            lambda config: config["schema"]["features"]["age"].update(
                {"min": 101, "max": 100}
            ),
            "min не должен превышать",
        ),
        (
            lambda config: config["schema"]["features"]["region"].__setitem__(
                "min", 0
            ),
            "только для numeric",
        ),
        (
            lambda config: config["quality"].__setitem__(
                "max_missing_fraction", 1.1
            ),
            r"диапазоне \[0, 1\]",
        ),
        (
            lambda config: config["drift"].__setitem__("alpha", 0.0),
            "drift.alpha",
        ),
        (
            lambda config: config["drift"].__setitem__(
                "numeric_methods", ["ks", "ks"]
            ),
            "повторяющиеся методы",
        ),
        (
            lambda config: config["drift"].__setitem__("n_bins", 1),
            "drift.n_bins",
        ),
        (
            lambda config: config["drift"].__setitem__("psi_smoothing", 0),
            "psi_smoothing",
        ),
        (
            lambda config: config["drift"].__setitem__("js_base", 1),
            "js_base",
        ),
        (
            lambda config: config["drift"]["distance_thresholds"].__setitem__(
                "psi", -0.1
            ),
            "не должен быть отрицательным",
        ),
        (
            lambda config: config["adversarial"].__setitem__("n_splits", 1),
            "n_splits",
        ),
        (
            lambda config: config["adversarial"].__setitem__(
                "roc_auc_threshold", 1.1
            ),
            "roc_auc_threshold",
        ),
        (
            lambda config: config["adversarial"].__setitem__(
                "exclude_columns", ["id", "id"]
            ),
            "повторяющиеся имена",
        ),
    ],
)
def test_invalid_config_raises_clear_value_error(
    valid_config: dict[str, Any],
    mutation: Mutation,
    message: str,
) -> None:
    mutation(valid_config)

    with pytest.raises(ValueError, match=message):
        validate_config(valid_config)
