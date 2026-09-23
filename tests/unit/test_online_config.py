"""Тесты строгой конфигурации online-мониторинга."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from data_drift_guardian.online.config import (
    load_monitoring_config,
    validate_monitoring_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def valid_monitoring_config() -> dict[str, Any]:
    return yaml.safe_load(
        (PROJECT_ROOT / "configs" / "online.yaml").read_text(encoding="utf-8")
    )


def test_validate_monitoring_config_returns_json_safe_independent_copy(
    valid_monitoring_config: dict[str, Any],
) -> None:
    original = copy.deepcopy(valid_monitoring_config)

    result = validate_monitoring_config(valid_monitoring_config)

    assert result["analysis"]["contract_version"] == "0.2"
    assert result["online"]["window"] == {
        "strategy": "tumbling",
        "size": 500,
        "min_batch_rows": 100,
    }
    assert result["online"]["alerts"]["timeout_seconds"] == 5.0
    assert result["online"]["max_request_bytes"] == 25_000_000
    result["online"]["window"]["size"] = 999
    assert valid_monitoring_config == original
    json.dumps(result, allow_nan=False)


def test_load_monitoring_config_reads_yaml(
    tmp_path: Path,
    valid_monitoring_config: dict[str, Any],
) -> None:
    path = tmp_path / "monitoring.YAML"
    path.write_text(
        yaml.safe_dump(valid_monitoring_config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    assert load_monitoring_config(path) == validate_monitoring_config(
        valid_monitoring_config
    )


def test_online_demo_uses_feature_specific_wasserstein_thresholds(
    valid_monitoring_config: dict[str, Any],
) -> None:
    drift = validate_monitoring_config(valid_monitoring_config)["analysis"]["drift"]

    assert drift["distance_thresholds"]["wasserstein"] is None
    assert drift["feature_thresholds"]["age"]["wasserstein"] == 3.0
    assert drift["feature_thresholds"]["income"]["wasserstein"] is None


def test_load_monitoring_config_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="не найден"):
        load_monitoring_config(tmp_path / "missing.yaml")


def test_load_monitoring_config_rejects_invalid_yaml(tmp_path: Path) -> None:
    path = tmp_path / "broken.yaml"
    path.write_text("online: [", encoding="utf-8")

    with pytest.raises(ValueError, match="Не удалось прочитать YAML"):
        load_monitoring_config(path)


Mutation = Callable[[dict[str, Any]], None]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda config: config.pop("online"), "обязательные ключи"),
        (lambda config: config.__setitem__("unknown", 1), "неизвестные ключи"),
        (
            lambda config: config["online"].__setitem__("enabled", "yes"),
            "online.enabled",
        ),
        (
            lambda config: config["online"].__setitem__("state_path", ""),
            "online.state_path",
        ),
        (
            lambda config: config["online"].__setitem__("max_request_bytes", 0),
            "max_request_bytes",
        ),
        (
            lambda config: config["online"]["window"].__setitem__(
                "strategy", "sliding"
            ),
            "strategy",
        ),
        (
            lambda config: config["online"]["window"].__setitem__("size", 1),
            "window.size",
        ),
        (
            lambda config: config["online"]["window"].update(
                {"size": 100, "min_batch_rows": 101}
            ),
            "не должен превышать",
        ),
        (
            lambda config: config["online"]["alerts"].__setitem__(
                "webhook_url_env", "NOT-AN-ENV"
            ),
            "переменной окружения",
        ),
        (
            lambda config: config["online"]["alerts"].__setitem__(
                "timeout_seconds", 0
            ),
            "timeout_seconds",
        ),
        (
            lambda config: config["online"]["alerts"].__setitem__(
                "max_attempts", 0
            ),
            "max_attempts",
        ),
        (
            lambda config: config["online"]["performance"].__setitem__(
                "min_feedback_rows", 1
            ),
            "min_feedback_rows",
        ),
        (
            lambda config: config["online"]["performance"].__setitem__(
                "baseline_accuracy", 1.1
            ),
            r"диапазоне \[0, 1\]",
        ),
        (
            lambda config: config["online"]["performance"].__setitem__(
                "baseline_accuracy", 0.8
            ),
            "должны задаваться вместе",
        ),
        (
            lambda config: config["online"].update(
                {
                    "state_path": "same.path",
                    "alerts": {
                        **config["online"]["alerts"],
                        "jsonl_path": "same.path",
                    },
                }
            ),
            "должны различаться",
        ),
    ],
)
def test_invalid_monitoring_config_is_rejected(
    valid_monitoring_config: dict[str, Any],
    mutation: Mutation,
    message: str,
) -> None:
    mutation(valid_monitoring_config)

    with pytest.raises(ValueError, match=message):
        validate_monitoring_config(valid_monitoring_config)
