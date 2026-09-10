"""Интеграционные тесты общего API ``analyze``."""

from __future__ import annotations

import copy
import json
from typing import Any

import numpy as np
import pandas as pd
import pytest

from data_drift_guardian import analyze
from data_drift_guardian.config import load_config


@pytest.fixture
def valid_config() -> dict[str, Any]:
    return load_config("configs/default.yaml")


@pytest.fixture
def healthy_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    reference = pd.DataFrame(
        {
            "age": np.arange(20.0, 40.0),
            "income": np.arange(50_000.0, 70_000.0, 1_000.0),
            "region": pd.Series(
                ["central", "northwest", "south", "volga"] * 5,
                dtype="string",
            ),
        }
    )
    current = pd.DataFrame(
        {
            "age": np.arange(21.0, 41.0),
            "income": np.arange(51_000.0, 71_000.0, 1_000.0),
            "region": pd.Series(
                ["central", "northwest", "south", "volga"] * 5,
                dtype="string",
            ),
        }
    )
    return reference, current


def _single_numeric_config() -> dict[str, Any]:
    return {
        "contract_version": "0.1",
        "random_seed": 7,
        "schema": {
            "allow_extra_columns": False,
            "features": {
                "value": {"kind": "numeric", "nullable": True},
            },
        },
        "quality": {
            "max_missing_fraction": 1.0,
            "max_missing_increase_pp": 100.0,
            "max_duplicate_fraction": 1.0,
        },
        "drift": {
            "numeric_methods": ["wasserstein"],
            "categorical_methods": [],
            "alpha": 0.05,
            "multiple_testing": "none",
            "n_bins": 5,
            "psi_smoothing": 0.000001,
            "js_base": 2,
            "distance_thresholds": {
                "wasserstein": 5.0,
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


def test_normal_pass_combines_real_modules_and_preserves_inputs(
    valid_config: dict[str, Any],
    healthy_frames: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    reference, current = healthy_frames
    original_reference = reference.copy(deep=True)
    original_current = current.copy(deep=True)
    original_config = copy.deepcopy(valid_config)

    result = analyze(reference, current, valid_config)

    assert result["contract_version"] == "0.1"
    assert result["metadata"] == {
        "reference_rows": 20,
        "current_rows": 20,
        "random_seed": 42,
    }
    assert result["schema"]["status"] == "ok"
    assert result["quality"]["status"] == "ok"
    assert result["drift"]["status"] == "ok"
    assert list(result["drift"]["features"]) == ["age", "income", "region"]
    assert list(result["drift"]["features"]["age"]["checks"]) == [
        "ks",
        "wasserstein",
        "psi",
        "js",
    ]

    region_checks = result["drift"]["features"]["region"]["checks"]
    assert region_checks["chi2"]["status"] == "skipped"
    assert region_checks["chi2"]["reason"] is not None
    assert region_checks["psi"]["status"] == "ok"
    assert region_checks["js"]["status"] == "ok"
    assert result["adversarial"]["status"] == "skipped"
    assert result["summary"] == {
        "status": "ok",
        "has_alerts": False,
        "n_alerts": 0,
        "analyzed_features": 3,
        "skipped_features": 0,
    }

    pd.testing.assert_frame_equal(reference, original_reference)
    pd.testing.assert_frame_equal(current, original_current)
    assert valid_config == original_config
    json.dumps(result, allow_nan=False)


def test_incompatible_feature_is_skipped_without_cancelling_other_features(
    valid_config: dict[str, Any],
    healthy_frames: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    reference, current = healthy_frames
    current["age"] = pd.Series([str(value) for value in current["age"]], dtype="string")

    result = analyze(reference, current, valid_config)

    assert result["schema"]["status"] == "error"
    assert result["schema"]["valid_features"] == ["income", "region"]
    assert list(result["quality"]["feature_checks"]) == ["income", "region"]
    assert result["drift"]["features"]["age"]["status"] == "skipped"
    assert result["drift"]["features"]["age"]["checks"] == {}
    assert result["drift"]["features"]["age"]["reason"] is not None
    assert result["drift"]["features"]["income"]["status"] == "ok"
    assert result["drift"]["features"]["region"]["status"] == "ok"
    assert result["summary"]["status"] == "error"
    assert result["summary"]["analyzed_features"] == 2
    assert result["summary"]["skipped_features"] == 1
    assert any(
        alert["source"] == "schema"
        and alert["feature"] == "age"
        and alert["check"] == "type_mismatch"
        for alert in result["alerts"]
    )
    json.dumps(result, allow_nan=False)


def test_config_none_infers_feature_types_only_from_reference() -> None:
    reference = pd.DataFrame(
        {
            "value": np.arange(20.0),
            "group": pd.Series([f"group-{index % 3}" for index in range(20)]),
            "active": pd.Series([True, False] * 10, dtype="bool"),
        }
    )
    current = pd.DataFrame(
        {
            "value": np.arange(20.0),
            "group": pd.Series([f"group-{index % 3}" for index in range(20)]),
            "active": pd.Series([True, False] * 10, dtype="bool"),
        }
    )

    result = analyze(reference, current)

    assert result["schema"]["status"] == "ok"
    assert result["schema"]["valid_features"] == ["value", "group", "active"]
    assert result["drift"]["features"]["value"]["feature_type"] == "numeric"
    assert result["drift"]["features"]["group"]["feature_type"] == "categorical"
    assert result["drift"]["features"]["active"]["feature_type"] == "categorical"
    assert {
        check["name"] for check in result["quality"]["feature_checks"]["value"]
    } == {
        "reference_missing_fraction",
        "current_missing_fraction",
        "missing_increase_pp",
        "reference_infinite_count",
        "current_infinite_count",
    }
    json.dumps(result, allow_nan=False)


def test_configured_distance_threshold_creates_explainable_drift_alert() -> None:
    config = _single_numeric_config()
    reference = pd.DataFrame({"value": [0.0, 1.0, 2.0]})
    current = pd.DataFrame({"value": [10.0, 11.0, 12.0]})

    result = analyze(reference, current, config)

    check = result["drift"]["features"]["value"]["checks"]["wasserstein"]
    assert check["value"] == pytest.approx(10.0)
    assert check["threshold"] == 5.0
    assert check["alert"] is True
    assert check["status"] == "warning"
    assert result["drift"]["features"]["value"]["alert"] is True
    assert result["summary"]["status"] == "warning"
    assert result["summary"]["has_alerts"] is True
    assert result["summary"]["n_alerts"] == 1
    assert result["alerts"][0]["source"] == "drift"
    assert result["alerts"][0]["feature"] == "value"
    assert result["alerts"][0]["check"] == "wasserstein"


def test_raw_p_value_rule_is_used_only_when_multiple_testing_is_none() -> None:
    config = _single_numeric_config()
    config["drift"]["numeric_methods"] = ["ks"]
    reference = pd.DataFrame({"value": np.arange(100.0)})
    current = pd.DataFrame({"value": np.arange(100.0) + 1_000.0})

    result = analyze(reference, current, config)

    check = result["drift"]["features"]["value"]["checks"]["ks"]
    assert check["p_value"] is not None
    assert check["p_value"] < config["drift"]["alpha"]
    assert check["threshold"] == config["drift"]["alpha"]
    assert check["alert"] is True
    assert check["details"]["decision"] == "raw_p_value"


def test_bh_configuration_defers_alert_until_correction_is_available(
    valid_config: dict[str, Any],
    healthy_frames: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    reference, current = healthy_frames
    current["age"] = current["age"] + 1_000.0

    result = analyze(reference, current, valid_config)

    check = result["drift"]["features"]["age"]["checks"]["ks"]
    assert check["p_value"] is not None
    assert check["p_value"] < valid_config["drift"]["alpha"]
    assert check["threshold"] == valid_config["drift"]["alpha"]
    assert check["adjusted_p_value"] is None
    assert check["alert"] is None
    assert check["details"]["decision"] == "deferred_until_bh_correction"


def test_all_missing_feature_is_explicitly_skipped_in_drift() -> None:
    config = _single_numeric_config()
    config["drift"]["numeric_methods"] = ["ks", "wasserstein", "psi", "js"]
    reference = pd.DataFrame({"value": [np.nan, np.nan, np.nan]})
    current = pd.DataFrame({"value": [np.nan, np.nan, np.nan]})

    result = analyze(reference, current, config)

    feature = result["drift"]["features"]["value"]
    assert feature["status"] == "skipped"
    assert feature["n_reference_valid"] == 0
    assert feature["n_current_valid"] == 0
    assert set(feature["checks"]) == {"ks", "wasserstein", "psi", "js"}
    assert all(check["status"] == "skipped" for check in feature["checks"].values())
    assert result["summary"]["skipped_features"] == 1
    json.dumps(result, allow_nan=False)


def test_enabled_unavailable_adversarial_module_is_explicitly_skipped(
    valid_config: dict[str, Any],
    healthy_frames: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    reference, current = healthy_frames
    valid_config["adversarial"]["enabled"] = True

    result = analyze(reference, current, valid_config)

    assert result["adversarial"]["status"] == "skipped"
    assert result["adversarial"]["alert"] is None
    assert result["adversarial"]["reason"] is not None
    assert "не реализован" in result["adversarial"]["reason"]


def test_invalid_inputs_follow_public_error_contract(
    valid_config: dict[str, Any],
    healthy_frames: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    reference, current = healthy_frames

    with pytest.raises(TypeError, match="pandas.DataFrame"):
        analyze([], current, valid_config)  # type: ignore[arg-type]

    invalid_config = copy.deepcopy(valid_config)
    invalid_config["unknown"] = True
    with pytest.raises(ValueError, match="неизвестные ключи"):
        analyze(reference, current, invalid_config)
