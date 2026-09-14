"""Tests for BH correction and explainable drift alerts."""

from __future__ import annotations

import copy
import json

import pytest

from data_drift_guardian.alerts import apply_multiple_testing, build_drift_alerts
from data_drift_guardian.contracts import CheckResult, FeatureResult


def _check(name: str, p_value: float | None, status: str = "ok") -> CheckResult:
    return {
        "name": name,
        "status": status,  # type: ignore[typeddict-item]
        "value": 1.0 if p_value is not None else None,
        "threshold": 0.05 if p_value is not None else None,
        "p_value": p_value,
        "adjusted_p_value": None,
        "alert": None,
        "reason": None if status == "ok" else "not applicable",
        "details": {
            "decision": "deferred_until_bh_correction",
            "decision_quantity": "p_value",
        },
    }


def _feature(checks: dict[str, CheckResult]) -> FeatureResult:
    return {
        "feature_type": "numeric",
        "status": "ok",
        "n_reference_valid": 100,
        "n_current_valid": 100,
        "checks": checks,
        "alert": None,
        "reason": None,
    }


def test_bh_matches_known_adjusted_p_values_and_preserves_raw_values() -> None:
    features = {
        "a": _feature({"ks": _check("ks", 0.01)}),
        "b": _feature({"chi2": _check("chi2", 0.04)}),
        "c": _feature({"ks": _check("ks", 0.03)}),
        "d": _feature({"chi2": _check("chi2", 0.002)}),
    }

    result = apply_multiple_testing(
        features,
        {"multiple_testing": "bh", "alpha": 0.05},
    )

    assert result["a"]["checks"]["ks"]["adjusted_p_value"] == pytest.approx(0.02)
    assert result["b"]["checks"]["chi2"]["adjusted_p_value"] == pytest.approx(0.04)
    assert result["c"]["checks"]["ks"]["adjusted_p_value"] == pytest.approx(0.04)
    assert result["d"]["checks"]["chi2"]["adjusted_p_value"] == pytest.approx(0.008)
    assert result["a"]["checks"]["ks"]["p_value"] == 0.01
    assert all(feature["alert"] is True for feature in result.values())
    assert all(
        feature["checks"][next(iter(feature["checks"]))]["details"][
            "multiple_testing_family_size"
        ]
        == 4
        for feature in result.values()
    )


def test_skipped_checks_and_non_p_value_metrics_are_not_in_family() -> None:
    features = {
        "a": _feature(
            {
                "ks": _check("ks", 0.01),
                "wasserstein": _check("wasserstein", None),
            }
        ),
        "b": _feature({"chi2": _check("chi2", None, status="skipped")}),
    }

    result = apply_multiple_testing(
        features,
        {"multiple_testing": "bh", "alpha": 0.05},
    )

    ks = result["a"]["checks"]["ks"]
    assert ks["adjusted_p_value"] == pytest.approx(0.01)
    assert ks["details"]["multiple_testing_family_size"] == 1
    assert result["a"]["checks"]["wasserstein"]["adjusted_p_value"] is None
    assert result["b"]["checks"]["chi2"]["status"] == "skipped"


def test_adjusted_p_value_equal_to_alpha_does_not_trigger_strict_rule() -> None:
    features = {"a": _feature({"ks": _check("ks", 0.05)})}

    result = apply_multiple_testing(
        features,
        {"multiple_testing": "bh", "alpha": 0.05},
    )

    check = result["a"]["checks"]["ks"]
    assert check["adjusted_p_value"] == pytest.approx(0.05)
    assert check["alert"] is False
    assert check["status"] == "ok"


def test_none_method_returns_independent_copy_without_bh_changes() -> None:
    features = {"a": _feature({"ks": _check("ks", 0.01)})}

    result = apply_multiple_testing(
        features,
        {"multiple_testing": "none", "alpha": 0.05},
    )

    assert result == features
    assert result is not features
    assert result["a"] is not features["a"]


def test_alert_message_contains_metric_value_threshold_and_explanation() -> None:
    features = {"age": _feature({"ks": _check("ks", 0.001)})}
    corrected = apply_multiple_testing(
        features,
        {"multiple_testing": "bh", "alpha": 0.05},
    )

    alerts = build_drift_alerts(
        corrected,
        {"multiple_testing": "bh", "alpha": 0.05},
    )

    assert len(alerts) == 1
    alert = alerts[0]
    assert alert["source"] == "drift"
    assert alert["feature"] == "age"
    assert alert["check"] == "ks"
    assert "adjusted_p_value=0.001" in alert["message"]
    assert "raw_p_value=0.001" in alert["message"]
    assert "alpha=0.05" in alert["message"]
    assert "меньше" in alert["message"]


def test_distance_alert_message_uses_value_and_threshold() -> None:
    check = _check("wasserstein", None)
    check.update(
        {
            "status": "warning",
            "value": 8.0,
            "threshold": 5.0,
            "alert": True,
            "reason": "Метрика превышает порог",
        }
    )
    check["details"]["decision_quantity"] = "value"
    features = {"age": _feature({"wasserstein": check})}

    alerts = build_drift_alerts(
        features,
        {"multiple_testing": "none", "alpha": 0.05},
    )

    assert "value=8" in alerts[0]["message"]
    assert "threshold=5" in alerts[0]["message"]


def test_non_triggered_and_undecided_checks_do_not_create_alerts() -> None:
    no_alert = _check("ks", 0.5)
    no_alert["alert"] = False
    undecided = _check("wasserstein", None)
    features = {"x": _feature({"ks": no_alert, "wasserstein": undecided})}

    assert (
        build_drift_alerts(
            features,
            {"multiple_testing": "none", "alpha": 0.05},
        )
        == []
    )


def test_functions_do_not_mutate_features_and_results_are_json_safe() -> None:
    features = {"a": _feature({"ks": _check("ks", 0.001)})}
    original = copy.deepcopy(features)

    corrected = apply_multiple_testing(
        features,
        {"multiple_testing": "bh", "alpha": 0.05},
    )
    alerts = build_drift_alerts(
        corrected,
        {"multiple_testing": "bh", "alpha": 0.05},
    )

    assert features == original
    json.dumps({"features": corrected, "alerts": alerts}, allow_nan=False)


@pytest.mark.parametrize(
    "config",
    [
        {"multiple_testing": "unknown", "alpha": 0.05},
        {"multiple_testing": "bh", "alpha": 0.0},
        {"multiple_testing": "bh", "alpha": True},
    ],
)
def test_invalid_settings_raise_value_error(config: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        apply_multiple_testing({}, config)
