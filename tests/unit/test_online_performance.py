"""Тесты метрик качества по delayed Feedback."""

from __future__ import annotations

import pytest

from data_drift_guardian.online.contracts import (
    OnlinePerformanceConfig,
    PerformanceSample,
)
from data_drift_guardian.online.performance import evaluate_performance


def performance_config(*, enabled: bool = True) -> OnlinePerformanceConfig:
    return {
        "enabled": enabled,
        "min_feedback_rows": 4,
        "baseline_accuracy": 0.9,
        "max_accuracy_drop": 0.1,
        "baseline_roc_auc": 0.9,
        "max_roc_auc_drop": 0.1,
    }


def samples(*, degraded: bool = False) -> list[PerformanceSample]:
    labels = [0, 1, 0, 1]
    predictions = [1 - label if degraded else label for label in labels]
    scores = [0.9 if prediction == 1 else 0.1 for prediction in predictions]
    return [
        PerformanceSample(
            event_id=f"event-{index}",
            prediction=prediction,
            score=score,
            y_true=label,
        )
        for index, (label, prediction, score) in enumerate(
            zip(labels, predictions, scores, strict=True)
        )
    ]


def test_stable_feedback_is_evaluated_without_alerts() -> None:
    status, result = evaluate_performance(samples(), performance_config())

    assert status == "evaluated"
    assert result["reason"] is None
    assert result["accuracy"]["value"] == 1.0
    assert result["roc_auc"]["value"] == 1.0
    assert result["alerts"] == []


def test_accuracy_and_roc_auc_drop_create_distinct_performance_alerts() -> None:
    status, result = evaluate_performance(
        samples(degraded=True),
        performance_config(),
    )

    assert status == "evaluated"
    assert result["accuracy"]["value"] == 0.0
    assert result["roc_auc"]["value"] == 0.0
    assert result["accuracy"]["alert"] is True
    assert result["roc_auc"]["alert"] is True
    assert {alert["metric"] for alert in result["alerts"]} == {
        "accuracy",
        "roc_auc",
    }
    assert all(
        alert["check"] == "performance_degradation"
        for alert in result["alerts"]
    )


@pytest.mark.parametrize(
    ("input_samples", "config", "reason"),
    [
        (samples()[:2], performance_config(), "insufficient_feedback"),
        (samples(), performance_config(enabled=False), "performance_disabled"),
    ],
)
def test_missing_or_disabled_feedback_is_not_reported_as_success(
    input_samples: list[PerformanceSample],
    config: OnlinePerformanceConfig,
    reason: str,
) -> None:
    status, result = evaluate_performance(input_samples, config)

    assert status == "not_evaluated"
    assert result["reason"] == reason
    assert result["accuracy"]["value"] is None
    assert result["roc_auc"]["value"] is None
    assert result["alerts"] == []
