"""Оценка качества модели по запаздывающему production Feedback."""

from __future__ import annotations

from typing import Any

from sklearn.metrics import roc_auc_score

from .contracts import OnlinePerformanceConfig, PerformanceSample, PerformanceStatus
from .storage import canonical_json


def _binary_label(value: object) -> int | None:
    if type(value) is bool:
        return int(value)
    if type(value) is int and value in {0, 1}:
        return value
    if type(value) is float and value in {0.0, 1.0}:
        return int(value)
    return None


def _metric_result(
    *,
    value: float | None,
    sample_rows: int,
    baseline: float | None,
    max_drop: float | None,
    unavailable_reason: str | None,
) -> dict[str, Any]:
    drop = None if value is None or baseline is None else baseline - value
    alert = bool(
        drop is not None
        and max_drop is not None
        and drop > max_drop
    )
    return {
        "value": value,
        "sample_rows": sample_rows,
        "baseline": baseline,
        "max_drop": max_drop,
        "drop": drop,
        "alert": alert,
        "reason": unavailable_reason if value is None else None,
    }


def _performance_alert(metric: str, result: dict[str, Any]) -> dict[str, Any]:
    value = result["value"]
    baseline = result["baseline"]
    drop = result["drop"]
    max_drop = result["max_drop"]
    return {
        "source": "performance",
        "feature": None,
        "check": "performance_degradation",
        "metric": metric,
        "severity": "warning",
        "value": value,
        "baseline": baseline,
        "drop": drop,
        "max_drop": max_drop,
        "message": (
            f"{metric}: value={value:.6g}, baseline={baseline:.6g}, "
            f"drop={drop:.6g}, max_drop={max_drop:.6g}. "
            "Падение качества превышает допустимый порог; это сигнал "
            "возможного concept drift, а не его причинное доказательство"
        ),
    }


def evaluate_performance(
    samples: list[PerformanceSample],
    config: OnlinePerformanceConfig,
) -> tuple[PerformanceStatus, dict[str, Any]]:
    """Посчитать доступные метрики и не выдавать отсутствие меток за норму."""

    minimum = config["min_feedback_rows"]
    feedback_rows = len(samples)
    disabled = not config["enabled"]

    prediction_samples = [sample for sample in samples if sample.prediction is not None]
    binary_score_samples = [
        (label, sample.score)
        for sample in samples
        if sample.score is not None
        and (label := _binary_label(sample.y_true)) is not None
    ]

    accuracy_value: float | None = None
    accuracy_reason: str | None = None
    roc_auc_value: float | None = None
    roc_auc_reason: str | None = None

    if disabled:
        accuracy_reason = "performance_disabled"
        roc_auc_reason = "performance_disabled"
    elif feedback_rows < minimum:
        accuracy_reason = "insufficient_feedback"
        roc_auc_reason = "insufficient_feedback"
    else:
        if len(prediction_samples) >= minimum:
            correct = sum(
                canonical_json(sample.prediction) == canonical_json(sample.y_true)
                for sample in prediction_samples
            )
            accuracy_value = correct / len(prediction_samples)
        else:
            accuracy_reason = "insufficient_prediction_rows"

        if len(binary_score_samples) < minimum:
            roc_auc_reason = "insufficient_binary_score_rows"
        elif len({label for label, _ in binary_score_samples}) < 2:
            roc_auc_reason = "single_class_feedback"
        else:
            roc_auc_value = float(
                roc_auc_score(
                    [label for label, _ in binary_score_samples],
                    [score for _, score in binary_score_samples],
                )
            )

    accuracy = _metric_result(
        value=accuracy_value,
        sample_rows=len(prediction_samples),
        baseline=config["baseline_accuracy"],
        max_drop=config["max_accuracy_drop"],
        unavailable_reason=accuracy_reason,
    )
    roc_auc = _metric_result(
        value=roc_auc_value,
        sample_rows=len(binary_score_samples),
        baseline=config["baseline_roc_auc"],
        max_drop=config["max_roc_auc_drop"],
        unavailable_reason=roc_auc_reason,
    )

    alerts = [
        _performance_alert(metric, result)
        for metric, result in (("accuracy", accuracy), ("roc_auc", roc_auc))
        if result["alert"]
    ]
    status: PerformanceStatus = (
        "evaluated"
        if accuracy_value is not None or roc_auc_value is not None
        else "not_evaluated"
    )
    if disabled:
        reason = "performance_disabled"
    elif feedback_rows < minimum:
        reason = "insufficient_feedback"
    elif status == "not_evaluated":
        reason = "metrics_unavailable"
    else:
        reason = None

    return status, {
        "status": status,
        "reason": reason,
        "feedback_rows": feedback_rows,
        "min_feedback_rows": minimum,
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "alerts": alerts,
    }
