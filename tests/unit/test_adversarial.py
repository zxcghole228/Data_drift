"""Tests for out-of-fold Adversarial Validation."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from data_drift_guardian.adversarial import adversarial_validate


def _config(
    *,
    n_splits: int = 3,
    threshold: float | None = None,
    exclude: list[str] | None = None,
) -> dict[str, object]:
    return {
        "enabled": True,
        "n_splits": n_splits,
        "roc_auc_threshold": threshold,
        "exclude_columns": [] if exclude is None else exclude,
        "random_seed": 42,
        "feature_types": {"value": "numeric", "group": "categorical"},
    }


def test_clear_shift_has_high_oof_auc_and_explainable_importance() -> None:
    rng = np.random.default_rng(1)
    reference = pd.DataFrame(
        {
            "value": rng.normal(0.0, 1.0, 150),
            "group": ["common"] * 150,
        }
    )
    current = pd.DataFrame(
        {
            "value": rng.normal(8.0, 1.0, 150),
            "group": ["common"] * 150,
        }
    )

    result = adversarial_validate(reference, current, _config(threshold=0.8))

    assert result["status"] == "warning"
    assert result["roc_auc"] is not None
    assert result["roc_auc"] > 0.95
    assert result["threshold"] == 0.8
    assert result["alert"] is True
    assert len(result["fold_auc"]) == 3
    assert set(result["feature_importance"]) == {"value", "group"}
    assert sum(result["feature_importance"].values()) == pytest.approx(1.0)
    assert result["feature_importance"]["value"] > 0.9
    assert result["importance_type"] == "mean_normalized_gain_across_folds"


def test_independent_same_distribution_is_near_random_and_reproducible() -> None:
    rng = np.random.default_rng(7)
    reference = pd.DataFrame(
        {
            "value": rng.normal(0.0, 1.0, 300),
            "group": rng.choice(["a", "b", "c"], 300),
        }
    )
    current = pd.DataFrame(
        {
            "value": rng.normal(0.0, 1.0, 300),
            "group": rng.choice(["a", "b", "c"], 300),
        }
    )

    first = adversarial_validate(reference, current, _config())
    second = adversarial_validate(reference, current, _config())

    assert first == second
    assert first["status"] == "ok"
    assert first["roc_auc"] is not None
    assert 0.35 <= first["roc_auc"] <= 0.65
    assert first["alert"] is None
    assert first["threshold"] is None


def test_new_validation_categories_and_missing_values_do_not_crash() -> None:
    reference = pd.DataFrame(
        {
            "value": [float(index) for index in range(60)],
            "group": ["a", "b", None] * 20,
        }
    )
    current = pd.DataFrame(
        {
            "value": [float(index) for index in range(60)],
            "group": ["a"] * 58 + ["new-only-on-two-rows", None],
        }
    )

    result = adversarial_validate(reference, current, _config())

    assert result["status"] == "ok"
    assert result["roc_auc"] is not None
    assert len(result["fold_auc"]) == 3


def test_infinities_are_treated_as_missing_without_mutating_inputs() -> None:
    reference = pd.DataFrame(
        {"value": [0.0, 1.0, np.inf] * 20, "group": ["a", "b"] * 30}
    )
    current = pd.DataFrame(
        {"value": [0.0, 1.0, -np.inf] * 20, "group": ["a", "b"] * 30}
    )
    original_reference = reference.copy(deep=True)
    original_current = current.copy(deep=True)

    result = adversarial_validate(reference, current, _config())

    assert result["status"] == "ok"
    pd.testing.assert_frame_equal(reference, original_reference)
    pd.testing.assert_frame_equal(current, original_current)
    json.dumps(result, allow_nan=False)


def test_excluded_identifier_and_target_never_reach_model() -> None:
    reference = pd.DataFrame(
        {
            "value": [0.0] * 80,
            "group": ["same"] * 80,
            "id": np.arange(80),
            "target": [0] * 80,
        }
    )
    current = pd.DataFrame(
        {
            "value": [0.0] * 80,
            "group": ["same"] * 80,
            "id": np.arange(1_000, 1_080),
            "target": [1] * 80,
        }
    )
    config = _config(exclude=["id", "target"])

    result = adversarial_validate(reference, current, config)

    assert result["status"] == "ok"
    assert result["roc_auc"] is not None
    assert 0.45 <= result["roc_auc"] <= 0.55
    assert set(result["feature_importance"]) == {"value", "group"}


@pytest.mark.parametrize(
    ("n_reference", "n_current", "n_splits", "reason"),
    [
        (0, 10, 2, "непустые"),
        (2, 10, 3, "n_splits=3"),
    ],
)
def test_small_batches_are_skipped(
    n_reference: int,
    n_current: int,
    n_splits: int,
    reason: str,
) -> None:
    reference = pd.DataFrame(
        {"value": np.arange(n_reference), "group": ["a"] * n_reference}
    )
    current = pd.DataFrame({"value": np.arange(n_current), "group": ["a"] * n_current})

    result = adversarial_validate(reference, current, _config(n_splits=n_splits))

    assert result["status"] == "skipped"
    assert result["roc_auc"] is None
    assert result["alert"] is None
    assert reason in str(result["reason"])


def test_no_common_or_non_excluded_features_is_skipped() -> None:
    reference = pd.DataFrame({"value": np.arange(20), "group": ["a"] * 20})
    current = reference.copy(deep=True)

    result = adversarial_validate(
        reference,
        current,
        _config(exclude=["value", "group"]),
    )

    assert result["status"] == "skipped"
    assert "не осталось" in str(result["reason"])


def test_threshold_not_exceeded_produces_explicit_false_alert() -> None:
    reference = pd.DataFrame({"value": np.arange(80), "group": ["a", "b"] * 40})
    current = reference.copy(deep=True)

    result = adversarial_validate(reference, current, _config(threshold=0.8))

    assert result["status"] == "ok"
    assert result["alert"] is False
    assert result["reason"] is None


@pytest.mark.parametrize(
    "change",
    [
        {"n_splits": 1},
        {"n_splits": True},
        {"roc_auc_threshold": 1.1},
        {"exclude_columns": ["x", "x"]},
        {"random_seed": -1},
        {"feature_types": {"value": "unknown"}},
        {"unexpected": 1},
    ],
)
def test_invalid_configuration_raises_value_error(change: dict[str, object]) -> None:
    config = _config()
    config.update(change)
    frame = pd.DataFrame({"value": np.arange(10), "group": ["a"] * 10})

    with pytest.raises(ValueError):
        adversarial_validate(frame, frame, config)


def test_invalid_input_types_raise_type_error() -> None:
    frame = pd.DataFrame({"value": np.arange(10), "group": ["a"] * 10})
    with pytest.raises(TypeError, match="pandas.DataFrame"):
        adversarial_validate([], frame, _config())  # type: ignore[arg-type]
