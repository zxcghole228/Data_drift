"""Tests for out-of-fold Adversarial Validation."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from data_drift_guardian.adversarial import (
    _prepare_groups,
    _split_indices,
    adversarial_validate,
)


def _config(
    *,
    n_splits: int = 3,
    threshold: float | None = None,
    exclude: list[str] | None = None,
    group_column: str | None = None,
) -> dict[str, object]:
    return {
        "enabled": True,
        "n_splits": n_splits,
        "roc_auc_threshold": threshold,
        "exclude_columns": [] if exclude is None else exclude,
        "group_column": group_column,
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
    assert result["split_strategy"] == "stratified_kfold"
    assert result["group_column"] is None
    assert result["n_groups"] is None


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


def test_grouped_split_has_no_entity_overlap_and_is_reproducible() -> None:
    entity_ids = [f"entity-{index}" for index in range(30) for _ in range(2)]
    reference = pd.DataFrame(
        {
            "value": np.linspace(0.0, 1.0, 60),
            "group": ["a", "b"] * 30,
            "entity_id": entity_ids,
        }
    )
    current = pd.DataFrame(
        {
            "value": np.linspace(0.05, 1.05, 60),
            "group": ["a", "b"] * 30,
            "entity_id": entity_ids,
        }
    )
    config = _config(group_column="entity_id")

    first = adversarial_validate(reference, current, config)
    second = adversarial_validate(reference, current, config)

    assert first == second
    assert first["status"] == "ok"
    assert first["split_strategy"] == "stratified_group_kfold"
    assert first["group_column"] == "entity_id"
    assert first["n_groups"] == 30
    assert "entity_id" not in first["feature_importance"]
    json.dumps(first, allow_nan=False)

    groups, n_groups, reason = _prepare_groups(reference, current, "entity_id")
    assert reason is None
    assert groups is not None
    assert n_groups == 30
    features = pd.concat(
        [reference[["value"]], current[["value"]]],
        ignore_index=True,
    )
    labels = np.concatenate([np.zeros(60, dtype=np.int8), np.ones(60, dtype=np.int8)])
    for train_indices, validation_indices in _split_indices(
        features,
        labels,
        groups=groups,
        n_splits=3,
        seed=42,
    ):
        assert set(groups[train_indices]).isdisjoint(groups[validation_indices])


def test_grouped_split_prevents_repeated_entity_fingerprint_leakage() -> None:
    rng = np.random.default_rng(20260918)
    repeats = 20
    reference_ids = [f"r-{index:02d}" for index in range(40) for _ in range(repeats)]
    current_ids = [f"c-{index:02d}" for index in range(40) for _ in range(repeats)]
    reference = pd.DataFrame(
        {
            "entity_id": reference_ids,
            "fingerprint": reference_ids,
            "value": rng.normal(size=len(reference_ids)),
        }
    )
    current = pd.DataFrame(
        {
            "entity_id": current_ids,
            "fingerprint": current_ids,
            "value": rng.normal(size=len(current_ids)),
        }
    )
    config = {
        "enabled": True,
        "n_splits": 4,
        "roc_auc_threshold": None,
        "exclude_columns": ["entity_id"],
        "random_seed": 42,
        "feature_types": {"fingerprint": "categorical", "value": "numeric"},
    }

    ordinary = adversarial_validate(
        reference,
        current,
        {**config, "group_column": None},
    )
    grouped = adversarial_validate(
        reference,
        current,
        {**config, "group_column": "entity_id"},
    )

    assert ordinary["roc_auc"] == pytest.approx(1.0)
    assert grouped["roc_auc"] == pytest.approx(0.5)
    assert ordinary["split_strategy"] == "stratified_kfold"
    assert grouped["split_strategy"] == "stratified_group_kfold"


@pytest.mark.parametrize(
    ("entity_ids", "reason"),
    [
        (["same"] * 12, "недостаточно групп"),
        ([None, *[f"entity-{index}" for index in range(11)]], "пропущенные"),
    ],
)
def test_impossible_grouped_split_is_skipped(
    entity_ids: list[object],
    reason: str,
) -> None:
    reference = pd.DataFrame(
        {"value": np.arange(12.0), "group": ["a"] * 12, "entity_id": entity_ids}
    )
    current = reference.copy(deep=True)

    result = adversarial_validate(
        reference,
        current,
        _config(group_column="entity_id"),
    )

    assert result["status"] == "skipped"
    assert result["split_strategy"] == "stratified_group_kfold"
    assert reason in str(result["reason"])


def test_missing_group_column_is_skipped() -> None:
    frame = pd.DataFrame({"value": np.arange(12.0), "group": ["a"] * 12})

    result = adversarial_validate(
        frame,
        frame,
        _config(group_column="entity_id"),
    )

    assert result["status"] == "skipped"
    assert "отсутствует" in str(result["reason"])


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
        {"group_column": ""},
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
