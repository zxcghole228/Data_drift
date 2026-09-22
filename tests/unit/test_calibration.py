"""Tests for the reproducible calibration runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

from data_drift_guardian.contracts import AnalysisConfig, AnalysisResult, CheckResult
from data_drift_guardian.config import load_config
from scripts import run_calibration as calibration


def _check(
    name: str,
    *,
    alert: bool | None,
    value: float | None = 0.1,
    threshold: float | None = 0.05,
    details: dict[str, Any] | None = None,
) -> CheckResult:
    return {
        "name": name,
        "status": "warning" if alert else "ok",
        "value": value,
        "threshold": threshold,
        "p_value": 0.01,
        "adjusted_p_value": 0.02,
        "alert": alert,
        "reason": "test alert" if alert else None,
        "details": {} if details is None else details,
    }


def _analysis_result() -> AnalysisResult:
    """Return the minimal complete result shape used by the flattener."""

    return cast(
        AnalysisResult,
        {
            "contract_version": "0.2",
            "effective_config": {},
            "metadata": {
                "reference_rows": 40,
                "current_rows": 30,
                "random_seed": 42,
            },
            "schema": {
                "status": "ok",
                "missing_columns": {"reference": [], "current": []},
                "extra_columns": {"reference": [], "current": []},
                "duplicate_columns": {"reference": [], "current": []},
                "type_mismatches": [],
                "valid_features": ["age", "income", "region"],
                "reason": None,
            },
            "quality": {
                "status": "ok",
                "dataset_checks": [
                    _check("current_duplicate_fraction", alert=False)
                ],
                "feature_checks": {
                    "income": [_check("missing_increase_pp", alert=True)]
                },
                "reason": None,
            },
            "drift": {
                "status": "warning",
                "features": {
                    "age": {
                        "feature_type": "numeric",
                        "status": "warning",
                        "n_reference_valid": 40,
                        "n_current_valid": 30,
                        "checks": {
                            "psi": _check(
                                "psi",
                                alert=True,
                                details={"threshold_source": "global"},
                            )
                        },
                        "alert": True,
                        "reason": None,
                    },
                    "region": {
                        "feature_type": "categorical",
                        "status": "warning",
                        "n_reference_valid": 40,
                        "n_current_valid": 30,
                        "checks": {
                            "chi2": _check(
                                "chi2",
                                alert=True,
                                details={"cramers_v": 0.42},
                            )
                        },
                        "alert": True,
                        "reason": None,
                    },
                },
                "reason": None,
            },
            "adversarial": {
                "status": "warning",
                "roc_auc": 0.8,
                "threshold": 0.7,
                "fold_auc": [0.78, 0.81, 0.82],
                "feature_importance": {"age": 1.0},
                "importance_type": "gain",
                "split_strategy": "stratified_kfold",
                "group_column": None,
                "n_groups": None,
                "alert": True,
                "reason": "test alert",
            },
            "alerts": [],
            "summary": {
                "status": "warning",
                "has_alerts": True,
                "n_alerts": 4,
                "analyzed_features": 3,
                "skipped_features": 0,
            },
        },
    )


def test_default_grid_has_twenty_controls_for_every_current_size() -> None:
    cases = calibration.build_cases()

    assert len(calibration.DEFAULT_SEEDS) == 20
    assert len(cases) == 27
    assert len({case.case_id for case in cases}) == len(cases)
    for n_current in calibration.DEFAULT_CURRENT_SIZES:
        controls = [
            case
            for case in cases
            if case.scenario == "none" and case.n_current == n_current
        ]
        assert len(controls) * len(calibration.DEFAULT_SEEDS) == 20


def test_build_cases_rejects_invalid_grid_values() -> None:
    with pytest.raises(ValueError, match="current_sizes"):
        calibration.build_cases(current_sizes=(0,))
    with pytest.raises(ValueError, match="missing_fractions"):
        calibration.build_cases(missing_fractions=(0.01,))
    with pytest.raises(ValueError, match="acceptance_age_shift"):
        calibration.build_cases(acceptance_age_shift=0.0)


def test_demo_calibrated_config_has_feature_specific_wasserstein_threshold() -> None:
    config = load_config("configs/demo_calibrated.yaml")

    assert config["drift"]["distance_thresholds"] == {
        "wasserstein": None,
        "psi": 0.10,
        "js": 0.05,
    }
    assert config["drift"]["feature_thresholds"]["age"]["wasserstein"] == 3.0
    assert config["drift"]["feature_thresholds"]["income"]["wasserstein"] is None
    assert config["adversarial"]["enabled"] is True


def test_result_rows_keep_diagnostics_and_expected_signal_labels() -> None:
    case = calibration.CalibrationCase(
        case_id="combined_acceptance_n30",
        scenario="combined",
        drift_family="combined",
        drift_level="acceptance",
        n_current=30,
        age_shift_years=8.0,
        current_missing_fraction=0.08,
    )

    rows = calibration.result_rows(
        _analysis_result(),
        case=case,
        seed=17,
        n_reference=40,
        reference_missing_fraction=0.02,
        elapsed_seconds=0.25,
        commit_sha="abc123",
        config_sha256="def456",
    )
    by_method = {(row["feature"], row["method"]): row for row in rows}

    assert by_method[("age", "psi")]["threshold_source"] == "global"
    assert by_method[("region", "chi2")]["cramers_v"] == pytest.approx(0.42)
    assert by_method[(None, "roc_auc")]["split_strategy"] == "stratified_kfold"
    assert by_method[("income", "missing_increase_pp")]["expected_signal"] is True
    assert by_method[(None, "current_duplicate_fraction")]["expected_signal"] is False
    assert {row["commit_sha"] for row in rows} == {"abc123"}
    assert {row["config_sha256"] for row in rows} == {"def456"}


def test_run_calibration_is_reproducible_except_for_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int]] = []

    def fake_analyze(
        reference: pd.DataFrame,
        current: pd.DataFrame,
        config: AnalysisConfig,
    ) -> AnalysisResult:
        calls.append((len(reference), len(current)))
        return _analysis_result()

    monkeypatch.setattr(calibration, "analyze", fake_analyze)
    case = calibration.CalibrationCase(
        case_id="control_n30",
        scenario="none",
        drift_family="control",
        drift_level="control",
        n_current=30,
        age_shift_years=0.0,
        current_missing_fraction=0.02,
    )
    arguments = {
        "cases": [case],
        "seeds": [3, 7],
        "n_reference": 40,
        "reference_missing_fraction": 0.02,
        "config": cast(AnalysisConfig, {}),
        "commit_sha": "abc123",
        "config_sha256": "def456",
    }

    first = calibration.run_calibration(**arguments)
    second = calibration.run_calibration(**arguments)

    stable_columns = [
        column for column in first.columns if column != "elapsed_seconds"
    ]
    pd.testing.assert_frame_equal(first[stable_columns], second[stable_columns])
    assert calls == [(40, 30), (40, 30), (40, 30), (40, 30)]
    assert first["seed"].nunique() == 2
    assert not first["expected_signal"].any()


def test_summary_separates_false_detection_and_non_target_rates() -> None:
    common = {
        "drift_family": "numeric",
        "drift_level": "weak",
        "n_current": 500,
        "age_shift_years": 2.0,
        "current_missing_fraction": 0.02,
        "feature": "age",
        "commit_sha": "abc123",
        "config_sha256": "def456",
    }
    runs = pd.DataFrame(
        [
            {
                **common,
                "case_id": "control_n500",
                "scenario": "none",
                "source": "drift",
                "method": "psi",
                "expected_signal": False,
                "seed": 1,
                "alert": False,
                "value": 0.01,
            },
            {
                **common,
                "case_id": "control_n500",
                "scenario": "none",
                "source": "drift",
                "method": "psi",
                "expected_signal": False,
                "seed": 2,
                "alert": True,
                "value": 0.2,
            },
            {
                **common,
                "case_id": "numeric_weak_n500",
                "scenario": "numeric",
                "source": "drift",
                "method": "psi",
                "expected_signal": True,
                "seed": 1,
                "alert": True,
                "value": 0.2,
            },
            {
                **common,
                "case_id": "numeric_weak_n500",
                "scenario": "numeric",
                "source": "drift",
                "method": "psi",
                "expected_signal": True,
                "seed": 2,
                "alert": True,
                "value": 0.3,
            },
            {
                **common,
                "case_id": "numeric_weak_n500",
                "scenario": "numeric",
                "source": "quality",
                "method": "current_missing_fraction",
                "expected_signal": False,
                "seed": 1,
                "alert": False,
                "value": 0.02,
            },
            {
                **common,
                "case_id": "numeric_weak_n500",
                "scenario": "numeric",
                "source": "quality",
                "method": "current_missing_fraction",
                "expected_signal": False,
                "seed": 2,
                "alert": False,
                "value": 0.02,
            },
        ]
    )

    summary = calibration.summarize_runs(runs)
    rates = dict(zip(summary["rate_kind"], summary["observed_rate"], strict=True))

    assert rates == {
        "false_alert_rate": pytest.approx(0.5),
        "detection_rate": pytest.approx(1.0),
        "non_target_alert_rate": pytest.approx(0.0),
    }
    assert set(summary["seeds"]) == {"1|2"}


def test_writer_creates_strict_json_and_protects_existing_files(
    tmp_path: Path,
) -> None:
    runs = pd.DataFrame([{"case_id": "control", "value": 0.1}])
    summary = pd.DataFrame([{"case_id": "control", "observed_rate": 0.0}])
    metadata = {"commit_sha": "abc123", "elapsed_seconds": 1.25}

    paths = calibration.write_calibration_outputs(
        output_dir=tmp_path,
        runs=runs,
        summary=summary,
        metadata=metadata,
    )

    assert set(paths) == {"runs", "summary", "metadata"}
    assert all(path.is_file() for path in paths.values())
    assert json.loads(paths["metadata"].read_text(encoding="utf-8")) == metadata
    assert not list(tmp_path.glob("*.tmp"))
    with pytest.raises(FileExistsError, match="--overwrite"):
        calibration.write_calibration_outputs(
            output_dir=tmp_path,
            runs=runs,
            summary=summary,
            metadata=metadata,
        )
