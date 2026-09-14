"""End-to-end checks from persisted tables to the JSON-safe result."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from data_drift_guardian import analyze
from data_drift_guardian.config import load_config
from data_drift_guardian.ingestion import load_table
from scripts.generate_demo_data import generate_demo_data, write_generated_data


@pytest.mark.parametrize("file_format", ["csv", "parquet"])
def test_generated_files_reach_real_statistics_and_adversarial_result(
    tmp_path: Path,
    file_format: str,
) -> None:
    generated = generate_demo_data(
        seed=42,
        n_reference=300,
        n_current=180,
        scenario="combined",
    )
    paths = write_generated_data(
        generated,
        output_dir=tmp_path,
        formats=(file_format,),  # type: ignore[arg-type]
    )
    reference = load_table(tmp_path / paths[f"reference_{file_format}"])
    current = load_table(tmp_path / paths[f"current_{file_format}"])
    config = load_config("configs/default.yaml")
    config["adversarial"].update({"enabled": True, "roc_auc_threshold": 0.7})

    result = analyze(reference, current, config)

    assert result["schema"]["status"] == "ok"
    assert result["drift"]["features"]["age"]["checks"]["ks"]["value"] is not None
    assert (
        result["drift"]["features"]["region"]["checks"]["chi2"]["p_value"] is not None
    )
    assert result["adversarial"]["roc_auc"] is not None
    json.dumps(result, allow_nan=False)


def test_loaded_schema_error_remains_visible_while_valid_features_continue(
    tmp_path: Path,
) -> None:
    reference_path = tmp_path / "reference.csv"
    current_path = tmp_path / "current.csv"
    pd.DataFrame(
        {"age": [20.0, 30.0], "income": [100.0, 200.0], "region": ["a", "b"]}
    ).to_csv(reference_path, index=False)
    pd.DataFrame({"age": [21.0, 31.0], "income": [110.0, 210.0]}).to_csv(
        current_path, index=False
    )

    result = analyze(
        load_table(reference_path),
        load_table(current_path),
        load_config("configs/default.yaml"),
    )

    assert result["schema"]["status"] == "error"
    assert result["drift"]["features"]["region"]["status"] == "skipped"
    assert result["drift"]["features"]["age"]["checks"]["ks"]["status"] == "ok"
    assert any(alert["check"] == "missing_column" for alert in result["alerts"])


def test_header_only_current_file_returns_partial_result_instead_of_crashing(
    tmp_path: Path,
) -> None:
    reference_path = tmp_path / "reference.csv"
    current_path = tmp_path / "current.csv"
    pd.DataFrame({"age": [20.0], "income": [100.0], "region": ["a"]}).to_csv(
        reference_path, index=False
    )
    current_path.write_text("age,income,region\n", encoding="utf-8")

    result = analyze(
        load_table(reference_path),
        load_table(current_path),
        load_config("configs/default.yaml"),
    )

    assert result["schema"]["status"] == "error"
    assert result["summary"]["status"] == "error"
    assert result["adversarial"]["status"] == "skipped"
    assert any(alert["check"] == "empty_dataset" for alert in result["alerts"])
    json.dumps(result, allow_nan=False)


def test_constant_and_all_missing_columns_are_explicit_in_common_path(
    tmp_path: Path,
) -> None:
    reference = pd.DataFrame(
        {
            "constant": [5.0] * 20,
            "empty": [np.nan] * 20,
            "category": ["a", "b"] * 10,
        }
    )
    current = reference.copy(deep=True)
    reference_path = tmp_path / "reference.parquet"
    current_path = tmp_path / "current.parquet"
    reference.to_parquet(reference_path, index=False)
    current.to_parquet(current_path, index=False)
    config: dict[str, Any] = {
        "contract_version": "0.1",
        "random_seed": 42,
        "schema": {
            "allow_extra_columns": False,
            "features": {
                "constant": {"kind": "numeric", "nullable": True},
                "empty": {"kind": "numeric", "nullable": True},
                "category": {"kind": "categorical", "nullable": True},
            },
        },
        "quality": {
            "max_missing_fraction": 1.0,
            "max_missing_increase_pp": 100.0,
            "max_duplicate_fraction": 1.0,
        },
        "drift": {
            "numeric_methods": ["ks", "wasserstein", "psi", "js"],
            "categorical_methods": ["chi2", "psi", "js"],
            "alpha": 0.05,
            "multiple_testing": "bh",
            "n_bins": 5,
            "psi_smoothing": 0.000001,
            "js_base": 2,
            "distance_thresholds": {
                "wasserstein": None,
                "psi": None,
                "js": None,
            },
        },
        "adversarial": {
            "enabled": True,
            "n_splits": 3,
            "roc_auc_threshold": None,
            "exclude_columns": [],
        },
    }

    result = analyze(load_table(reference_path), load_table(current_path), config)

    constant = result["drift"]["features"]["constant"]
    empty = result["drift"]["features"]["empty"]
    assert constant["checks"]["ks"]["status"] == "ok"
    assert constant["checks"]["wasserstein"]["value"] == pytest.approx(0.0)
    assert empty["status"] == "skipped"
    assert all(check["status"] == "skipped" for check in empty["checks"].values())
    assert result["adversarial"]["status"] == "ok"
    json.dumps(result, allow_nan=False)
