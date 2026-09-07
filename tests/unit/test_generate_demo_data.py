from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts.generate_demo_data import (
    DEFAULT_AGE_SHIFT_YEARS,
    generate_demo_data,
    write_generated_data,
)


def test_same_seed_reproduces_independent_batches() -> None:
    first = generate_demo_data(seed=17, n_reference=200, n_current=200)
    second = generate_demo_data(seed=17, n_reference=200, n_current=200)

    pd.testing.assert_frame_equal(first.reference, second.reference)
    pd.testing.assert_frame_equal(first.current, second.current)
    assert first.metadata == second.metadata
    assert not first.reference.equals(first.current)


@pytest.mark.parametrize(
    ("scenario", "expected_reference_missing", "expected_current_missing"),
    [
        ("none", 0.02, 0.02),
        ("missingness", 0.02, 0.08),
        ("combined", 0.02, 0.08),
    ],
)
def test_income_missingness_is_exact(
    scenario: str,
    expected_reference_missing: float,
    expected_current_missing: float,
) -> None:
    generated = generate_demo_data(
        seed=42,
        n_reference=1_000,
        n_current=1_000,
        scenario=scenario,  # type: ignore[arg-type]
    )

    assert generated.reference["income"].isna().mean() == pytest.approx(
        expected_reference_missing
    )
    assert generated.current["income"].isna().mean() == pytest.approx(
        expected_current_missing
    )


def test_numeric_scenario_changes_only_current_age() -> None:
    control = generate_demo_data(
        seed=101, n_reference=300, n_current=300, scenario="none"
    )
    shifted = generate_demo_data(
        seed=101, n_reference=300, n_current=300, scenario="numeric"
    )

    pd.testing.assert_frame_equal(control.reference, shifted.reference)
    pd.testing.assert_series_equal(control.current["income"], shifted.current["income"])
    pd.testing.assert_series_equal(control.current["region"], shifted.current["region"])
    expected_age = (control.current["age"] + DEFAULT_AGE_SHIFT_YEARS).clip(18, 100)
    pd.testing.assert_series_equal(shifted.current["age"], expected_age)


def test_categorical_scenario_preserves_numeric_features() -> None:
    control = generate_demo_data(
        seed=2026, n_reference=400, n_current=400, scenario="none"
    )
    shifted = generate_demo_data(
        seed=2026, n_reference=400, n_current=400, scenario="categorical"
    )

    pd.testing.assert_frame_equal(control.reference, shifted.reference)
    pd.testing.assert_series_equal(control.current["age"], shifted.current["age"])
    pd.testing.assert_series_equal(control.current["income"], shifted.current["income"])
    assert not control.current["region"].equals(shifted.current["region"])


def test_values_and_dtypes_match_demo_contract() -> None:
    generated = generate_demo_data(
        seed=11, n_reference=250, n_current=150, scenario="combined"
    )

    for frame in (generated.reference, generated.current):
        assert list(frame.columns) == ["age", "income", "region"]
        assert frame["age"].dtype == np.dtype("float64")
        assert frame["income"].dtype == np.dtype("float64")
        assert frame["region"].dtype == np.dtype("object")
        assert frame["age"].between(18, 100).all()
        assert frame["income"].dropna().between(0, 1_000_000).all()
        assert set(frame["region"].dropna().unique()) <= {
            "central",
            "northwest",
            "south",
            "volga",
            "siberia",
        }


def test_writer_creates_csv_parquet_and_metadata(tmp_path) -> None:
    generated = generate_demo_data(
        seed=7, n_reference=30, n_current=20, scenario="combined"
    )
    paths = write_generated_data(
        generated,
        output_dir=tmp_path,
        formats=("csv", "parquet"),
    )

    assert set(paths) == {
        "reference_csv",
        "current_csv",
        "reference_parquet",
        "current_parquet",
        "metadata",
    }
    for relative_path in paths.values():
        assert (tmp_path / relative_path).is_file()

    csv_reference = pd.read_csv(tmp_path / paths["reference_csv"])
    parquet_reference = pd.read_parquet(tmp_path / paths["reference_parquet"])
    assert csv_reference.shape == generated.reference.shape
    assert parquet_reference.shape == generated.reference.shape
    assert list(csv_reference.columns) == list(generated.reference.columns)
    assert list(parquet_reference.columns) == list(generated.reference.columns)

    metadata = json.loads((tmp_path / paths["metadata"]).read_text(encoding="utf-8"))
    assert metadata["scenario"] == "combined"
    assert metadata["seed"] == 7
    assert metadata["files"]["reference_csv"] == paths["reference_csv"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_reference": 0},
        {"n_current": -1},
        {"reference_income_missing_fraction": -0.01},
        {"drifted_income_missing_fraction": 1.01},
        {"scenario": "unknown"},
    ],
)
def test_invalid_generation_parameters_raise_value_error(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        generate_demo_data(**kwargs)  # type: ignore[arg-type]
