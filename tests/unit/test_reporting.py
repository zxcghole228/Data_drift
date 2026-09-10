"""Тесты сопоставимых графиков Reference и Current."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from data_drift_guardian.reporting import distribution_figure
from plotly.graph_objects import Figure


def test_numeric_figure_uses_common_bins_and_normalized_fractions() -> None:
    reference = pd.Series([0.0, 1.0, 2.0, np.nan, np.inf], name="age")
    current = pd.Series([0.0, 2.0, 3.0, 3.0], name="age")

    figure = distribution_figure(reference, current, "numeric")

    assert isinstance(figure, Figure)
    assert len(figure.data) == 2
    assert list(figure.data[0].x) == pytest.approx(list(figure.data[1].x))
    assert list(figure.data[0].width) == pytest.approx(list(figure.data[1].width))
    assert sum(figure.data[0].y) == pytest.approx(1.0)
    assert sum(figure.data[1].y) == pytest.approx(1.0)
    assert figure.layout.barmode == "overlay"
    note = figure.layout.annotations[-1].text
    assert "Reference — 1" in note
    assert "бесконечностей: Reference — 1" in note


def test_numeric_figure_handles_constant_and_empty_sample() -> None:
    reference = pd.Series([5.0, 5.0], name="score")
    current = pd.Series([np.nan, np.inf], name="score")

    figure = distribution_figure(reference, current, "numeric")

    assert len(figure.data) == 2
    assert len(figure.data[0].x) == 1
    assert sum(figure.data[0].y) == pytest.approx(1.0)
    assert sum(figure.data[1].y) == pytest.approx(0.0)


def test_numeric_figure_explains_when_both_samples_have_no_finite_values() -> None:
    reference = pd.Series([pd.NA, pd.NA], dtype="Float64", name="income")
    current = pd.Series([np.nan, -np.inf], name="income")

    figure = distribution_figure(reference, current, "numeric")

    assert len(figure.data) == 0
    assert any(
        "Нет конечных значений" in annotation.text
        for annotation in figure.layout.annotations
    )


def test_categorical_figure_uses_union_and_per_sample_proportions() -> None:
    reference = pd.Series(["north", "north", "south", pd.NA], name="region")
    current = pd.Series(["south", "east", "east", pd.NA], name="region")

    figure = distribution_figure(reference, current, "categorical")

    assert len(figure.data) == 2
    assert list(figure.data[0].x) == list(figure.data[1].x)
    reference_shares = dict(zip(figure.data[0].x, figure.data[0].y, strict=True))
    current_shares = dict(zip(figure.data[1].x, figure.data[1].y, strict=True))
    assert reference_shares == pytest.approx(
        {"east": 0.0, "north": 2 / 3, "south": 1 / 3}
    )
    assert current_shares == pytest.approx(
        {"east": 2 / 3, "north": 0.0, "south": 1 / 3}
    )
    assert figure.layout.barmode == "group"
    assert "Reference — 1" in figure.layout.annotations[-1].text


def test_distribution_figure_does_not_change_input_series() -> None:
    reference = pd.Series([1.0, np.nan, np.inf], name="value")
    current = pd.Series([2.0, 3.0], name="value")
    original_reference = reference.copy(deep=True)
    original_current = current.copy(deep=True)

    distribution_figure(reference, current, "numeric")

    pd.testing.assert_series_equal(reference, original_reference)
    pd.testing.assert_series_equal(current, original_current)


@pytest.mark.parametrize("invalid", ["number", "category", "", None])
def test_unknown_feature_type_raises_value_error(invalid: object) -> None:
    series = pd.Series([1, 2, 3])

    with pytest.raises(ValueError, match="feature_type"):
        distribution_figure(series, series, invalid)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("reference", "current", "message"),
    [
        ([1, 2], pd.Series([1, 2]), "reference"),
        (pd.Series([1, 2]), [1, 2], "current"),
    ],
)
def test_non_series_input_raises_type_error(
    reference: object,
    current: object,
    message: str,
) -> None:
    with pytest.raises(TypeError, match=message):
        distribution_figure(reference, current, "numeric")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "series",
    [
        pd.Series(["1", "2"], dtype="string"),
        pd.Series([True, False], dtype="boolean"),
    ],
)
def test_numeric_figure_rejects_non_numeric_dtype(series: pd.Series) -> None:
    with pytest.raises(TypeError, match="числовой dtype"):
        distribution_figure(series, series, "numeric")
