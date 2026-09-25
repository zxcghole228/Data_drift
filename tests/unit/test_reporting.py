"""Тесты сопоставимых графиков Reference и Current."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from data_drift_guardian import analyze
from data_drift_guardian.config import load_config
from data_drift_guardian.contracts import AnalysisResult
from data_drift_guardian.reporting import distribution_figure, export_html
from plotly.graph_objects import Figure


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


@pytest.fixture
def report_payload() -> tuple[AnalysisResult, pd.DataFrame, pd.DataFrame]:
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
    result = analyze(reference, current, config=load_config(DEFAULT_CONFIG))
    return result, reference, current


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


def test_export_html_creates_self_contained_report_with_all_sections(
    tmp_path: Path,
    report_payload: tuple[AnalysisResult, pd.DataFrame, pd.DataFrame],
) -> None:
    result, reference, current = report_payload
    output_path = tmp_path / "nested" / "report.html"

    returned_path = export_html(
        result,
        output_path,
        reference=reference,
        current=current,
    )

    content = output_path.read_text(encoding="utf-8")
    assert returned_path == output_path
    assert content.startswith("<!doctype html>")
    assert '<meta charset="utf-8">' in content
    assert "Сводка" in content
    assert "Schema Validation" in content
    assert "Data Quality" in content
    assert "Data Drift" in content
    assert "Adversarial Validation" in content
    assert "Фактически применённая конфигурация" in content
    assert "Распределения" in content
    assert content.count('id="plotly-library"') == 1
    assert "Plotly.newPlot" in content
    assert 'src="https://cdn.plot.ly' not in content
    assert "contract_version" in content


def test_html_report_uses_dark_compact_scrollable_layout(
    tmp_path: Path,
    report_payload: tuple[AnalysisResult, pd.DataFrame, pd.DataFrame],
) -> None:
    result, reference, current = report_payload
    output_path = tmp_path / "dark-report.html"

    export_html(
        result,
        output_path,
        reference=reference,
        current=current,
    )

    content = output_path.read_text(encoding="utf-8")
    assert "color-scheme: dark" in content
    assert 'class="table-scroll"' in content
    assert "Таблица прокручивается по горизонтали" in content
    assert "position: sticky" in content
    assert "<th>Детали</th>" not in content
    assert "<th>Алерт</th>" not in content
    assert 'paper_bgcolor":"#111827"' in content
    assert content.index('id="distributions"') < content.index('id="config"')
    assert '<details class="panel disclosure">' in content


def test_export_html_without_frames_is_complete_and_explains_missing_plots(
    tmp_path: Path,
    report_payload: tuple[AnalysisResult, pd.DataFrame, pd.DataFrame],
) -> None:
    result, _, _ = report_payload
    output_path = tmp_path / "report.html"

    export_html(result, output_path)

    content = output_path.read_text(encoding="utf-8")
    assert "Reference и Current не были переданы" in content
    assert 'id="plotly-library"' not in content
    assert "Plotly.newPlot" not in content
    assert "Проверка пропущена" in content


def test_export_html_escapes_user_strings_in_text_and_plotly(
    tmp_path: Path,
    report_payload: tuple[AnalysisResult, pd.DataFrame, pd.DataFrame],
) -> None:
    result, reference, current = report_payload
    payload = '</script><script>alert("xss")</script>'
    result["alerts"] = [
        {
            "source": "drift",
            "feature": payload,
            "check": "unsafe-check",
            "severity": "warning",
            "message": payload,
        }
    ]
    current.loc[0, "region"] = payload
    output_path = tmp_path / "unsafe.html"

    export_html(
        result,
        output_path,
        reference=reference,
        current=current,
    )

    content = output_path.read_text(encoding="utf-8")
    assert payload not in content
    assert '&lt;/script&gt;&lt;script&gt;alert(&quot;' in content
    assert "unsafe-check" in content


def test_export_html_requires_both_frames_or_neither(
    tmp_path: Path,
    report_payload: tuple[AnalysisResult, pd.DataFrame, pd.DataFrame],
) -> None:
    result, reference, _ = report_payload

    with pytest.raises(ValueError, match="одновременно reference и current"):
        export_html(result, tmp_path / "report.html", reference=reference)

    assert not (tmp_path / "report.html").exists()


def test_export_html_protects_existing_file_and_supports_overwrite(
    tmp_path: Path,
    report_payload: tuple[AnalysisResult, pd.DataFrame, pd.DataFrame],
) -> None:
    result, _, _ = report_payload
    output_path = tmp_path / "report.html"
    output_path.write_text("исходный отчёт", encoding="utf-8")

    with pytest.raises(FileExistsError, match="overwrite=True"):
        export_html(result, output_path)

    assert output_path.read_text(encoding="utf-8") == "исходный отчёт"

    export_html(result, output_path, overwrite=True)

    assert output_path.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_export_html_serialization_error_preserves_existing_file(
    tmp_path: Path,
    report_payload: tuple[AnalysisResult, pd.DataFrame, pd.DataFrame],
) -> None:
    result, _, _ = report_payload
    invalid_result = copy.deepcopy(result)
    invalid_result["effective_config"]["random_seed"] = float("nan")  # type: ignore[typeddict-item]
    output_path = tmp_path / "report.html"
    output_path.write_text("не заменять", encoding="utf-8")

    with pytest.raises(ValueError):
        export_html(invalid_result, output_path, overwrite=True)

    assert output_path.read_text(encoding="utf-8") == "не заменять"
    assert not list(tmp_path.glob(".report.html.*.tmp"))


def test_export_html_preserves_input_frames(
    tmp_path: Path,
    report_payload: tuple[AnalysisResult, pd.DataFrame, pd.DataFrame],
) -> None:
    result, reference, current = report_payload
    original_reference = reference.copy(deep=True)
    original_current = current.copy(deep=True)

    export_html(
        result,
        tmp_path / "report.html",
        reference=reference,
        current=current,
    )

    pd.testing.assert_frame_equal(reference, original_reference)
    pd.testing.assert_frame_equal(current, original_current)
