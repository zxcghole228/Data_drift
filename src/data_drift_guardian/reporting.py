"""Графики распределений и экспорт отчёта."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from .contracts import AnalysisResult, FeatureType


REFERENCE_COLOR = "#2563EB"
CURRENT_COLOR = "#F97316"
MAX_NUMERIC_BINS = 30


def distribution_figure(
    reference: pd.Series,
    current: pd.Series,
    feature_type: FeatureType,
) -> go.Figure:
    """Построить сопоставимые распределения Reference и Current.

    Для числового признака функция использует общие границы интервалов и
    показывает долю конечных значений каждого набора в интервале. Для
    категориального признака показывается доля каждой категории среди
    непустых значений соответствующего набора.

    Пропуски не включаются в распределения. Положительная и отрицательная
    бесконечность дополнительно исключаются из числового графика. Количество
    исключённых значений всегда указано под графиком.

    Args:
        reference: Значения признака из эталонной выборки.
        current: Значения признака из текущей выборки.
        feature_type: Семантический тип ``numeric`` или ``categorical``.

    Returns:
        Готовая фигура Plotly, не изменяющая входные Series.

    Raises:
        TypeError: Если входы не являются ``pandas.Series`` или числовой
            признак имеет нечисловой dtype.
        ValueError: Если передан неизвестный семантический тип.
    """
    if not isinstance(reference, pd.Series):
        raise TypeError(
            "reference должен иметь тип pandas.Series, "
            f"получен {type(reference).__name__}"
        )
    if not isinstance(current, pd.Series):
        raise TypeError(
            "current должен иметь тип pandas.Series, "
            f"получен {type(current).__name__}"
        )
    if feature_type not in {"numeric", "categorical"}:
        raise ValueError(
            "feature_type должен быть 'numeric' или 'categorical', "
            f"получено {feature_type!r}"
        )

    feature_name = str(reference.name or current.name or "признак")
    if feature_type == "numeric":
        return _numeric_figure(reference, current, feature_name)
    return _categorical_figure(reference, current, feature_name)


def _numeric_values(series: pd.Series, dataset: str) -> tuple[np.ndarray, int, int]:
    if not is_numeric_dtype(series.dtype) or is_bool_dtype(series.dtype):
        raise TypeError(
            f"Для numeric-графика {dataset} должен иметь числовой dtype, "
            f"получен {series.dtype}"
        )

    without_missing = series.dropna().to_numpy()
    if np.iscomplexobj(without_missing):
        raise TypeError(
            f"Для numeric-графика {dataset} не может иметь комплексный dtype"
        )

    values = without_missing.astype(float, copy=False)
    finite_mask = np.isfinite(values)
    missing_count = int(series.isna().sum())
    infinite_count = int((~finite_mask).sum())
    return values[finite_mask], missing_count, infinite_count


def _numeric_edges(reference: np.ndarray, current: np.ndarray) -> np.ndarray:
    combined = np.concatenate((reference, current))
    minimum = float(combined.min())
    maximum = float(combined.max())

    if minimum == maximum:
        padding = max(abs(minimum) * 0.05, 0.5)
        return np.array([minimum - padding, maximum + padding], dtype=float)

    automatic = np.histogram_bin_edges(combined, bins="auto")
    n_bins = min(max(len(automatic) - 1, 1), MAX_NUMERIC_BINS)
    return np.linspace(minimum, maximum, n_bins + 1)


def _numeric_trace(
    values: np.ndarray,
    edges: np.ndarray,
    *,
    name: str,
    color: str,
) -> go.Bar:
    counts, _ = np.histogram(values, bins=edges)
    fractions = counts / len(values) if len(values) else counts.astype(float)
    centers = (edges[:-1] + edges[1:]) / 2
    widths = np.diff(edges)
    customdata = np.column_stack((edges[:-1], edges[1:], counts))

    return go.Bar(
        name=name,
        x=centers,
        y=fractions,
        width=widths,
        marker_color=color,
        opacity=0.55,
        customdata=customdata,
        hovertemplate=(
            "Интервал: [%{customdata[0]:.4g}; %{customdata[1]:.4g}]<br>"
            "Доля: %{y:.2%}<br>"
            "Наблюдений: %{customdata[2]:.0f}<extra>%{fullData.name}</extra>"
        ),
    )


def _numeric_figure(
    reference: pd.Series,
    current: pd.Series,
    feature_name: str,
) -> go.Figure:
    reference_values, reference_missing, reference_infinite = _numeric_values(
        reference, "reference"
    )
    current_values, current_missing, current_infinite = _numeric_values(
        current, "current"
    )

    figure = go.Figure()
    if len(reference_values) or len(current_values):
        edges = _numeric_edges(reference_values, current_values)
        figure.add_trace(
            _numeric_trace(
                reference_values,
                edges,
                name="Reference",
                color=REFERENCE_COLOR,
            )
        )
        figure.add_trace(
            _numeric_trace(
                current_values,
                edges,
                name="Current",
                color=CURRENT_COLOR,
            )
        )
    else:
        figure.add_annotation(
            text="Нет конечных значений для построения распределения",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )

    _apply_common_layout(
        figure,
        title=f"Числовое распределение: {feature_name}",
        x_title=feature_name,
        barmode="overlay",
    )
    _add_exclusion_note(
        figure,
        (
            "Доли рассчитаны среди конечных значений каждого набора. "
            f"Исключено пропусков: Reference — {reference_missing}, "
            f"Current — {current_missing}; бесконечностей: "
            f"Reference — {reference_infinite}, Current — {current_infinite}."
        ),
    )
    return figure


def _category_counts(series: pd.Series) -> tuple[dict[tuple[str, str], int], int]:
    counts: dict[tuple[str, str], int] = {}
    for value in series.dropna().tolist():
        key = (type(value).__name__, repr(value))
        counts[key] = counts.get(key, 0) + 1
    return counts, int(series.isna().sum())


def _category_labels(keys: list[tuple[str, str]]) -> list[str]:
    plain_labels = [
        str(ast.literal_eval(representation))
        if type_name == "str"
        else representation
        for type_name, representation in keys
    ]
    duplicates = {label for label in plain_labels if plain_labels.count(label) > 1}
    return [
        f"{label} ({key[0]})" if label in duplicates else label
        for key, label in zip(keys, plain_labels, strict=True)
    ]


def _categorical_figure(
    reference: pd.Series,
    current: pd.Series,
    feature_name: str,
) -> go.Figure:
    reference_counts, reference_missing = _category_counts(reference)
    current_counts, current_missing = _category_counts(current)
    keys = sorted(
        reference_counts.keys() | current_counts.keys(),
        key=lambda key: (
            -(reference_counts.get(key, 0) + current_counts.get(key, 0)),
            key,
        ),
    )
    labels = _category_labels(keys)
    reference_total = sum(reference_counts.values())
    current_total = sum(current_counts.values())

    def proportions(counts: dict[tuple[str, str], int], total: int) -> list[float]:
        if total == 0:
            return [0.0 for _ in keys]
        return [counts.get(key, 0) / total for key in keys]

    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            name="Reference",
            x=labels,
            y=proportions(reference_counts, reference_total),
            marker_color=REFERENCE_COLOR,
            customdata=[reference_counts.get(key, 0) for key in keys],
            hovertemplate=(
                "Категория: %{x}<br>Доля: %{y:.2%}<br>"
                "Наблюдений: %{customdata}<extra>%{fullData.name}</extra>"
            ),
        )
    )
    figure.add_trace(
        go.Bar(
            name="Current",
            x=labels,
            y=proportions(current_counts, current_total),
            marker_color=CURRENT_COLOR,
            customdata=[current_counts.get(key, 0) for key in keys],
            hovertemplate=(
                "Категория: %{x}<br>Доля: %{y:.2%}<br>"
                "Наблюдений: %{customdata}<extra>%{fullData.name}</extra>"
            ),
        )
    )
    if not keys:
        figure.add_annotation(
            text="Нет непустых категорий для построения распределения",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )

    _apply_common_layout(
        figure,
        title=f"Распределение категорий: {feature_name}",
        x_title=feature_name,
        barmode="group",
    )
    _add_exclusion_note(
        figure,
        (
            "Доли рассчитаны среди непустых значений каждого набора. "
            f"Исключено пропусков: Reference — {reference_missing}, "
            f"Current — {current_missing}."
        ),
    )
    return figure


def _apply_common_layout(
    figure: go.Figure,
    *,
    title: str,
    x_title: str,
    barmode: str,
) -> None:
    figure.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title="Доля наблюдений",
        yaxis_tickformat=".1%",
        barmode=barmode,
        legend_title_text="Выборка",
        margin={"b": 105, "l": 55, "r": 25, "t": 60},
    )


def _add_exclusion_note(figure: go.Figure, text: str) -> None:
    figure.add_annotation(
        text=text,
        x=0,
        y=-0.24,
        xref="paper",
        yref="paper",
        xanchor="left",
        yanchor="top",
        align="left",
        showarrow=False,
        font={"size": 11, "color": "#475569"},
    )


def export_html(result: AnalysisResult, output_path: str | Path) -> Path:
    """Экспорт отчёта — итоговое требование; реализация пока не запланирована по дням."""
    raise NotImplementedError("HTML-экспорт отмечен в docs/deliverables.md")
