"""Графики распределений и экспорт отчёта."""

from __future__ import annotations

import ast
import html
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from plotly.io import to_html
from plotly.offline import get_plotlyjs

from .contracts import AnalysisResult, FeatureType


REFERENCE_COLOR = "#2563EB"
CURRENT_COLOR = "#F97316"
MAX_NUMERIC_BINS = 30

STATUS_LABELS = {
    "ok": "Успешно",
    "warning": "Предупреждение",
    "critical": "Критическая проблема",
    "skipped": "Проверка пропущена",
    "error": "Ошибка",
}

REPORT_STYLES = """
:root {
  color-scheme: dark;
  --background: #070b14;
  --surface: #111827;
  --surface-raised: #172033;
  --surface-hover: #1e293b;
  --border: #334155;
  --text: #e5e7eb;
  --muted: #94a3b8;
  --ok: #86efac;
  --ok-bg: #14532d;
  --warning: #fde68a;
  --warning-bg: #78350f;
  --critical: #fecaca;
  --critical-bg: #7f1d1d;
  --skipped: #cbd5e1;
  --skipped-bg: #334155;
  --error: #fecaca;
  --error-bg: #7f1d1d;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background:
    radial-gradient(circle at top, #172033 0, var(--background) 36rem),
    var(--background);
  color: var(--text);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
    "Segoe UI", sans-serif;
  line-height: 1.5;
}
main { width: min(1500px, calc(100% - 32px)); margin: 0 auto; padding: 32px 0 64px; }
h1, h2, h3 { line-height: 1.2; }
h1 { margin-bottom: 8px; }
h2 { margin-top: 36px; border-bottom: 1px solid var(--border); padding-bottom: 8px; }
.muted { color: var(--muted); }
.panel {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  margin: 16px 0;
  padding: 20px;
  overflow: hidden;
  box-shadow: 0 14px 36px rgb(0 0 0 / 18%);
}
.summary-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
  gap: 12px;
}
.summary-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 16px;
}
.summary-card strong { display: block; font-size: 1.35rem; margin-top: 4px; }
.badge {
  border-radius: 999px;
  display: inline-block;
  font-weight: 700;
  padding: 4px 10px;
}
.status-ok { color: var(--ok); background: var(--ok-bg); }
.status-warning { color: var(--warning); background: var(--warning-bg); }
.status-critical { color: var(--critical); background: var(--critical-bg); }
.status-skipped { color: var(--skipped); background: var(--skipped-bg); }
.status-error { color: var(--error); background: var(--error-bg); }
.table-shell {
  border: 1px solid var(--border);
  border-radius: 10px;
  margin: 12px 0 20px;
  overflow: hidden;
}
.table-hint {
  background: var(--surface-raised);
  border-bottom: 1px solid var(--border);
  color: var(--muted);
  font-size: 0.82rem;
  padding: 7px 10px;
}
.table-scroll {
  max-width: 100%;
  overflow-x: auto;
  overscroll-behavior-inline: contain;
  scrollbar-color: #64748b var(--surface-raised);
  scrollbar-width: thin;
}
.table-scroll:focus { outline: 2px solid #60a5fa; outline-offset: -2px; }
table {
  border-collapse: separate;
  border-spacing: 0;
  font-size: 0.9rem;
  min-width: 100%;
  width: max-content;
}
th, td {
  border-bottom: 1px solid var(--border);
  max-width: 360px;
  overflow-wrap: anywhere;
  padding: 9px 11px;
  text-align: left;
  vertical-align: top;
}
th {
  background: var(--surface-raised);
  color: #f8fafc;
  position: sticky;
  top: 0;
  white-space: nowrap;
  z-index: 2;
}
th:first-child { left: 0; z-index: 4; }
td:first-child {
  background: var(--surface);
  box-shadow: 1px 0 0 var(--border);
  left: 0;
  position: sticky;
  z-index: 1;
}
tbody tr:last-child td { border-bottom: 0; }
tbody tr:hover td { background: var(--surface-hover); }
tbody tr:hover td:first-child { background: var(--surface-hover); }
td.preformatted { min-width: 220px; white-space: pre-wrap; word-break: break-word; }
pre {
  background: #070b14;
  border: 1px solid var(--border);
  border-radius: 8px;
  color: #e2e8f0;
  margin: 8px 0 0;
  overflow-x: auto;
  padding: 14px;
  white-space: pre-wrap;
  word-break: break-word;
}
.alert { border-left: 5px solid var(--warning); }
.alert-critical { border-left-color: var(--critical); }
.plot { min-height: 420px; overflow: visible; }
.disclosure summary {
  cursor: pointer;
  font-weight: 700;
  list-style-position: outside;
}
.disclosure summary::marker { color: #60a5fa; }
.empty { color: var(--muted); font-style: italic; }
@media print {
  :root {
    color-scheme: light;
    --background: #ffffff;
    --surface: #ffffff;
    --surface-raised: #f1f5f9;
    --surface-hover: #ffffff;
    --border: #cbd5e1;
    --text: #0f172a;
    --muted: #475569;
  }
  body { background: #ffffff; }
  main { width: 100%; padding: 0; }
  .panel, .summary-card { break-inside: avoid; box-shadow: none; }
  .table-hint { display: none; }
  .table-scroll { overflow: visible; }
  table { font-size: 0.75rem; width: 100%; }
  th, td { white-space: normal; }
  th:first-child, td:first-child { position: static; }
}
"""


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


def _format_value(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "да" if value else "нет"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _escape(value: object) -> str:
    return html.escape(_format_value(value), quote=True)


def _json_text(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )


def _status_badge(status: object) -> str:
    status_text = str(status)
    css_status = status_text if status_text in STATUS_LABELS else "skipped"
    label = STATUS_LABELS.get(status_text, status_text)
    return (
        f'<span class="badge status-{css_status}">'
        f"{_escape(label)} ({_escape(status_text)})</span>"
    )


def _reason_html(reason: object) -> str:
    if reason is None or reason == "":
        return ""
    return f'<p><strong>Причина:</strong> {_escape(reason)}</p>'


def _render_table(
    headers: list[str],
    rows: list[list[object]],
    *,
    label: str = "Таблица результатов",
) -> str:
    if not rows:
        return '<p class="empty">Нет данных для отображения.</p>'

    header_html = "".join(f"<th>{_escape(header)}</th>" for header in headers)
    body_rows: list[str] = []
    for row in rows:
        cells: list[str] = []
        for value in row:
            text = _format_value(value)
            css_class = ' class="preformatted"' if "\n" in text else ""
            cells.append(f"<td{css_class}>{html.escape(text, quote=True)}</td>")
        body_rows.append(f"<tr>{''.join(cells)}</tr>")

    hint = (
        '<div class="table-hint">Таблица прокручивается по горизонтали. '
        "Используйте полосу снизу, Shift + колесо мыши или стрелки после "
        "фокусировки.</div>"
        if len(headers) >= 6
        else ""
    )
    return (
        '<div class="table-shell">'
        f"{hint}"
        '<div class="table-scroll" role="region" tabindex="0" '
        f'aria-label="{html.escape(label, quote=True)}">'
        "<table><thead><tr>"
        f"{header_html}"
        "</tr></thead><tbody>"
        f"{''.join(body_rows)}"
        "</tbody></table></div></div>"
    )


def _render_summary(result: AnalysisResult) -> str:
    summary = result["summary"]
    metadata = result["metadata"]
    cards = [
        ("Строк Reference", metadata["reference_rows"]),
        ("Строк Current", metadata["current_rows"]),
        ("Проанализировано признаков", summary["analyzed_features"]),
        ("Пропущено признаков", summary["skipped_features"]),
        ("Алертов", summary["n_alerts"]),
        ("Random seed", metadata["random_seed"]),
    ]
    card_html = "".join(
        (
            '<div class="summary-card">'
            f'<span class="muted">{_escape(label)}</span>'
            f"<strong>{_escape(value)}</strong>"
            "</div>"
        )
        for label, value in cards
    )
    return (
        '<section id="summary">'
        "<h2>Сводка</h2>"
        '<div class="panel">'
        f"<p>{_status_badge(summary['status'])}</p>"
        f'<p class="muted">Версия контракта: {_escape(result["contract_version"])}</p>'
        "</div>"
        f'<div class="summary-grid">{card_html}</div>'
        "</section>"
    )


def _render_alerts(result: AnalysisResult) -> str:
    alerts = result["alerts"]
    if not alerts:
        content = '<div class="panel"><p>Алертов нет.</p></div>'
    else:
        rendered: list[str] = []
        for alert in alerts:
            severity = alert["severity"]
            extra_class = " alert-critical" if severity == "critical" else ""
            feature = (
                f" · признак {_escape(alert['feature'])}"
                if alert["feature"] is not None
                else ""
            )
            rendered.append(
                f'<article class="panel alert{extra_class}">'
                f"<strong>{_escape(alert['source'])} / "
                f"{_escape(alert['check'])}{feature}</strong>"
                f"<p>{_escape(alert['message'])}</p>"
                f'<span class="muted">Критичность: {_escape(severity)}</span>'
                "</article>"
            )
        content = "".join(rendered)
    return f'<section id="alerts"><h2>Алерты</h2>{content}</section>'


def _render_schema(result: AnalysisResult) -> str:
    schema = result["schema"]
    column_rows: list[list[object]] = []
    for issue_name, issue_label in (
        ("missing_columns", "Отсутствующие колонки"),
        ("extra_columns", "Лишние колонки"),
        ("duplicate_columns", "Повторяющиеся колонки"),
    ):
        for dataset, columns in schema[issue_name].items():
            if columns:
                column_rows.append([issue_label, dataset, ", ".join(columns)])

    mismatch_rows = [
        [
            mismatch["column"],
            mismatch["dataset"],
            mismatch["expected"],
            mismatch["actual"],
        ]
        for mismatch in schema["type_mismatches"]
    ]
    valid_features = ", ".join(schema["valid_features"]) or "—"
    return (
        '<section id="schema"><h2>Schema Validation</h2><div class="panel">'
        f"<p>{_status_badge(schema['status'])}</p>"
        f"{_reason_html(schema['reason'])}"
        f"<p><strong>Допустимые признаки:</strong> {_escape(valid_features)}</p>"
        "<h3>Колонки</h3>"
        f"{_render_table(['Нарушение', 'Выборка', 'Колонки'], column_rows)}"
        "<h3>Несовместимые типы</h3>"
        f"{_render_table(['Колонка', 'Выборка', 'Ожидалось', 'Получено'], mismatch_rows)}"
        "</div></section>"
    )


CHECK_LABELS = {
    "reference_duplicate_fraction": "Доля дубликатов Reference",
    "current_duplicate_fraction": "Доля дубликатов Current",
    "reference_missing_fraction": "Доля пропусков Reference",
    "current_missing_fraction": "Доля пропусков Current",
    "missing_increase_pp": "Рост пропусков, п.п.",
    "reference_infinite_count": "Бесконечности Reference",
    "current_infinite_count": "Бесконечности Current",
    "reference_min": "Минимум Reference",
    "current_min": "Минимум Current",
    "reference_max": "Максимум Reference",
    "current_max": "Максимум Current",
    "ks": "KS-test",
    "wasserstein": "Wasserstein",
    "chi2": "χ²-test",
    "psi": "PSI",
    "js": "Jensen–Shannon",
}


def _check_label(check: dict[str, Any]) -> str:
    name = str(check["name"])
    return CHECK_LABELS.get(name, name)


def _quality_row(feature: str | None, check: dict[str, Any]) -> list[object]:
    return [
        feature or "Вся таблица",
        _check_label(check),
        check["status"],
        _format_value(check["value"]),
        _format_value(check["threshold"]),
        check["reason"] or "—",
    ]


QUALITY_HEADERS = [
    "Признак",
    "Проверка",
    "Статус",
    "Значение",
    "Порог",
    "Причина",
]


def _drift_metric_row(feature: str, check: dict[str, Any]) -> list[object]:
    details = check["details"]
    return [
        feature,
        _check_label(check),
        check["status"],
        _format_value(check["value"]),
        _format_value(check["threshold"]),
        _format_value(check["p_value"]),
        _format_value(check["adjusted_p_value"]),
        details.get("threshold_source", "—"),
        check["reason"] or "—",
    ]


DRIFT_HEADERS = [
    "Признак",
    "Метод",
    "Статус",
    "Значение",
    "Порог",
    "p-value",
    "Скорр. p-value",
    "Источник порога",
    "Причина",
]


CATEGORY_DIAGNOSTIC_HEADERS = [
    "Признак",
    "Метод",
    "Cramér's V",
    "Новых категорий",
    "Исчезнувших категорий",
    "Pooled-категорий",
    "Доля pooled-наблюдений",
]


def _category_diagnostic_row(
    feature: str,
    check: dict[str, Any],
) -> list[object] | None:
    details = check["details"]
    diagnostic_keys = {
        "cramers_v",
        "new_category_count",
        "disappeared_category_count",
        "pooled_category_count",
        "pooled_observation_fraction",
    }
    if not diagnostic_keys.intersection(details):
        return None

    pooled_fraction = details.get("pooled_observation_fraction")
    if isinstance(pooled_fraction, dict):
        pooled_fraction = pooled_fraction.get("combined")
    return [
        feature,
        _check_label(check),
        _format_value(details.get("cramers_v")),
        _format_value(details.get("new_category_count")),
        _format_value(details.get("disappeared_category_count")),
        _format_value(details.get("pooled_category_count")),
        _format_value(pooled_fraction),
    ]


def _render_quality(result: AnalysisResult) -> str:
    quality = result["quality"]
    rows = [
        _quality_row(None, check)
        for check in quality["dataset_checks"]
    ]
    for feature, checks in quality["feature_checks"].items():
        rows.extend(_quality_row(feature, check) for check in checks)

    return (
        '<section id="quality"><h2>Data Quality</h2><div class="panel">'
        f"<p>{_status_badge(quality['status'])}</p>"
        f"{_reason_html(quality['reason'])}"
        f"{_render_table(QUALITY_HEADERS, rows, label='Проверки Data Quality')}"
        "</div></section>"
    )


def _render_drift(result: AnalysisResult) -> str:
    drift = result["drift"]
    feature_rows: list[list[object]] = []
    check_rows: list[list[object]] = []
    category_rows: list[list[object]] = []
    for feature, feature_result in drift["features"].items():
        feature_rows.append(
            [
                feature,
                feature_result["feature_type"],
                feature_result["status"],
                feature_result["n_reference_valid"],
                feature_result["n_current_valid"],
                feature_result["reason"] or "—",
            ]
        )
        for check in feature_result["checks"].values():
            check_rows.append(_drift_metric_row(feature, check))
            diagnostic_row = _category_diagnostic_row(feature, check)
            if diagnostic_row is not None:
                category_rows.append(diagnostic_row)

    category_diagnostics = ""
    if category_rows:
        diagnostic_table = _render_table(
            CATEGORY_DIAGNOSTIC_HEADERS,
            category_rows,
            label="Диагностика категориального дрейфа",
        )
        category_diagnostics = (
            '<details class="disclosure"><summary>'
            "Дополнительная диагностика категориальных признаков "
            f"({len(category_rows)})</summary>"
            f"{diagnostic_table}"
            "</details>"
        )

    feature_headers = [
        "Признак",
        "Тип",
        "Статус",
        "Валидных Reference",
        "Валидных Current",
        "Причина",
    ]
    return (
        '<section id="drift"><h2>Data Drift</h2><div class="panel">'
        f"<p>{_status_badge(drift['status'])}</p>"
        f"{_reason_html(drift['reason'])}"
        "<h3>Сводка по признакам</h3>"
        f"{_render_table(feature_headers, feature_rows, label='Сводка Data Drift по признакам')}"
        "<h3>Метрики</h3>"
        f"{_render_table(DRIFT_HEADERS, check_rows, label='Метрики Data Drift')}"
        f"{category_diagnostics}"
        "</div></section>"
    )


def _render_adversarial(result: AnalysisResult) -> str:
    adversarial = result["adversarial"]
    overview_rows = [
        ["Статус", adversarial["status"]],
        ["ROC-AUC", _format_value(adversarial["roc_auc"])],
        ["Порог", _format_value(adversarial["threshold"])],
        ["Алерт", _format_value(adversarial["alert"])],
        ["Стратегия split", adversarial["split_strategy"]],
        ["Group column", _format_value(adversarial["group_column"])],
        ["Количество групп", _format_value(adversarial["n_groups"])],
        ["Тип importance", _format_value(adversarial["importance_type"])],
    ]
    fold_rows = [
        [index, value]
        for index, value in enumerate(adversarial["fold_auc"], start=1)
    ]
    importance_rows = sorted(
        ([feature, value] for feature, value in adversarial["feature_importance"].items()),
        key=lambda row: float(row[1]),
        reverse=True,
    )
    return (
        '<section id="adversarial"><h2>Adversarial Validation</h2>'
        '<div class="panel">'
        f"<p>{_status_badge(adversarial['status'])}</p>"
        f"{_reason_html(adversarial['reason'])}"
        f"{_render_table(['Поле', 'Значение'], overview_rows)}"
        "<h3>ROC-AUC по фолдам</h3>"
        f"{_render_table(['Фолд', 'ROC-AUC'], fold_rows)}"
        "<h3>Важности признаков</h3>"
        f"{_render_table(['Признак', 'Важность'], importance_rows)}"
        "</div></section>"
    )


def _render_effective_config(result: AnalysisResult) -> str:
    config = html.escape(_json_text(result["effective_config"]), quote=True)
    return (
        '<section id="config"><h2>Фактически применённая конфигурация</h2>'
        '<details class="panel disclosure"><summary>Показать конфигурацию</summary>'
        f"<pre>{config}</pre></details></section>"
    )


def _apply_report_plot_theme(figure: go.Figure) -> None:
    """Применить тёмную тему только к графикам автономного HTML-отчёта."""

    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#0f172a",
        font={"color": "#e5e7eb"},
        legend={
            "bgcolor": "rgba(17, 24, 39, 0.82)",
            "bordercolor": "#334155",
            "borderwidth": 1,
        },
    )
    figure.update_xaxes(gridcolor="#334155", zerolinecolor="#475569")
    figure.update_yaxes(gridcolor="#334155", zerolinecolor="#475569")
    for annotation in figure.layout.annotations:
        annotation.font.color = "#94a3b8"


def _escape_figure_labels(figure: go.Figure, feature_type: FeatureType) -> None:
    """Экранировать только пользовательские подписи, сохранив Plotly-шаблоны."""

    if figure.layout.title.text is not None:
        figure.layout.title.text = html.escape(
            str(figure.layout.title.text), quote=True
        )
    if figure.layout.xaxis.title.text is not None:
        figure.layout.xaxis.title.text = html.escape(
            str(figure.layout.xaxis.title.text), quote=True
        )
    if feature_type == "categorical":
        for trace in figure.data:
            trace.x = tuple(html.escape(str(value), quote=True) for value in trace.x)


def _render_distributions(
    result: AnalysisResult,
    reference: pd.DataFrame | None,
    current: pd.DataFrame | None,
) -> tuple[str, bool]:
    if reference is None or current is None:
        return (
            '<section id="distributions"><h2>Распределения</h2>'
            '<div class="panel"><p class="empty">'
            "Графики не включены: Reference и Current не были переданы в экспорт."
            "</p></div></section>",
            False,
        )

    figures: list[str] = []
    for index, (feature, feature_result) in enumerate(
        result["drift"]["features"].items(), start=1
    ):
        if feature not in reference.columns or feature not in current.columns:
            figures.append(
                '<article class="panel">'
                f"<h3>{_escape(feature)}</h3>"
                '<p class="empty">График недоступен: признак отсутствует в одной '
                "из переданных таблиц.</p></article>"
            )
            continue

        feature_type = feature_result["feature_type"]
        try:
            figure = distribution_figure(
                reference[feature],
                current[feature],
                feature_type,
            )
        except (TypeError, ValueError) as exc:
            figures.append(
                '<article class="panel">'
                f"<h3>{_escape(feature)}</h3>"
                f'<p class="empty">График недоступен: {_escape(exc)}</p>'
                "</article>"
            )
            continue

        _apply_report_plot_theme(figure)
        _escape_figure_labels(figure, feature_type)
        fragment = to_html(
            figure,
            full_html=False,
            include_plotlyjs=False,
            div_id=f"distribution-{index}",
            config={"displaylogo": False, "responsive": True},
        )
        figures.append(
            '<article class="panel plot">'
            f"<h3>{_escape(feature)}</h3>{fragment}</article>"
        )

    if not figures:
        figures.append(
            '<div class="panel"><p class="empty">'
            "Нет признаков для построения распределений.</p></div>"
        )
    has_plot = any("Plotly.newPlot" in figure for figure in figures)
    return (
        '<section id="distributions"><h2>Распределения</h2>'
        f"{''.join(figures)}</section>",
        has_plot,
    )


def _validate_report_inputs(
    result: object,
    reference: object,
    current: object,
) -> None:
    if not isinstance(result, dict):
        raise TypeError(
            "result должен быть словарём AnalysisResult, "
            f"получен {type(result).__name__}"
        )
    if (reference is None) != (current is None):
        raise ValueError(
            "Для графиков необходимо передать одновременно reference и current"
        )
    if reference is not None and not isinstance(reference, pd.DataFrame):
        raise TypeError(
            "reference должен иметь тип pandas.DataFrame, "
            f"получен {type(reference).__name__}"
        )
    if current is not None and not isinstance(current, pd.DataFrame):
        raise TypeError(
            "current должен иметь тип pandas.DataFrame, "
            f"получен {type(current).__name__}"
        )


def _build_html_document(
    result: AnalysisResult,
    *,
    reference: pd.DataFrame | None,
    current: pd.DataFrame | None,
) -> str:
    _validate_report_inputs(result, reference, current)
    distributions, has_plot = _render_distributions(result, reference, current)
    plotly_script = (
        f'<script id="plotly-library">{get_plotlyjs()}</script>' if has_plot else ""
    )
    body = "".join(
        (
            _render_summary(result),
            _render_alerts(result),
            _render_schema(result),
            _render_quality(result),
            _render_drift(result),
            _render_adversarial(result),
            distributions,
            _render_effective_config(result),
        )
    )
    return (
        "<!doctype html>\n"
        '<html lang="ru"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>Data Drift Guardian — отчёт</title>"
        f"<style>{REPORT_STYLES}</style>{plotly_script}</head>"
        "<body><main><header><h1>Data Drift Guardian</h1>"
        '<p class="muted">Автономный отчёт по качеству данных и сдвигу распределений.</p>'
        f"</header>{body}</main></body></html>"
    )


def _prepare_output_path(path: str | Path, *, overwrite: bool) -> Path:
    if not isinstance(path, (str, Path)):
        raise TypeError(
            "output_path должен иметь тип str или pathlib.Path, "
            f"получен {type(path).__name__}"
        )
    output_path = Path(path).expanduser()
    if output_path.exists():
        if output_path.is_dir():
            raise IsADirectoryError(
                f"Путь HTML-отчёта указывает на директорию: {output_path}"
            )
        if not output_path.is_file():
            raise ValueError(
                f"Путь HTML-отчёта не является обычным файлом: {output_path}"
            )
        if not overwrite:
            raise FileExistsError(
                f"HTML-отчёт уже существует: {output_path}. "
                "Передайте overwrite=True для замены."
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not output_path.parent.is_dir():
        raise NotADirectoryError(
            "Родительский путь HTML-отчёта не является директорией: "
            f"{output_path.parent}"
        )
    return output_path


def _write_text_atomic(content: str, output_path: Path) -> None:
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(content)
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def export_html(
    result: AnalysisResult,
    output_path: str | Path,
    *,
    reference: pd.DataFrame | None = None,
    current: pd.DataFrame | None = None,
    overwrite: bool = False,
) -> Path:
    """Создать автономный HTML-отчёт из готового ``AnalysisResult``.

    Статистические показатели берутся только из ``result`` и не вычисляются
    повторно. Исходные DataFrame необязательны и используются исключительно
    для построения сопоставимых Plotly-распределений.

    Args:
        result: Полный JSON-безопасный результат ``analyze``.
        output_path: Путь создаваемого HTML-файла.
        reference: Необязательная эталонная таблица для графиков.
        current: Необязательная текущая таблица для графиков.
        overwrite: Разрешить атомарную замену существующего файла.

    Returns:
        Фактический путь записанного отчёта.

    Raises:
        TypeError: При неверном типе аргумента.
        ValueError: Если передан только один DataFrame либо результат не
            сериализуется без ``NaN``/``Infinity``.
        FileExistsError: Если файл уже существует и ``overwrite=False``.
        OSError: При ошибке подготовки директории или записи.
    """
    prepared_path = _prepare_output_path(output_path, overwrite=overwrite)
    document = _build_html_document(
        result,
        reference=reference,
        current=current,
    )
    _write_text_atomic(document, prepared_path)
    return prepared_path
