"""Графики и HTML. Ответственный: Павел; в первую неделю — базовые графики."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .contracts import AnalysisResult, FeatureType

if TYPE_CHECKING:
    import pandas as pd
    import plotly.graph_objects as go


def distribution_figure(
    reference: pd.Series,
    current: pd.Series,
    feature_type: FeatureType,
) -> go.Figure:
    """Построить сопоставимые распределения; пока заглушка."""
    raise NotImplementedError("Графики запланированы в docs/plans/pavel.md")


def export_html(result: AnalysisResult, output_path: str | Path) -> Path:
    """Экспорт отчёта — итоговое требование; реализация пока не запланирована по дням."""
    raise NotImplementedError("HTML-экспорт отмечен в docs/deliverables.md")
