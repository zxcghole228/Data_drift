"""Загрузка CSV/Parquet. Ответственный: Павел; см. план на 07.09."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def load_table(path: str | Path) -> pd.DataFrame:
    """Прочитать таблицу; текущая версия является заглушкой."""
    raise NotImplementedError("Загрузчик запланирован в docs/plans/pavel.md")
