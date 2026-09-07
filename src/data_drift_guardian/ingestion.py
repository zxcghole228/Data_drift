"""Загрузка локальных табличных данных из CSV и Parquet."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


SUPPORTED_EXTENSIONS = frozenset({".csv", ".parquet"})


def load_table(path: str | Path) -> pd.DataFrame:
    """Загрузить локальный CSV- или Parquet-файл в ``pandas.DataFrame``.

    Функция отвечает только за чтение файла. Она не преобразует типы,
    не исправляет пропуски и не проверяет схему данных.

    Args:
        path: Путь к локальному файлу в виде строки или ``pathlib.Path``.

    Returns:
        Таблица, прочитанная средствами pandas.

    Raises:
        TypeError: Если ``path`` не является строкой или ``pathlib.Path``.
        FileNotFoundError: Если указанный путь не существует.
        IsADirectoryError: Если путь указывает на директорию.
        ValueError: Если путь не является обычным файлом или его расширение
            не входит в список поддерживаемых.

    Ошибки чтения содержимого от pandas или PyArrow передаются вызывающему
    коду без изменения.
    """
    if not isinstance(path, (str, Path)):
        raise TypeError(
            "path должен иметь тип str или pathlib.Path, "
            f"получен {type(path).__name__}"
        )

    table_path = Path(path).expanduser()

    if not table_path.exists():
        raise FileNotFoundError(f"Файл не найден: {table_path}")

    if table_path.is_dir():
        raise IsADirectoryError(f"Ожидался файл, получена директория: {table_path}")

    if not table_path.is_file():
        raise ValueError(f"Путь не указывает на обычный файл: {table_path}")

    extension = table_path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        actual = extension or "<без расширения>"
        raise ValueError(
            f"Неподдерживаемый формат файла {actual}: {table_path}. "
            f"Поддерживаются: {supported}"
        )

    if extension == ".csv":
        return pd.read_csv(table_path)

    return pd.read_parquet(table_path)
