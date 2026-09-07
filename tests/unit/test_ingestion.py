"""Тесты загрузки локальных CSV- и Parquet-файлов."""

from pathlib import Path

import pandas as pd
import pytest

from data_drift_guardian.ingestion import load_table


@pytest.fixture
def sample_table() -> pd.DataFrame:
    """Небольшая таблица, создаваемая только в памяти теста."""
    return pd.DataFrame(
        {
            "age": [24, 37, 51],
            "income": [42_000.50, 73_000.00, 51_500.25],
            "region": ["north", "south", "west"],
        }
    )


def test_load_table_reads_csv_from_string_path(
    tmp_path: Path,
    sample_table: pd.DataFrame,
) -> None:
    csv_path = tmp_path / "reference.CSV"
    sample_table.to_csv(csv_path, index=False)

    result = load_table(str(csv_path))

    pd.testing.assert_frame_equal(result, sample_table)


def test_load_table_reads_parquet_from_path(
    tmp_path: Path,
    sample_table: pd.DataFrame,
) -> None:
    parquet_path = tmp_path / "current.PARQUET"
    sample_table.to_parquet(parquet_path, index=False)

    result = load_table(parquet_path)

    pd.testing.assert_frame_equal(result, sample_table)


def test_load_table_raises_for_missing_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError, match="Файл не найден"):
        load_table(missing_path)


def test_load_table_raises_for_directory(tmp_path: Path) -> None:
    directory_path = tmp_path / "tables"
    directory_path.mkdir()

    with pytest.raises(IsADirectoryError, match="получена директория"):
        load_table(directory_path)


def test_load_table_raises_for_unsupported_extension(tmp_path: Path) -> None:
    unsupported_path = tmp_path / "table.xlsx"
    unsupported_path.write_text("not an Excel file", encoding="utf-8")

    with pytest.raises(ValueError, match=r"Неподдерживаемый формат файла \.xlsx"):
        load_table(unsupported_path)


def test_load_table_raises_for_path_without_extension(tmp_path: Path) -> None:
    extensionless_path = tmp_path / "table"
    extensionless_path.write_text("no extension", encoding="utf-8")

    with pytest.raises(ValueError, match="<без расширения>"):
        load_table(extensionless_path)


def test_load_table_raises_for_invalid_path_type() -> None:
    with pytest.raises(TypeError, match="получен int"):
        load_table(123)  # type: ignore[arg-type]
