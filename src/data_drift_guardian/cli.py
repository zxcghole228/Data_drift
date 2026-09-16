"""Командный интерфейс Data Drift Guardian."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from collections.abc import Sequence
from pathlib import Path
from tempfile import NamedTemporaryFile

from .config import load_config
from .contracts import AnalysisResult
from .ingestion import load_table
from .pipeline import analyze


EXIT_SUCCESS = 0
EXIT_ERROR = 1
EXIT_ALERT = 2


class CliArgumentParser(argparse.ArgumentParser):
    """ArgumentParser, возвращающий код 1 для некорректной команды.

    Стандартный ``argparse`` использует код 2 для ошибки аргументов, но в
    контракте CLI этот код зарезервирован для ``--fail-on-alert``.
    """

    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(EXIT_ERROR, f"{self.prog}: ошибка: {message}\n")


def build_parser() -> argparse.ArgumentParser:
    """Построить parser публичного командного интерфейса."""

    parser = CliArgumentParser(
        prog="python -m data_drift_guardian",
        description=(
            "Сравнить Reference и Current, выполнить проверки качества и "
            "дрейфа и сохранить полный AnalysisResult в JSON."
        ),
    )
    parser.add_argument(
        "--reference",
        required=True,
        type=Path,
        metavar="PATH",
        help="Reference в формате CSV или Parquet.",
    )
    parser.add_argument(
        "--current",
        required=True,
        type=Path,
        metavar="PATH",
        help="Current в формате CSV или Parquet.",
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        metavar="PATH",
        help="YAML-конфигурация анализа.",
    )
    parser.add_argument(
        "--json-output",
        required=True,
        type=Path,
        metavar="PATH",
        help="Путь для полного JSON-результата.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Разрешить замену существующего JSON-файла.",
    )
    parser.add_argument(
        "--fail-on-alert",
        action="store_true",
        help="Вернуть код 2, если анализ создал хотя бы один алерт.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Показать traceback при ошибке запуска.",
    )
    return parser


def _prepare_output_path(path: Path, *, overwrite: bool) -> Path:
    """Проверить output до анализа и подготовить родительскую директорию."""

    output_path = path.expanduser()
    if output_path.exists():
        if output_path.is_dir():
            raise IsADirectoryError(
                f"Путь JSON-результата указывает на директорию: {output_path}"
            )
        if not output_path.is_file():
            raise ValueError(
                f"Путь JSON-результата не является обычным файлом: {output_path}"
            )
        if not overwrite:
            raise FileExistsError(
                f"Файл результата уже существует: {output_path}. "
                "Передайте --overwrite для замены."
            )

    parent = output_path.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSError(
            f"Не удалось подготовить директорию результата {parent}: {exc}"
        ) from exc
    if not parent.is_dir():
        raise NotADirectoryError(
            f"Родительский путь результата не является директорией: {parent}"
        )
    return output_path


def _serialize_result(result: AnalysisResult) -> str:
    """Сериализовать полный результат до начала файловой записи."""

    return json.dumps(
        result,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )


def _write_text_atomic(content: str, output_path: Path) -> None:
    """Атомарно записать UTF-8 текст через временный файл рядом с целевым."""

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


def write_json_atomic(result: AnalysisResult, output_path: Path) -> None:
    """Проверить JSON-безопасность и атомарно сохранить AnalysisResult."""

    serialized = _serialize_result(result)
    _write_text_atomic(serialized, output_path)


def _print_summary(result: AnalysisResult, output_path: Path) -> None:
    summary = result["summary"]
    print("Анализ завершён")
    print(f"Статус: {summary['status']}")
    print(f"Проанализировано признаков: {summary['analyzed_features']}")
    print(f"Пропущено признаков: {summary['skipped_features']}")
    print(f"Алертов: {summary['n_alerts']}")
    print(f"JSON: {output_path}")


def _run(args: argparse.Namespace) -> int:
    output_path = _prepare_output_path(
        args.json_output,
        overwrite=args.overwrite,
    )
    reference = load_table(args.reference)
    current = load_table(args.current)
    config = load_config(args.config)

    result = analyze(reference, current, config=config)
    write_json_atomic(result, output_path)
    _print_summary(result, output_path)

    if args.fail_on_alert and result["summary"]["has_alerts"]:
        return EXIT_ALERT
    return EXIT_SUCCESS


def main(argv: Sequence[str] | None = None) -> int:
    """Запустить CLI и вернуть документированный код завершения."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _run(args)
    except Exception as exc:  # CLI является границей обработки ошибок.
        message = str(exc) or type(exc).__name__
        print(f"Ошибка: {message}", file=sys.stderr)
        if args.debug:
            traceback.print_exc()
        return EXIT_ERROR
