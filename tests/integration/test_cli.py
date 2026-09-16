"""Интеграционные тесты командного интерфейса."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from data_drift_guardian import cli
from data_drift_guardian.contracts import AnalysisResult


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


def _frames(*, with_alert: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_rows = 100
    reference = pd.DataFrame(
        {
            "age": np.linspace(20.0, 60.0, n_rows),
            "income": np.linspace(40_000.0, 80_000.0, n_rows),
            "region": ["north", "south", "central", "west"] * 25,
        }
    )
    current = reference.copy(deep=True)
    if with_alert:
        current.loc[:19, "income"] = np.nan
    return reference, current


def _write_tables(
    tmp_path: Path,
    *,
    file_format: str,
    with_alert: bool = False,
) -> tuple[Path, Path]:
    reference, current = _frames(with_alert=with_alert)
    reference_path = tmp_path / f"reference.{file_format}"
    current_path = tmp_path / f"current.{file_format}"
    if file_format == "csv":
        reference.to_csv(reference_path, index=False)
        current.to_csv(current_path, index=False)
    else:
        reference.to_parquet(reference_path, index=False)
        current.to_parquet(current_path, index=False)
    return reference_path, current_path


def _arguments(
    reference_path: Path,
    current_path: Path,
    output_path: Path,
    *,
    config_path: Path = DEFAULT_CONFIG,
    extra: list[str] | None = None,
) -> list[str]:
    arguments = [
        "--reference",
        str(reference_path),
        "--current",
        str(current_path),
        "--config",
        str(config_path),
        "--json-output",
        str(output_path),
    ]
    if extra:
        arguments.extend(extra)
    return arguments


def _read_strict_json(path: Path) -> dict[str, Any]:
    def reject_non_finite(value: str) -> None:
        raise AssertionError(f"JSON содержит недопустимую константу {value}")

    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_non_finite,
    )


def test_module_help_is_available() -> None:
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    source_path = str(PROJECT_ROOT / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (source_path, environment.get("PYTHONPATH")))
    )

    completed = subprocess.run(
        [sys.executable, "-m", "data_drift_guardian", "--help"],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert completed.returncode == cli.EXIT_SUCCESS
    assert "--reference" in completed.stdout
    assert "--json-output" in completed.stdout
    assert "--html-output" in completed.stdout
    assert completed.stderr == ""


def test_missing_required_arguments_return_code_one(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main([])

    captured = capsys.readouterr()
    assert exc_info.value.code == cli.EXIT_ERROR
    assert "--reference" in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize("file_format", ["csv", "parquet"])
def test_cli_writes_complete_json_for_supported_tables(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    file_format: str,
) -> None:
    reference_path, current_path = _write_tables(
        tmp_path,
        file_format=file_format,
    )
    output_path = tmp_path / "nested" / "result.json"

    exit_code = cli.main(_arguments(reference_path, current_path, output_path))

    captured = capsys.readouterr()
    result = _read_strict_json(output_path)
    assert exit_code == cli.EXIT_SUCCESS
    assert result["contract_version"] == "0.2"
    assert result["summary"]["status"] == "ok"
    assert result["summary"]["n_alerts"] == 0
    assert "effective_config" in result
    assert "Анализ завершён" in captured.out
    assert str(output_path) in captured.out
    assert captured.err == ""


def test_existing_output_requires_explicit_overwrite(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    reference_path, current_path = _write_tables(tmp_path, file_format="csv")
    output_path = tmp_path / "result.json"
    output_path.write_text("исходное содержимое", encoding="utf-8")

    refused = cli.main(_arguments(reference_path, current_path, output_path))
    refused_output = capsys.readouterr()

    assert refused == cli.EXIT_ERROR
    assert output_path.read_text(encoding="utf-8") == "исходное содержимое"
    assert "--overwrite" in refused_output.err

    replaced = cli.main(
        _arguments(
            reference_path,
            current_path,
            output_path,
            extra=["--overwrite"],
        )
    )
    replaced_output = capsys.readouterr()

    assert replaced == cli.EXIT_SUCCESS
    assert _read_strict_json(output_path)["contract_version"] == "0.2"
    assert replaced_output.err == ""


def test_fail_on_alert_returns_two_after_writing_result(
    tmp_path: Path,
) -> None:
    reference_path, current_path = _write_tables(
        tmp_path,
        file_format="csv",
        with_alert=True,
    )
    output_path = tmp_path / "alert.json"

    exit_code = cli.main(
        _arguments(
            reference_path,
            current_path,
            output_path,
            extra=["--fail-on-alert"],
        )
    )

    result = _read_strict_json(output_path)
    assert exit_code == cli.EXIT_ALERT
    assert result["summary"]["has_alerts"] is True
    assert result["summary"]["n_alerts"] > 0


def test_alert_does_not_fail_without_explicit_flag(tmp_path: Path) -> None:
    reference_path, current_path = _write_tables(
        tmp_path,
        file_format="csv",
        with_alert=True,
    )
    output_path = tmp_path / "alert.json"

    exit_code = cli.main(_arguments(reference_path, current_path, output_path))

    assert exit_code == cli.EXIT_SUCCESS
    assert _read_strict_json(output_path)["summary"]["has_alerts"] is True


def test_invalid_yaml_does_not_create_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    reference_path, current_path = _write_tables(tmp_path, file_format="csv")
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("schema: [", encoding="utf-8")
    output_path = tmp_path / "result.json"

    exit_code = cli.main(
        _arguments(
            reference_path,
            current_path,
            output_path,
            config_path=config_path,
        )
    )

    captured = capsys.readouterr()
    assert exit_code == cli.EXIT_ERROR
    assert not output_path.exists()
    assert "YAML" in captured.err
    assert "Traceback" not in captured.err


def test_schema_problem_is_preserved_as_structured_result(tmp_path: Path) -> None:
    reference, current = _frames()
    current = current.drop(columns="region")
    reference_path = tmp_path / "reference.csv"
    current_path = tmp_path / "current.csv"
    reference.to_csv(reference_path, index=False)
    current.to_csv(current_path, index=False)
    output_path = tmp_path / "schema-problem.json"

    exit_code = cli.main(_arguments(reference_path, current_path, output_path))

    result = _read_strict_json(output_path)
    assert exit_code == cli.EXIT_SUCCESS
    assert result["schema"]["status"] == "error"
    assert "region" in result["schema"]["missing_columns"]["current"]
    assert result["summary"]["has_alerts"] is True


def test_debug_controls_traceback_and_failed_run_leaves_no_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, current_path = _write_tables(tmp_path, file_format="csv")
    missing_reference = tmp_path / "missing.csv"
    normal_output = tmp_path / "normal.json"

    normal_code = cli.main(
        _arguments(missing_reference, current_path, normal_output)
    )
    normal_message = capsys.readouterr()

    assert normal_code == cli.EXIT_ERROR
    assert "Файл не найден" in normal_message.err
    assert "Traceback" not in normal_message.err
    assert not normal_output.exists()

    debug_output = tmp_path / "debug.json"
    debug_code = cli.main(
        _arguments(
            missing_reference,
            current_path,
            debug_output,
            extra=["--debug"],
        )
    )
    debug_message = capsys.readouterr()

    assert debug_code == cli.EXIT_ERROR
    assert "Traceback" in debug_message.err
    assert "FileNotFoundError" in debug_message.err
    assert not debug_output.exists()


def test_serialization_failure_keeps_existing_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference_path, current_path = _write_tables(tmp_path, file_format="csv")
    output_path = tmp_path / "result.json"
    output_path.write_text("не заменять", encoding="utf-8")
    invalid_result = cast(AnalysisResult, {"invalid": float("nan")})
    monkeypatch.setattr(cli, "analyze", lambda *args, **kwargs: invalid_result)

    exit_code = cli.main(
        _arguments(
            reference_path,
            current_path,
            output_path,
            extra=["--overwrite"],
        )
    )

    assert exit_code == cli.EXIT_ERROR
    assert output_path.read_text(encoding="utf-8") == "не заменять"
    assert not list(tmp_path.glob(".result.json.*.tmp"))


def test_cli_writes_json_and_html_after_single_analysis(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference_path, current_path = _write_tables(tmp_path, file_format="csv")
    json_output = tmp_path / "outputs" / "result.json"
    html_output = tmp_path / "outputs" / "report.html"
    original_analyze = cli.analyze
    calls = 0

    def counting_analyze(*args: object, **kwargs: object) -> AnalysisResult:
        nonlocal calls
        calls += 1
        return original_analyze(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(cli, "analyze", counting_analyze)

    exit_code = cli.main(
        _arguments(
            reference_path,
            current_path,
            json_output,
            extra=["--html-output", str(html_output)],
        )
    )

    captured = capsys.readouterr()
    html_content = html_output.read_text(encoding="utf-8")
    assert exit_code == cli.EXIT_SUCCESS
    assert calls == 1
    assert _read_strict_json(json_output)["contract_version"] == "0.2"
    assert html_content.startswith("<!doctype html>")
    assert "Plotly.newPlot" in html_content
    assert str(json_output) in captured.out
    assert str(html_output) in captured.out
    assert captured.err == ""


def test_existing_html_is_rejected_before_analysis(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference_path, current_path = _write_tables(tmp_path, file_format="csv")
    json_output = tmp_path / "result.json"
    html_output = tmp_path / "report.html"
    html_output.write_text("исходный HTML", encoding="utf-8")
    calls = 0

    def unexpected_analyze(*args: object, **kwargs: object) -> AnalysisResult:
        nonlocal calls
        calls += 1
        raise AssertionError("analyze не должен запускаться")

    monkeypatch.setattr(cli, "analyze", unexpected_analyze)

    exit_code = cli.main(
        _arguments(
            reference_path,
            current_path,
            json_output,
            extra=["--html-output", str(html_output)],
        )
    )

    captured = capsys.readouterr()
    assert exit_code == cli.EXIT_ERROR
    assert calls == 0
    assert not json_output.exists()
    assert html_output.read_text(encoding="utf-8") == "исходный HTML"
    assert "--overwrite" in captured.err


def test_json_and_html_output_paths_must_differ(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    reference_path, current_path = _write_tables(tmp_path, file_format="csv")
    output_path = tmp_path / "same-output"

    exit_code = cli.main(
        _arguments(
            reference_path,
            current_path,
            output_path,
            extra=["--html-output", str(output_path)],
        )
    )

    captured = capsys.readouterr()
    assert exit_code == cli.EXIT_ERROR
    assert not output_path.exists()
    assert "должны различаться" in captured.err


def test_overwrite_replaces_existing_json_and_html(
    tmp_path: Path,
) -> None:
    reference_path, current_path = _write_tables(tmp_path, file_format="parquet")
    json_output = tmp_path / "result.json"
    html_output = tmp_path / "report.html"
    json_output.write_text("старый JSON", encoding="utf-8")
    html_output.write_text("старый HTML", encoding="utf-8")

    exit_code = cli.main(
        _arguments(
            reference_path,
            current_path,
            json_output,
            extra=[
                "--html-output",
                str(html_output),
                "--overwrite",
            ],
        )
    )

    assert exit_code == cli.EXIT_SUCCESS
    assert _read_strict_json(json_output)["contract_version"] == "0.2"
    assert html_output.read_text(encoding="utf-8").startswith("<!doctype html>")
