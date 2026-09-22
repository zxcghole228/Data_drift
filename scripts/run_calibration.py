"""Run the reproducible synthetic calibration grid.

The runner intentionally calls the public ``analyze`` API.  It writes one
long-form CSV with every decision, one aggregated CSV with observed alert
rates and one JSON file that records the inputs, environment, runtime and Git
revision.

Run the full documented grid from the repository root::

    python scripts/run_calibration.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_drift_guardian import analyze
from data_drift_guardian.config import load_config
from data_drift_guardian.contracts import AnalysisConfig, AnalysisResult, CheckResult
from scripts.generate_demo_data import Scenario, generate_demo_data

DriftFamily = Literal[
    "control", "numeric", "categorical", "missingness", "combined"
]

DEFAULT_SEEDS: tuple[int, ...] = tuple(range(2026091900, 2026091920))
DEFAULT_CURRENT_SIZES: tuple[int, ...] = (500, 1_200, 3_000)
DEFAULT_AGE_SHIFTS: tuple[float, ...] = (2.0, 4.0, 8.0)
DEFAULT_MISSING_FRACTIONS: tuple[float, ...] = (0.04, 0.08, 0.15)
DEFAULT_N_REFERENCE = 5_000
DEFAULT_REFERENCE_MISSING_FRACTION = 0.02
DEFAULT_ACCEPTANCE_AGE_SHIFT = 8.0
DEFAULT_ACCEPTANCE_MISSING_FRACTION = 0.08
DEFAULT_CONFIG_PATH = Path("configs/demo_calibrated.yaml")
DEFAULT_OUTPUT_DIR = Path("report/experiments")

RUNS_FILENAME = "calibration_runs.csv"
SUMMARY_FILENAME = "calibration_summary.csv"
METADATA_FILENAME = "calibration_metadata.json"

PACKAGE_NAMES: tuple[str, ...] = (
    "data-drift-guardian",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "lightgbm",
    "plotly",
    "streamlit",
    "PyYAML",
    "pyarrow",
    "pytest",
)


@dataclass(frozen=True)
class CalibrationCase:
    """One scenario and magnitude at a fixed Current size."""

    case_id: str
    scenario: Scenario
    drift_family: DriftFamily
    drift_level: str
    n_current: int
    age_shift_years: float
    current_missing_fraction: float


def _require_positive_ints(values: Sequence[int], name: str) -> tuple[int, ...]:
    normalized = tuple(values)
    if not normalized or any(type(value) is not int or value <= 0 for value in values):
        raise ValueError(f"{name} должен содержать положительные целые числа")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} не должен содержать повторы")
    return normalized


def _require_non_negative_ints(
    values: Sequence[int], name: str
) -> tuple[int, ...]:
    normalized = tuple(values)
    if not normalized or any(type(value) is not int or value < 0 for value in values):
        raise ValueError(f"{name} должен содержать неотрицательные целые числа")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} не должен содержать повторы")
    return normalized


def _require_positive_floats(
    values: Sequence[float], name: str
) -> tuple[float, ...]:
    normalized = tuple(float(value) for value in values)
    if not normalized or any(
        not isfinite(value) or value <= 0.0 for value in normalized
    ):
        raise ValueError(f"{name} должен содержать положительные числа")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} не должен содержать повторы")
    return normalized


def _require_missing_fractions(
    values: Sequence[float], reference_fraction: float
) -> tuple[float, ...]:
    normalized = tuple(float(value) for value in values)
    reference_fraction = float(reference_fraction)
    if not isfinite(reference_fraction) or not 0.0 <= reference_fraction <= 1.0:
        raise ValueError("reference_missing_fraction должен быть в диапазоне [0, 1]")
    if not normalized or any(
        not isfinite(value) or value <= reference_fraction or value > 1.0
        for value in normalized
    ):
        raise ValueError(
            "missing_fractions должны быть больше reference_missing_fraction "
            "и не превышать 1"
        )
    if len(set(normalized)) != len(normalized):
        raise ValueError("missing_fractions не должен содержать повторы")
    return normalized


def _level_name(index: int, total: int) -> str:
    if total == 1:
        return "single"
    if index == 0:
        return "weak"
    if index == total - 1:
        return "strong"
    if total == 3 and index == 1:
        return "medium"
    return f"level_{index + 1}"


def _number_token(value: float) -> str:
    return format(value, "g").replace(".", "p")


def build_cases(
    *,
    current_sizes: Sequence[int] = DEFAULT_CURRENT_SIZES,
    age_shifts: Sequence[float] = DEFAULT_AGE_SHIFTS,
    missing_fractions: Sequence[float] = DEFAULT_MISSING_FRACTIONS,
    reference_missing_fraction: float = DEFAULT_REFERENCE_MISSING_FRACTION,
    acceptance_age_shift: float = DEFAULT_ACCEPTANCE_AGE_SHIFT,
    acceptance_missing_fraction: float = DEFAULT_ACCEPTANCE_MISSING_FRACTION,
) -> tuple[CalibrationCase, ...]:
    """Build the deterministic control/weak/strong calibration grid."""

    sizes = _require_positive_ints(current_sizes, "current_sizes")
    shifts = _require_positive_floats(age_shifts, "age_shifts")
    fractions = _require_missing_fractions(
        missing_fractions, reference_missing_fraction
    )
    acceptance_age_shift = float(acceptance_age_shift)
    acceptance_missing_fraction = float(acceptance_missing_fraction)
    if not isfinite(acceptance_age_shift) or acceptance_age_shift <= 0.0:
        raise ValueError("acceptance_age_shift должен быть положительным")
    if (
        not isfinite(acceptance_missing_fraction)
        or not reference_missing_fraction < acceptance_missing_fraction <= 1.0
    ):
        raise ValueError(
            "acceptance_missing_fraction должен быть больше контрольной доли "
            "пропусков и не превышать 1"
        )

    cases: list[CalibrationCase] = []
    for n_current in sizes:
        cases.append(
            CalibrationCase(
                case_id=f"control_n{n_current}",
                scenario="none",
                drift_family="control",
                drift_level="control",
                n_current=n_current,
                age_shift_years=0.0,
                current_missing_fraction=reference_missing_fraction,
            )
        )
        for index, shift in enumerate(shifts):
            level = _level_name(index, len(shifts))
            cases.append(
                CalibrationCase(
                    case_id=(
                        f"numeric_{level}_shift{_number_token(shift)}_n{n_current}"
                    ),
                    scenario="numeric",
                    drift_family="numeric",
                    drift_level=level,
                    n_current=n_current,
                    age_shift_years=shift,
                    current_missing_fraction=reference_missing_fraction,
                )
            )
        for index, fraction in enumerate(fractions):
            level = _level_name(index, len(fractions))
            cases.append(
                CalibrationCase(
                    case_id=(
                        f"missingness_{level}_fraction"
                        f"{_number_token(fraction)}_n{n_current}"
                    ),
                    scenario="missingness",
                    drift_family="missingness",
                    drift_level=level,
                    n_current=n_current,
                    age_shift_years=0.0,
                    current_missing_fraction=fraction,
                )
            )
        cases.append(
            CalibrationCase(
                case_id=f"categorical_standard_n{n_current}",
                scenario="categorical",
                drift_family="categorical",
                drift_level="standard",
                n_current=n_current,
                age_shift_years=0.0,
                current_missing_fraction=reference_missing_fraction,
            )
        )
        cases.append(
            CalibrationCase(
                case_id=f"combined_acceptance_n{n_current}",
                scenario="combined",
                drift_family="combined",
                drift_level="acceptance",
                n_current=n_current,
                age_shift_years=float(acceptance_age_shift),
                current_missing_fraction=float(acceptance_missing_fraction),
            )
        )
    return tuple(cases)


def _expected_signal(
    case: CalibrationCase,
    *,
    source: str,
    feature: str | None,
    method: str,
) -> bool:
    if case.scenario == "none":
        return False
    if source == "adversarial":
        return True
    numeric_target = source == "drift" and feature == "age"
    categorical_target = source == "drift" and feature == "region"
    missingness_target = (
        source == "quality"
        and feature == "income"
        and method in {"current_missing_fraction", "missing_increase_pp"}
    )
    if case.scenario == "numeric":
        return numeric_target
    if case.scenario == "categorical":
        return categorical_target
    if case.scenario == "missingness":
        return missingness_target
    return numeric_target or categorical_target or missingness_target


def _check_row(
    check: CheckResult,
    *,
    source: str,
    feature: str | None,
) -> dict[str, Any]:
    details = check["details"]
    return {
        "source": source,
        "feature": feature,
        "method": check["name"],
        "status": check["status"],
        "value": check["value"],
        "threshold": check["threshold"],
        "p_value": check["p_value"],
        "adjusted_p_value": check["adjusted_p_value"],
        "alert": check["alert"],
        "threshold_source": details.get("threshold_source"),
        "cramers_v": details.get("cramers_v"),
        "split_strategy": None,
        "reason": check["reason"],
    }


def result_rows(
    result: AnalysisResult,
    *,
    case: CalibrationCase,
    seed: int,
    n_reference: int,
    reference_missing_fraction: float,
    elapsed_seconds: float,
    commit_sha: str,
    config_sha256: str,
) -> list[dict[str, Any]]:
    """Flatten one public ``AnalysisResult`` into machine-readable decisions."""

    check_rows: list[dict[str, Any]] = []
    check_rows.extend(
        _check_row(check, source="quality", feature=None)
        for check in result["quality"]["dataset_checks"]
    )
    for feature, checks in result["quality"]["feature_checks"].items():
        check_rows.extend(
            _check_row(check, source="quality", feature=feature) for check in checks
        )
    for feature, feature_result in result["drift"]["features"].items():
        check_rows.extend(
            _check_row(check, source="drift", feature=feature)
            for check in feature_result["checks"].values()
        )

    adversarial = result["adversarial"]
    check_rows.append(
        {
            "source": "adversarial",
            "feature": None,
            "method": "roc_auc",
            "status": adversarial["status"],
            "value": adversarial["roc_auc"],
            "threshold": adversarial["threshold"],
            "p_value": None,
            "adjusted_p_value": None,
            "alert": adversarial["alert"],
            "threshold_source": (
                "global" if adversarial["threshold"] is not None else "not_configured"
            ),
            "cramers_v": None,
            "split_strategy": adversarial["split_strategy"],
            "reason": adversarial["reason"],
        }
    )

    base = {
        **asdict(case),
        "seed": seed,
        "n_reference": n_reference,
        "reference_missing_fraction": reference_missing_fraction,
        "analysis_status": result["summary"]["status"],
        "analysis_alerts": result["summary"]["n_alerts"],
        "elapsed_seconds": elapsed_seconds,
        "commit_sha": commit_sha,
        "config_sha256": config_sha256,
    }
    rows: list[dict[str, Any]] = []
    for check_row in check_rows:
        rows.append(
            {
                **base,
                **check_row,
                "expected_signal": _expected_signal(
                    case,
                    source=check_row["source"],
                    feature=check_row["feature"],
                    method=check_row["method"],
                ),
            }
        )
    return rows


def run_calibration(
    *,
    cases: Sequence[CalibrationCase],
    seeds: Sequence[int],
    n_reference: int,
    reference_missing_fraction: float,
    config: AnalysisConfig,
    commit_sha: str,
    config_sha256: str,
    progress: bool = False,
) -> pd.DataFrame:
    """Execute the grid with the public API and return long-form check rows."""

    normalized_seeds = _require_non_negative_ints(seeds, "seeds")
    if n_reference <= 0:
        raise ValueError("n_reference должен быть положительным")
    if not cases:
        raise ValueError("cases не должен быть пустым")

    rows: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases, start=1):
        for seed in normalized_seeds:
            generated = generate_demo_data(
                seed=seed,
                n_reference=n_reference,
                n_current=case.n_current,
                scenario=case.scenario,
                age_shift_years=case.age_shift_years,
                reference_income_missing_fraction=reference_missing_fraction,
                drifted_income_missing_fraction=case.current_missing_fraction,
            )
            started = perf_counter()
            result = analyze(generated.reference, generated.current, config=config)
            elapsed = perf_counter() - started
            rows.extend(
                result_rows(
                    result,
                    case=case,
                    seed=seed,
                    n_reference=n_reference,
                    reference_missing_fraction=reference_missing_fraction,
                    elapsed_seconds=elapsed,
                    commit_sha=commit_sha,
                    config_sha256=config_sha256,
                )
            )
        if progress:
            print(
                f"[{case_index}/{len(cases)}] {case.case_id}: "
                f"{len(normalized_seeds)} seed завершены",
                flush=True,
            )
    return pd.DataFrame(rows)


def summarize_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """Aggregate observed alert rates without claiming population guarantees."""

    required = {
        "case_id",
        "scenario",
        "drift_family",
        "drift_level",
        "n_current",
        "age_shift_years",
        "current_missing_fraction",
        "source",
        "feature",
        "method",
        "expected_signal",
        "seed",
        "alert",
        "value",
        "commit_sha",
        "config_sha256",
    }
    missing = sorted(required - set(runs.columns))
    if missing:
        raise ValueError("runs не содержит столбцы: " + ", ".join(missing))

    group_columns = [
        "case_id",
        "scenario",
        "drift_family",
        "drift_level",
        "n_current",
        "age_shift_years",
        "current_missing_fraction",
        "source",
        "feature",
        "method",
        "expected_signal",
        "commit_sha",
        "config_sha256",
    ]
    summary_rows: list[dict[str, Any]] = []
    grouped = runs.groupby(group_columns, dropna=False, sort=True)
    for group_key, group in grouped:
        record = dict(zip(group_columns, group_key, strict=True))
        decisions = group[group["alert"].isin([True, False])]
        n_decisions = len(decisions)
        n_alerts = sum(value is True for value in decisions["alert"])
        numeric_values = pd.to_numeric(group["value"], errors="coerce").dropna()
        if record["scenario"] == "none":
            rate_kind = "false_alert_rate"
        elif bool(record["expected_signal"]):
            rate_kind = "detection_rate"
        else:
            rate_kind = "non_target_alert_rate"
        summary_rows.append(
            {
                **record,
                "rate_kind": rate_kind,
                "seeds": "|".join(str(value) for value in sorted(group["seed"].unique())),
                "n_runs": int(group["seed"].nunique()),
                "n_decisions": n_decisions,
                "n_alerts": n_alerts,
                "observed_rate": (
                    float(n_alerts / n_decisions) if n_decisions else None
                ),
                "mean_value": (
                    float(numeric_values.mean()) if not numeric_values.empty else None
                ),
                "median_value": (
                    float(numeric_values.median()) if not numeric_values.empty else None
                ),
            }
        )
    return pd.DataFrame(summary_rows)


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in PACKAGE_NAMES:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def _git_metadata(project_root: Path) -> dict[str, Any]:
    def command(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
        except (FileNotFoundError, subprocess.CalledProcessError, UnicodeError):
            return None
        return completed.stdout.strip()

    status = command("status", "--porcelain")
    return {
        "commit_sha": command("rev-parse", "HEAD") or "unknown",
        "branch": command("branch", "--show-current") or "detached-or-unknown",
        "dirty": None if status is None else bool(status),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_metadata(
    *,
    config_path: Path,
    config_sha256: str,
    adversarial_enabled: bool,
    git: dict[str, Any],
    seeds: Sequence[int],
    cases: Sequence[CalibrationCase],
    n_reference: int,
    reference_missing_fraction: float,
    elapsed_seconds: float,
    runs: pd.DataFrame,
    summary: pd.DataFrame,
) -> dict[str, Any]:
    control_runs = len(
        runs.loc[runs["scenario"] == "none", ["case_id", "seed"]].drop_duplicates()
    )
    return {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "demo-specific synthetic calibration",
        "warning": (
            "Наблюдаемые доли являются первичной инженерной оценкой на "
            "синтетической сетке, а не гарантией для будущих production-данных."
        ),
        "git": git,
        "config": {
            "path": str(config_path.as_posix()),
            "sha256": config_sha256,
            "adversarial_enabled": adversarial_enabled,
        },
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "packages": _package_versions(),
        },
        "grid": {
            "seeds": list(seeds),
            "n_reference": n_reference,
            "reference_missing_fraction": reference_missing_fraction,
            "cases": [asdict(case) for case in cases],
        },
        "execution": {
            "elapsed_seconds": elapsed_seconds,
            "case_count": len(cases),
            "analysis_run_count": len(cases) * len(seeds),
            "control_run_count": control_runs,
            "minimum_control_runs_satisfied": control_runs >= 20,
            "decision_row_count": len(runs),
            "summary_row_count": len(summary),
        },
        "artifacts": {
            "runs": RUNS_FILENAME,
            "summary": SUMMARY_FILENAME,
            "metadata": METADATA_FILENAME,
        },
    }


def write_calibration_outputs(
    *,
    output_dir: Path,
    runs: pd.DataFrame,
    summary: pd.DataFrame,
    metadata: dict[str, Any],
    overwrite: bool = False,
) -> dict[str, Path]:
    """Atomically write the three calibration artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "runs": output_dir / RUNS_FILENAME,
        "summary": output_dir / SUMMARY_FILENAME,
        "metadata": output_dir / METADATA_FILENAME,
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Калибровочные артефакты уже существуют: {joined}. "
            "Передайте --overwrite для явной замены."
        )

    temporary_paths = {
        name: path.with_suffix(path.suffix + ".tmp")
        for name, path in paths.items()
    }
    try:
        runs.to_csv(temporary_paths["runs"], index=False, lineterminator="\n")
        summary.to_csv(
            temporary_paths["summary"], index=False, lineterminator="\n"
        )
        temporary_paths["metadata"].write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        for name, path in paths.items():
            temporary_paths[name].replace(path)
    finally:
        for path in temporary_paths.values():
            path.unlink(missing_ok=True)
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Выполнить воспроизводимую сетку калибровочных экспериментов "
            "через публичный analyze()."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--current-sizes", nargs="+", type=int, default=list(DEFAULT_CURRENT_SIZES)
    )
    parser.add_argument(
        "--age-shifts", nargs="+", type=float, default=list(DEFAULT_AGE_SHIFTS)
    )
    parser.add_argument(
        "--missing-fractions",
        nargs="+",
        type=float,
        default=list(DEFAULT_MISSING_FRACTIONS),
    )
    parser.add_argument("--n-reference", type=int, default=DEFAULT_N_REFERENCE)
    parser.add_argument(
        "--reference-missing-fraction",
        type=float,
        default=DEFAULT_REFERENCE_MISSING_FRACTION,
    )
    parser.add_argument(
        "--acceptance-age-shift",
        type=float,
        default=DEFAULT_ACCEPTANCE_AGE_SHIFT,
    )
    parser.add_argument(
        "--acceptance-missing-fraction",
        type=float,
        default=DEFAULT_ACCEPTANCE_MISSING_FRACTION,
    )
    parser.add_argument(
        "--disable-adversarial",
        action="store_true",
        help="Отключить тяжёлый ML-блок только для диагностического быстрого прогона.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        seeds = _require_non_negative_ints(args.seeds, "seeds")
        cases = build_cases(
            current_sizes=args.current_sizes,
            age_shifts=args.age_shifts,
            missing_fractions=args.missing_fractions,
            reference_missing_fraction=args.reference_missing_fraction,
            acceptance_age_shift=args.acceptance_age_shift,
            acceptance_missing_fraction=args.acceptance_missing_fraction,
        )
        if args.n_reference <= 0:
            raise ValueError("n_reference должен быть положительным")

        config_path = args.config.resolve()
        config = load_config(config_path)
        if args.disable_adversarial:
            config["adversarial"]["enabled"] = False
        config_sha256 = _sha256(config_path)
        git = _git_metadata(PROJECT_ROOT)

        print(
            f"Калибровка: {len(cases)} случаев × {len(seeds)} seed = "
            f"{len(cases) * len(seeds)} анализов"
        )
        started = perf_counter()
        runs = run_calibration(
            cases=cases,
            seeds=seeds,
            n_reference=args.n_reference,
            reference_missing_fraction=args.reference_missing_fraction,
            config=config,
            commit_sha=git["commit_sha"],
            config_sha256=config_sha256,
            progress=not args.quiet,
        )
        summary = summarize_runs(runs)
        elapsed = perf_counter() - started
        metadata = build_metadata(
            config_path=args.config,
            config_sha256=config_sha256,
            adversarial_enabled=config["adversarial"]["enabled"],
            git=git,
            seeds=seeds,
            cases=cases,
            n_reference=args.n_reference,
            reference_missing_fraction=args.reference_missing_fraction,
            elapsed_seconds=elapsed,
            runs=runs,
            summary=summary,
        )
        paths = write_calibration_outputs(
            output_dir=args.output_dir,
            runs=runs,
            summary=summary,
            metadata=metadata,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, TypeError, ValueError) as exc:
        print(f"Ошибка калибровки: {exc}", file=sys.stderr)
        return 1

    print(f"Готово за {elapsed:.2f} с")
    print(f"Строк решений: {len(runs)}")
    print(f"Строк сводки: {len(summary)}")
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
