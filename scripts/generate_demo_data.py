"""Generate reproducible reference/current datasets for the demo pipeline.

The two batches are sampled independently. A scenario changes only the current
batch, so ``none`` is a genuine no-drift control rather than a copy of the
reference data.

Run from the repository root, for example::

    python scripts/generate_demo_data.py --scenario none --format both
    python scripts/generate_demo_data.py --scenario combined --format both
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

Scenario = Literal["none", "numeric", "categorical", "missingness", "combined"]
OutputFormat = Literal["csv", "parquet"]

SCENARIOS: tuple[Scenario, ...] = (
    "none",
    "numeric",
    "categorical",
    "missingness",
    "combined",
)
REGIONS: tuple[str, ...] = ("central", "northwest", "south", "volga", "siberia")
BASE_REGION_PROBABILITIES = np.array([0.30, 0.15, 0.20, 0.20, 0.15])
DRIFTED_REGION_PROBABILITIES = np.array([0.55, 0.10, 0.10, 0.15, 0.10])

DEFAULT_SEED = 42
DEFAULT_N_REFERENCE = 5_000
DEFAULT_N_CURRENT = 3_000
DEFAULT_AGE_SHIFT_YEARS = 8.0
DEFAULT_REFERENCE_INCOME_MISSING_FRACTION = 0.02
DEFAULT_DRIFTED_INCOME_MISSING_FRACTION = 0.08


@dataclass(frozen=True)
class GeneratedData:
    """In-memory output of one deterministic generation run."""

    reference: pd.DataFrame
    current: pd.DataFrame
    metadata: dict[str, Any]


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _validate_fraction(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {value}")


def _apply_exact_missing_fraction(
    frame: pd.DataFrame,
    *,
    column: str,
    fraction: float,
    rng: np.random.Generator,
) -> None:
    """Set an exact, reproducibly selected fraction of a column to NaN."""

    n_missing = round(len(frame) * fraction)
    if n_missing == 0:
        return
    missing_indices = rng.permutation(len(frame))[:n_missing]
    frame.loc[missing_indices, column] = np.nan


def _generate_batch(
    *,
    seed_sequence: np.random.SeedSequence,
    n_rows: int,
    region_probabilities: np.ndarray,
    income_missing_fraction: float,
) -> pd.DataFrame:
    age_seed, income_seed, region_seed, missing_seed = seed_sequence.spawn(4)
    age_rng = np.random.default_rng(age_seed)
    income_rng = np.random.default_rng(income_seed)
    region_rng = np.random.default_rng(region_seed)
    missing_rng = np.random.default_rng(missing_seed)

    age = np.rint(np.clip(age_rng.normal(loc=42.0, scale=12.0, size=n_rows), 18, 100))
    income = np.clip(
        income_rng.lognormal(mean=np.log(85_000), sigma=0.55, size=n_rows),
        0,
        1_000_000,
    ).round(2)
    region = region_rng.choice(REGIONS, size=n_rows, p=region_probabilities)

    frame = pd.DataFrame(
        {
            # Float64 is intentional: CSV represents nullable numeric columns
            # consistently after missing values are introduced.
            "age": age.astype("float64"),
            "income": income.astype("float64"),
            "region": pd.Series(region, dtype="object"),
        }
    )
    _apply_exact_missing_fraction(
        frame,
        column="income",
        fraction=income_missing_fraction,
        rng=missing_rng,
    )
    return frame


def generate_demo_data(
    *,
    seed: int = DEFAULT_SEED,
    n_reference: int = DEFAULT_N_REFERENCE,
    n_current: int = DEFAULT_N_CURRENT,
    scenario: Scenario = "none",
    age_shift_years: float = DEFAULT_AGE_SHIFT_YEARS,
    reference_income_missing_fraction: float = (
        DEFAULT_REFERENCE_INCOME_MISSING_FRACTION
    ),
    drifted_income_missing_fraction: float = (DEFAULT_DRIFTED_INCOME_MISSING_FRACTION),
) -> GeneratedData:
    """Build independent reference/current batches for a selected scenario."""

    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; expected one of {SCENARIOS}")
    if n_reference <= 0 or n_current <= 0:
        raise ValueError("n_reference and n_current must be positive")
    _validate_fraction(
        "reference_income_missing_fraction", reference_income_missing_fraction
    )
    _validate_fraction(
        "drifted_income_missing_fraction", drifted_income_missing_fraction
    )

    has_numeric_drift = scenario in {"numeric", "combined"}
    has_categorical_drift = scenario in {"categorical", "combined"}
    has_missingness_drift = scenario in {"missingness", "combined"}

    root_seed = np.random.SeedSequence(seed)
    reference_seed, current_seed = root_seed.spawn(2)

    reference = _generate_batch(
        seed_sequence=reference_seed,
        n_rows=n_reference,
        region_probabilities=BASE_REGION_PROBABILITIES,
        income_missing_fraction=reference_income_missing_fraction,
    )
    current_region_probabilities = (
        DRIFTED_REGION_PROBABILITIES
        if has_categorical_drift
        else BASE_REGION_PROBABILITIES
    )
    current_missing_fraction = (
        drifted_income_missing_fraction
        if has_missingness_drift
        else reference_income_missing_fraction
    )
    current = _generate_batch(
        seed_sequence=current_seed,
        n_rows=n_current,
        region_probabilities=current_region_probabilities,
        income_missing_fraction=current_missing_fraction,
    )

    if has_numeric_drift:
        current["age"] = (current["age"] + age_shift_years).clip(lower=18, upper=100)

    metadata: dict[str, Any] = {
        "contract_version": "0.1",
        "scenario": scenario,
        "seed": seed,
        "n_reference": n_reference,
        "n_current": n_current,
        "independent_batches": True,
        "applied_drift": {
            "numeric": {
                "enabled": has_numeric_drift,
                "feature": "age",
                "shift_years": age_shift_years if has_numeric_drift else 0.0,
            },
            "categorical": {
                "enabled": has_categorical_drift,
                "feature": "region",
                "reference_probabilities": dict(
                    zip(REGIONS, BASE_REGION_PROBABILITIES.tolist(), strict=True)
                ),
                "current_probabilities": dict(
                    zip(REGIONS, current_region_probabilities.tolist(), strict=True)
                ),
            },
            "missingness": {
                "enabled": has_missingness_drift,
                "feature": "income",
                "reference_fraction": reference_income_missing_fraction,
                "current_fraction": current_missing_fraction,
                "increase_percentage_points": 100
                * (current_missing_fraction - reference_income_missing_fraction),
            },
        },
        "schema": {
            "age": {"dtype": str(reference["age"].dtype), "unit": "years"},
            "income": {
                "dtype": str(reference["income"].dtype),
                "unit": "currency_units_per_month",
            },
            "region": {"dtype": str(reference["region"].dtype), "unit": None},
        },
        "actual_missing_fraction": {
            "reference_income": float(reference["income"].isna().mean()),
            "current_income": float(current["income"].isna().mean()),
        },
    }
    return GeneratedData(reference=reference, current=current, metadata=metadata)


def write_generated_data(
    generated: GeneratedData,
    *,
    output_dir: Path,
    formats: Sequence[OutputFormat],
) -> dict[str, str]:
    """Write a scenario into its own directory and return relative file paths."""

    normalized_formats = tuple(dict.fromkeys(formats))
    if not normalized_formats:
        raise ValueError("at least one output format is required")
    unsupported = set(normalized_formats) - {"csv", "parquet"}
    if unsupported:
        raise ValueError(f"unsupported output formats: {sorted(unsupported)}")

    scenario_dir = output_dir / str(generated.metadata["scenario"])
    scenario_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    for output_format in normalized_formats:
        reference_path = scenario_dir / f"reference.{output_format}"
        current_path = scenario_dir / f"current.{output_format}"
        if output_format == "csv":
            generated.reference.to_csv(reference_path, index=False)
            generated.current.to_csv(current_path, index=False)
        else:
            generated.reference.to_parquet(reference_path, index=False)
            generated.current.to_parquet(current_path, index=False)
        paths[f"reference_{output_format}"] = str(
            reference_path.relative_to(output_dir)
        )
        paths[f"current_{output_format}"] = str(current_path.relative_to(output_dir))

    metadata_path = scenario_dir / "metadata.json"
    persisted_metadata = dict(generated.metadata)
    persisted_metadata["files"] = paths
    metadata_path.write_text(
        json.dumps(persisted_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    paths["metadata"] = str(metadata_path.relative_to(output_dir))
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=SCENARIOS, default="none")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--n-reference", type=_positive_int, default=DEFAULT_N_REFERENCE
    )
    parser.add_argument("--n-current", type=_positive_int, default=DEFAULT_N_CURRENT)
    parser.add_argument(
        "--age-shift-years", type=float, default=DEFAULT_AGE_SHIFT_YEARS
    )
    parser.add_argument(
        "--reference-income-missing-fraction",
        type=float,
        default=DEFAULT_REFERENCE_INCOME_MISSING_FRACTION,
    )
    parser.add_argument(
        "--drifted-income-missing-fraction",
        type=float,
        default=DEFAULT_DRIFTED_INCOME_MISSING_FRACTION,
    )
    parser.add_argument("--format", choices=("csv", "parquet", "both"), default="both")
    parser.add_argument("--output-dir", type=Path, default=Path("data/generated"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    formats: tuple[OutputFormat, ...] = (
        ("csv", "parquet") if args.format == "both" else (args.format,)
    )
    generated = generate_demo_data(
        seed=args.seed,
        n_reference=args.n_reference,
        n_current=args.n_current,
        scenario=args.scenario,
        age_shift_years=args.age_shift_years,
        reference_income_missing_fraction=args.reference_income_missing_fraction,
        drifted_income_missing_fraction=args.drifted_income_missing_fraction,
    )
    paths = write_generated_data(
        generated,
        output_dir=args.output_dir,
        formats=formats,
    )

    print(
        f"Generated scenario={args.scenario!r}, seed={args.seed}, "
        f"reference_rows={args.n_reference}, current_rows={args.n_current}"
    )
    for name, path in paths.items():
        print(f"{name}: {args.output_dir / path}")


if __name__ == "__main__":
    main()
