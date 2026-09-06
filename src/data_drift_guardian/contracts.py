"""Стартовый контракт 0.1. Ответственные: Михаил и Павел.

TypedDict описывает словарь, но сам по себе не выполняет runtime-валидацию.
Семантика полей и статусов: docs/contracts.md.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

Status = Literal["ok", "warning", "critical", "skipped", "error"]
FeatureType = Literal["numeric", "categorical"]
AlertSource = Literal["schema", "quality", "drift", "adversarial"]


class CheckResult(TypedDict):
    name: str
    status: Status
    value: float | int | None
    threshold: float | int | None
    p_value: float | None
    adjusted_p_value: float | None
    alert: bool | None
    reason: str | None
    details: dict[str, Any]


class TypeMismatch(TypedDict):
    column: str
    dataset: Literal["reference", "current"]
    expected: str
    actual: str


class SchemaResult(TypedDict):
    status: Status
    missing_columns: dict[str, list[str]]
    extra_columns: dict[str, list[str]]
    duplicate_columns: dict[str, list[str]]
    type_mismatches: list[TypeMismatch]
    valid_features: list[str]
    reason: str | None


class QualityResult(TypedDict):
    status: Status
    dataset_checks: list[CheckResult]
    feature_checks: dict[str, list[CheckResult]]
    reason: str | None


class FeatureResult(TypedDict):
    feature_type: FeatureType
    status: Status
    n_reference_valid: int
    n_current_valid: int
    checks: dict[str, CheckResult]
    alert: bool | None
    reason: str | None


class DriftResult(TypedDict):
    status: Status
    features: dict[str, FeatureResult]
    reason: str | None


class AdversarialResult(TypedDict):
    status: Status
    roc_auc: float | None
    fold_auc: list[float]
    feature_importance: dict[str, float]
    importance_type: str | None
    alert: bool | None
    reason: str | None


class Alert(TypedDict):
    source: AlertSource
    feature: str | None
    check: str
    severity: Literal["warning", "critical"]
    message: str


class Metadata(TypedDict):
    reference_rows: int
    current_rows: int
    random_seed: int


class Summary(TypedDict):
    status: Status
    has_alerts: bool
    n_alerts: int
    analyzed_features: int
    skipped_features: int


class AnalysisResult(TypedDict):
    contract_version: str
    metadata: Metadata
    schema: SchemaResult
    quality: QualityResult
    drift: DriftResult
    adversarial: AdversarialResult
    alerts: list[Alert]
    summary: Summary
