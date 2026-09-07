"""Стартовый контракт 0.1. Ответственные: Михаил и Павел.

TypedDict описывает словарь, но сам по себе не выполняет runtime-валидацию.
Семантика полей и статусов: docs/contracts.md.
"""

from __future__ import annotations

from typing import Any, Literal, NotRequired, TypedDict

Status = Literal["ok", "warning", "critical", "skipped", "error"]
FeatureType = Literal["numeric", "categorical"]
AlertSource = Literal["schema", "quality", "drift", "adversarial"]
MultipleTestingMethod = Literal["none", "bh"]


class FeatureConfig(TypedDict):
    """Семантическая схема одного признака.

    ``kind`` задаёт семейство допустимых pandas dtype. Точный физический dtype
    фиксируется в отчёте, но не требуется совпадение int64 с float64.
    """

    kind: FeatureType
    nullable: bool
    min: NotRequired[float | int]
    max: NotRequired[float | int]


class SchemaConfig(TypedDict):
    allow_extra_columns: bool
    features: dict[str, FeatureConfig]


class QualityConfig(TypedDict):
    max_missing_fraction: float
    max_missing_increase_pp: float
    max_duplicate_fraction: float


class DistanceThresholds(TypedDict):
    wasserstein: float | None
    psi: float | None
    js: float | None


class DriftConfig(TypedDict):
    numeric_methods: list[Literal["ks", "wasserstein", "psi", "js"]]
    categorical_methods: list[Literal["chi2", "psi", "js"]]
    alpha: float
    multiple_testing: MultipleTestingMethod
    n_bins: int
    psi_smoothing: float
    js_base: float
    distance_thresholds: DistanceThresholds


class AdversarialConfig(TypedDict):
    enabled: bool
    n_splits: int
    roc_auc_threshold: float | None
    exclude_columns: list[str]


class AnalysisConfig(TypedDict):
    contract_version: str
    random_seed: int
    schema: SchemaConfig
    quality: QualityConfig
    drift: DriftConfig
    adversarial: AdversarialConfig


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
