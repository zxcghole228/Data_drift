"""Типы конфигурации и записей постоянного online-состояния."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from ..contracts import AnalysisConfig

WindowStrategy = Literal["tumbling"]
ReferenceFormat = Literal["csv", "parquet"]
EventState = Literal["buffered", "claimed", "processed", "failed", "discarded"]
BatchState = Literal["accepted", "processing", "completed", "failed"]
RunSource = Literal["window", "batch"]
RunStatus = Literal["running", "completed", "failed"]
DeliveryChannel = Literal["jsonl", "webhook"]
DeliveryStatus = Literal["pending", "succeeded", "failed"]
PerformanceStatus = Literal["evaluated", "not_evaluated"]


class OnlineWindowConfig(TypedDict):
    strategy: WindowStrategy
    size: int
    min_batch_rows: int


class OnlineAlertConfig(TypedDict):
    jsonl_path: str | None
    webhook_url_env: str | None
    timeout_seconds: float
    max_attempts: int


class OnlinePerformanceConfig(TypedDict):
    enabled: bool
    min_feedback_rows: int
    baseline_accuracy: float | None
    max_accuracy_drop: float | None
    baseline_roc_auc: float | None
    max_roc_auc_drop: float | None


class OnlineConfig(TypedDict):
    enabled: bool
    state_path: str
    max_request_bytes: int
    window: OnlineWindowConfig
    alerts: OnlineAlertConfig
    performance: OnlinePerformanceConfig


class MonitoringConfig(TypedDict):
    analysis: AnalysisConfig
    online: OnlineConfig


class EventIngestionResult(TypedDict):
    accepted: int
    duplicates: int
    buffered_rows: int
    run_ids: list[str]


class FeedbackIngestionResult(TypedDict):
    accepted: int
    duplicates: int
    evaluations: list[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class ReferenceRecord:
    reference_id: str
    name: str
    content_sha256: str
    config_sha256: str
    source_format: ReferenceFormat
    row_count: int
    columns: list[str]
    dtypes: dict[str, str]
    content: bytes
    created_at: str
    active: bool


@dataclass(frozen=True, slots=True)
class EventRecord:
    sequence: int
    event_id: str
    payload_sha256: str
    reference_id: str
    occurred_at: str
    received_at: str
    features: dict[str, Any]
    prediction: Any | None
    score: float | None
    state: EventState
    window_id: str | None


@dataclass(frozen=True, slots=True)
class BatchRecord:
    batch_id: str
    payload_sha256: str
    reference_id: str
    received_at: str
    records: list[dict[str, Any]]
    row_count: int
    state: BatchState
    run_id: str | None


@dataclass(frozen=True, slots=True)
class RunRecord:
    run_id: str
    source: RunSource
    source_id: str
    reference_id: str
    config_sha256: str
    row_count: int
    status: RunStatus
    created_at: str
    completed_at: str | None
    analysis_result: dict[str, Any] | None
    error: str | None


@dataclass(frozen=True, slots=True)
class DeliveryRecord:
    delivery_id: str
    run_id: str
    channel: DeliveryChannel
    status: DeliveryStatus
    attempts: int
    payload: dict[str, Any]
    last_error: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class FeedbackRecord:
    event_id: str
    payload_sha256: str
    y_true: Any
    received_at: str


@dataclass(frozen=True, slots=True)
class PerformanceSample:
    event_id: str
    prediction: Any | None
    score: float | None
    y_true: Any


@dataclass(frozen=True, slots=True)
class PerformanceRecord:
    run_id: str
    status: PerformanceStatus
    result: dict[str, Any]
    created_at: str
    updated_at: str
