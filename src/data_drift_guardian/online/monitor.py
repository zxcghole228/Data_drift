"""Оркестрация Reference, событийных окон и готовых online-батчей."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import pandas as pd

from ..config import validate_config
from ..ingestion import SUPPORTED_EXTENSIONS, load_table
from ..pipeline import analyze
from ..schema import validate_schema
from .alerts import AlertDeliveryService
from .config import validate_monitoring_config
from .contracts import (
    DeliveryRecord,
    EventIngestionResult,
    EventRecord,
    FeedbackIngestionResult,
    MonitoringConfig,
    PerformanceRecord,
    ReferenceRecord,
    RunRecord,
)
from .performance import evaluate_performance
from .storage import MonitoringStore, canonical_json, sha256_json

_EVENT_REQUIRED_KEYS = frozenset({"event_id", "occurred_at", "features"})
_EVENT_OPTIONAL_KEYS = frozenset({"prediction", "score"})
_FEEDBACK_KEYS = frozenset({"event_id", "y_true"})


class OnlineMonitorError(RuntimeError):
    """Базовая ошибка оркестрации online-мониторинга."""


class NoActiveReferenceError(OnlineMonitorError):
    """Обработка невозможна без активной базовой выборки."""


class ReferenceConfigMismatchError(OnlineMonitorError):
    """Активный Reference зарегистрирован с другой конфигурацией анализа."""


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def _validated_monitoring_config(config: Mapping[str, Any]) -> MonitoringConfig:
    if set(config) == {"analysis", "online"}:
        analysis = config["analysis"]
        online = config["online"]
        if not isinstance(analysis, Mapping) or not isinstance(online, Mapping):
            raise ValueError("config.analysis и config.online должны быть отображениями")
        raw = {**analysis, "online": online}
        return validate_monitoring_config(raw)
    return validate_monitoring_config(config)


class OnlineMonitor:
    """Синхронный single-worker процессор persistent micro-batch потока."""

    def __init__(
        self,
        store: MonitoringStore,
        config: Mapping[str, Any],
    ) -> None:
        if not isinstance(store, MonitoringStore):
            raise TypeError("store должен иметь тип MonitoringStore")
        self.store = store
        self.config = _validated_monitoring_config(config)
        self.analysis_config = validate_config(self.config["analysis"])
        self.online_config = deepcopy(self.config["online"])
        self.config_sha256 = sha256_json(self.analysis_config)
        self.alert_delivery = AlertDeliveryService(
            self.store,
            self.online_config["alerts"],
        )

    def _deliver_alerts(self, run: RunRecord) -> None:
        """Не позволить ошибке внешнего канала изменить успешный Run."""

        try:
            self.alert_delivery.dispatch_run(run)
        except Exception:
            # Ожидаемые ошибки каналов сохраняются самим delivery service.
            # Защитная граница оставляет уже сохранённый анализ успешным.
            return

    def _evaluate_performance(self, run: RunRecord) -> PerformanceRecord | None:
        if run.status != "completed" or run.source != "window":
            return None
        samples = self.store.list_feedback_samples(run.source_id)
        status, result = evaluate_performance(
            samples,
            self.online_config["performance"],
        )
        return self.store.save_performance(
            run_id=run.run_id,
            status=status,
            result=result,
        )

    def _require_compatible_reference(
        self,
        reference: ReferenceRecord,
    ) -> ReferenceRecord:
        if reference.config_sha256 != self.config_sha256:
            raise ReferenceConfigMismatchError(
                "Reference зарегистрирован с другой analysis-конфигурацией: "
                f"reference={reference.config_sha256}, current={self.config_sha256}"
            )
        return reference

    def _active_reference(self) -> ReferenceRecord:
        reference = self.store.get_active_reference()
        if reference is None:
            raise NoActiveReferenceError(
                "Online-мониторинг не готов: активный Reference отсутствует"
            )
        return self._require_compatible_reference(reference)

    def get_active_reference(self) -> ReferenceRecord:
        """Вернуть готовый к анализу активный Reference или понятную ошибку."""

        return self._active_reference()

    @staticmethod
    def _reference_frame(reference: ReferenceRecord) -> pd.DataFrame:
        stream = BytesIO(reference.content)
        if reference.source_format == "csv":
            return pd.read_csv(stream)
        return pd.read_parquet(stream)

    def register_reference(
        self,
        path: str | Path,
        *,
        name: str | None = None,
        reference_id: str | None = None,
        activate: bool = False,
        discard_buffered: bool = False,
    ) -> ReferenceRecord:
        """Проверить, сохранить и при необходимости активировать Reference."""

        if not isinstance(path, (str, Path)):
            raise TypeError("path должен иметь тип str или pathlib.Path")
        if type(activate) is not bool or type(discard_buffered) is not bool:
            raise TypeError("activate и discard_buffered должны иметь тип bool")

        reference_path = Path(path).expanduser()
        frame = load_table(reference_path)
        schema = validate_schema(
            frame,
            frame,
            self.analysis_config["schema"],
        )
        if schema["status"] == "error":
            raise ValueError(
                "Reference не соответствует настроенной схеме: "
                f"{schema['reason'] or 'неизвестная ошибка схемы'}"
            )
        if any(not isinstance(column, str) for column in frame.columns):
            raise ValueError("Все имена колонок Reference должны быть строками")

        extension = reference_path.suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:  # защищено load_table
            raise ValueError(f"Неподдерживаемый формат Reference: {extension}")
        source_format = "csv" if extension == ".csv" else "parquet"
        normalized_name = reference_path.stem if name is None else name
        normalized_id = _new_id("ref") if reference_id is None else reference_id

        record, _ = self.store.add_reference(
            reference_id=normalized_id,
            name=normalized_name,
            content=reference_path.read_bytes(),
            source_format=source_format,
            config_sha256=self.config_sha256,
            row_count=len(frame),
            columns=cast(list[str], frame.columns.tolist()),
            dtypes={str(column): str(dtype) for column, dtype in frame.dtypes.items()},
        )
        if activate:
            return self.activate_reference(
                record.reference_id,
                discard_buffered=discard_buffered,
            )
        return record

    def activate_reference(
        self,
        reference_id: str,
        *,
        discard_buffered: bool = False,
    ) -> ReferenceRecord:
        reference = self.store.get_reference(reference_id)
        self._require_compatible_reference(reference)
        return self.store.activate_reference(
            reference_id,
            discard_buffered=discard_buffered,
        )

    @staticmethod
    def _normalize_event_envelope(event: object) -> dict[str, Any]:
        if not isinstance(event, Mapping):
            raise ValueError("Каждое событие должно быть отображением")
        non_string_keys = [key for key in event if not isinstance(key, str)]
        if non_string_keys:
            raise ValueError("Все ключи события должны быть строками")
        actual = set(event)
        missing = sorted(_EVENT_REQUIRED_KEYS - actual)
        unknown = sorted(actual - _EVENT_REQUIRED_KEYS - _EVENT_OPTIONAL_KEYS)
        if missing or unknown:
            problems: list[str] = []
            if missing:
                problems.append(f"отсутствуют ключи: {', '.join(missing)}")
            if unknown:
                problems.append(f"неизвестные ключи: {', '.join(unknown)}")
            raise ValueError(f"Некорректное событие: {'; '.join(problems)}")
        return dict(event)

    def ingest_events(self, events: list[Mapping[str, Any]]) -> EventIngestionResult:
        """Принять события и синхронно обработать все сформированные окна."""

        if not isinstance(events, list) or not events:
            raise ValueError("events должен быть непустым списком")
        normalized_events = [self._normalize_event_envelope(event) for event in events]
        reference = self._active_reference()

        accepted = 0
        duplicates = 0
        for event in normalized_events:
            _, created = self.store.add_event(
                event_id=event["event_id"],
                reference_id=reference.reference_id,
                occurred_at=event["occurred_at"],
                features=event["features"],
                prediction=event.get("prediction"),
                score=event.get("score"),
            )
            if created:
                accepted += 1
            else:
                duplicates += 1

        runs = self.process_ready_windows()
        return {
            "accepted": accepted,
            "duplicates": duplicates,
            "buffered_rows": self.store.count_events(state="buffered"),
            "run_ids": [run.run_id for run in runs],
        }

    @staticmethod
    def _normalize_feedback_envelope(feedback: object) -> dict[str, Any]:
        if not isinstance(feedback, Mapping):
            raise ValueError("Каждый Feedback должен быть отображением")
        if any(not isinstance(key, str) for key in feedback):
            raise ValueError("Все ключи Feedback должны быть строками")
        actual = set(feedback)
        missing = sorted(_FEEDBACK_KEYS - actual)
        unknown = sorted(actual - _FEEDBACK_KEYS)
        if missing or unknown:
            problems: list[str] = []
            if missing:
                problems.append(f"отсутствуют ключи: {', '.join(missing)}")
            if unknown:
                problems.append(f"неизвестные ключи: {', '.join(unknown)}")
            raise ValueError(f"Некорректный Feedback: {'; '.join(problems)}")
        return dict(feedback)

    def ingest_feedback(
        self,
        feedback: list[Mapping[str, Any]],
    ) -> FeedbackIngestionResult:
        """Сохранить y_true и переоценить затронутые завершённые окна."""

        if not isinstance(feedback, list) or not feedback:
            raise ValueError("feedback должен быть непустым списком")
        normalized = [self._normalize_feedback_envelope(item) for item in feedback]

        # Неизвестный Event должен отклонить запрос до первой записи Feedback.
        events = {
            item["event_id"]: self.store.get_event(item["event_id"])
            for item in normalized
        }
        accepted = 0
        duplicates = 0
        for item in normalized:
            _, created = self.store.add_feedback(
                event_id=item["event_id"],
                y_true=item["y_true"],
            )
            if created:
                accepted += 1
            else:
                duplicates += 1

        affected_runs: dict[str, RunRecord] = {}
        for event in events.values():
            if event.window_id is None:
                continue
            run = self.store.get_run_for_source("window", event.window_id)
            if run is not None and run.status == "completed":
                affected_runs[run.run_id] = run

        evaluations = []
        for run in affected_runs.values():
            record = self._evaluate_performance(run)
            if record is not None:
                evaluations.append(self.performance_summary(record))
        return {
            "accepted": accepted,
            "duplicates": duplicates,
            "evaluations": evaluations,
        }

    def _process_window(
        self,
        window_id: str,
        events: list[EventRecord],
    ) -> RunRecord:
        if not events:
            raise ValueError("Window не должен быть пустым")
        reference_ids = {event.reference_id for event in events}
        if len(reference_ids) != 1:
            raise OnlineMonitorError("Window содержит несколько версий Reference")
        reference = self._require_compatible_reference(
            self.store.get_reference(reference_ids.pop())
        )

        existing = self.store.get_run_for_source("window", window_id)
        if existing is None:
            run, _ = self.store.start_run(
                run_id=_new_id("run"),
                source="window",
                source_id=window_id,
                reference_id=reference.reference_id,
                config_sha256=self.config_sha256,
                row_count=len(events),
            )
        else:
            run = existing

        if run.status == "completed":
            self.store.finish_window(window_id, state="processed")
            self._evaluate_performance(run)
            self._deliver_alerts(run)
            return run
        if run.status == "failed":
            self.store.finish_window(window_id, state="failed")
            return run

        current = pd.DataFrame([event.features for event in events])
        try:
            result = analyze(
                self._reference_frame(reference),
                current,
                self.analysis_config,
            )
        except Exception as exc:
            failed = self.store.fail_run(
                run.run_id,
                f"{type(exc).__name__}: {exc}",
            )
            self.store.finish_window(window_id, state="failed")
            return failed

        completed = self.store.complete_run(run.run_id, result)
        self.store.finish_window(window_id, state="processed")
        self._evaluate_performance(completed)
        self._deliver_alerts(completed)
        return completed

    def process_ready_windows(self) -> list[RunRecord]:
        """Восстановить claimed-окна и обработать все полные buffered-окна."""

        completed: list[RunRecord] = []
        for window_id, events in self.store.list_claimed_windows():
            completed.append(self._process_window(window_id, events))

        reference = self._active_reference()
        size = self.online_config["window"]["size"]
        while True:
            window_id = _new_id("window")
            events = self.store.claim_window(
                window_id=window_id,
                reference_id=reference.reference_id,
                size=size,
            )
            if not events:
                break
            completed.append(self._process_window(window_id, events))
        return completed

    def process_batch(
        self,
        *,
        batch_id: str,
        records: list[dict[str, Any]],
    ) -> RunRecord:
        """Немедленно проанализировать готовый Current batch."""

        if not isinstance(records, list):
            raise ValueError("records должен быть списком")
        minimum = self.online_config["window"]["min_batch_rows"]
        if len(records) < minimum:
            raise ValueError(
                f"Batch должен содержать не меньше {minimum} строк; "
                f"получено {len(records)}"
            )
        reference = self._active_reference()
        batch, _ = self.store.add_batch(
            batch_id=batch_id,
            reference_id=reference.reference_id,
            records=records,
        )

        run = (
            self.store.get_run(batch.run_id)
            if batch.run_id is not None
            else self.store.get_run_for_source("batch", batch.batch_id)
        )
        if run is None:
            run, _ = self.store.start_run(
                run_id=_new_id("run"),
                source="batch",
                source_id=batch.batch_id,
                reference_id=reference.reference_id,
                config_sha256=self.config_sha256,
                row_count=batch.row_count,
            )
        if batch.state == "accepted":
            batch = self.store.attach_batch_run(batch.batch_id, run.run_id)

        if run.status == "completed":
            if batch.state == "processing":
                self.store.finish_batch(batch.batch_id, state="completed")
            self._deliver_alerts(run)
            return run
        if run.status == "failed":
            if batch.state == "processing":
                self.store.finish_batch(batch.batch_id, state="failed")
            return run

        current = pd.DataFrame(batch.records)
        try:
            result = analyze(
                self._reference_frame(reference),
                current,
                self.analysis_config,
            )
        except Exception as exc:
            failed = self.store.fail_run(
                run.run_id,
                f"{type(exc).__name__}: {exc}",
            )
            self.store.finish_batch(batch.batch_id, state="failed")
            return failed

        completed = self.store.complete_run(run.run_id, result)
        self.store.finish_batch(batch.batch_id, state="completed")
        self._deliver_alerts(completed)
        return completed

    @staticmethod
    def delivery_summary(delivery: DeliveryRecord) -> dict[str, Any]:
        return {
            "delivery_id": delivery.delivery_id,
            "channel": delivery.channel,
            "status": delivery.status,
            "attempts": delivery.attempts,
            "last_error": delivery.last_error,
            "created_at": delivery.created_at,
            "updated_at": delivery.updated_at,
        }

    @staticmethod
    def performance_summary(performance: PerformanceRecord) -> dict[str, Any]:
        return {
            "run_id": performance.run_id,
            **deepcopy(performance.result),
            "created_at": performance.created_at,
            "updated_at": performance.updated_at,
        }

    def run_envelope(self, run: RunRecord) -> dict[str, Any]:
        """Преобразовать Run в JSON-безопасный online-envelope версии 1.0."""

        envelope = {
            "online_contract_version": "1.0",
            "run_id": run.run_id,
            "source": run.source,
            "source_id": run.source_id,
            "status": run.status,
            "reference_id": run.reference_id,
            "config_sha256": run.config_sha256,
            "row_count": run.row_count,
            "created_at": run.created_at,
            "completed_at": run.completed_at,
            "analysis_result": deepcopy(run.analysis_result),
            "error": run.error,
            "deliveries": [
                self.delivery_summary(delivery)
                for delivery in self.store.list_deliveries(run_id=run.run_id)
            ],
            "performance": (
                None
                if (performance := self.store.get_performance(run.run_id)) is None
                else self.performance_summary(performance)
            ),
        }
        canonical_json(envelope)
        return envelope
