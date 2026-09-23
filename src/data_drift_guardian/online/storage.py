"""SQLite-хранилище постоянного состояния online-мониторинга."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import Any, Literal, cast

from .contracts import (
    BatchRecord,
    BatchState,
    DeliveryChannel,
    DeliveryRecord,
    DeliveryStatus,
    EventRecord,
    EventState,
    FeedbackRecord,
    PerformanceRecord,
    PerformanceSample,
    PerformanceStatus,
    ReferenceFormat,
    ReferenceRecord,
    RunRecord,
    RunSource,
    RunStatus,
)

SCHEMA_VERSION = 2

_REFERENCE_FORMATS = frozenset({"csv", "parquet"})
_EVENT_STATES = frozenset({"buffered", "claimed", "processed", "failed", "discarded"})
_BATCH_STATES = frozenset({"accepted", "processing", "completed", "failed"})
_RUN_SOURCES = frozenset({"window", "batch"})
_RUN_STATUSES = frozenset({"running", "completed", "failed"})
_DELIVERY_CHANNELS = frozenset({"jsonl", "webhook"})
_DELIVERY_STATUSES = frozenset({"pending", "succeeded", "failed"})
_PERFORMANCE_STATUSES = frozenset({"evaluated", "not_evaluated"})


class MonitoringStorageError(RuntimeError):
    """Базовая ошибка постоянного online-состояния."""


class UnsupportedSchemaVersionError(MonitoringStorageError):
    """База создана более новой версией приложения."""


class RecordNotFoundError(MonitoringStorageError):
    """Запрошенная запись отсутствует."""


class IdempotencyConflictError(MonitoringStorageError):
    """Идентификатор повторён с другим содержимым."""


class PendingEventsError(MonitoringStorageError):
    """Операция с Reference небезопасна при непустом буфере."""


class InvalidStateTransitionError(MonitoringStorageError):
    """Запрошен недопустимый переход состояния записи."""


def canonical_json(value: object) -> str:
    """Вернуть стабильное JSON-представление или выбросить понятную ошибку."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Значение не является JSON-безопасным: {exc}") from exc


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    if not isinstance(value, bytes):
        raise TypeError("value должен иметь тип bytes")
    return hashlib.sha256(value).hexdigest()


def _json_copy(value: object) -> Any:
    return json.loads(canonical_json(value))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalize_timestamp(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} должен быть непустой ISO 8601 строкой")
    candidate = value.strip()
    try:
        parsed = datetime.fromisoformat(candidate.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{context} должен быть корректной ISO 8601 датой") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{context} должен содержать часовой пояс")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _require_identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} должен быть непустой строкой")
    return value.strip()


def _require_sha256(value: object, context: str) -> str:
    normalized = _require_identifier(value, context).lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise ValueError(f"{context} должен быть SHA-256 в hex-формате")
    return normalized


def _require_choice(
    value: object,
    allowed: frozenset[str],
    context: str,
) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(
            f"{context} должен быть одним из: {', '.join(sorted(allowed))}"
        )
    return value


def _require_positive_int(value: object, context: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{context} должен быть положительным целым числом")
    return value


def _normalize_features(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("features должен быть непустым отображением")
    if any(not isinstance(key, str) or not key.strip() for key in value):
        raise ValueError("Все имена признаков должны быть непустыми строками")
    normalized = _json_copy(dict(value))
    if not isinstance(normalized, dict):  # pragma: no cover - гарантируется выше
        raise ValueError("features должен быть JSON-объектом")
    return cast(dict[str, Any], normalized)


def _normalize_records(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("records должен быть непустым списком объектов")
    return [_normalize_features(record) for record in value]


class MonitoringStore:
    """Транзакционное SQLite-хранилище одной инсталляции monitor-сервиса."""

    def __init__(self, path: str | Path, *, timeout_seconds: float = 5.0) -> None:
        if not isinstance(path, (str, Path)):
            raise TypeError(
                "path должен иметь тип str или pathlib.Path, "
                f"получен {type(path).__name__}"
            )
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, Real):
            raise ValueError("timeout_seconds должен быть положительным числом")
        self.timeout_seconds = float(timeout_seconds)
        if not isfinite(self.timeout_seconds) or self.timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds должен быть положительным числом")

        self.path = Path(path).expanduser()
        if self.path.exists() and self.path.is_dir():
            raise IsADirectoryError(f"Ожидался путь к SQLite-файлу: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._migrate()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(
            str(self.path),
            timeout=self.timeout_seconds,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(f"PRAGMA busy_timeout = {int(self.timeout_seconds * 1000)}")
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except BaseException:
                connection.rollback()
                raise
            else:
                connection.commit()

    def _migrate(self) -> None:
        with self._connect() as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version > SCHEMA_VERSION:
                raise UnsupportedSchemaVersionError(
                    "SQLite schema имеет неподдерживаемую версию "
                    f"{version}; приложение поддерживает {SCHEMA_VERSION}"
                )
            if version == SCHEMA_VERSION:
                return

            if version == 0:
                connection.executescript(
                    """
                BEGIN IMMEDIATE;

                CREATE TABLE reference_versions (
                    reference_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    config_sha256 TEXT NOT NULL,
                    source_format TEXT NOT NULL CHECK (source_format IN ('csv', 'parquet')),
                    row_count INTEGER NOT NULL CHECK (row_count > 0),
                    columns_json TEXT NOT NULL,
                    dtypes_json TEXT NOT NULL,
                    content BLOB NOT NULL,
                    created_at TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 0 CHECK (active IN (0, 1))
                );

                CREATE UNIQUE INDEX one_active_reference
                    ON reference_versions(active) WHERE active = 1;

                CREATE TABLE events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    payload_sha256 TEXT NOT NULL,
                    reference_id TEXT NOT NULL REFERENCES reference_versions(reference_id),
                    occurred_at TEXT NOT NULL,
                    received_at TEXT NOT NULL,
                    features_json TEXT NOT NULL,
                    prediction_json TEXT NOT NULL,
                    score REAL,
                    state TEXT NOT NULL CHECK (
                        state IN ('buffered', 'claimed', 'processed', 'failed', 'discarded')
                    ),
                    window_id TEXT
                );

                CREATE INDEX events_buffer_order
                    ON events(reference_id, state, sequence);
                CREATE INDEX events_window_id ON events(window_id);

                CREATE TABLE runs (
                    run_id TEXT PRIMARY KEY,
                    source TEXT NOT NULL CHECK (source IN ('window', 'batch')),
                    source_id TEXT NOT NULL,
                    reference_id TEXT NOT NULL REFERENCES reference_versions(reference_id),
                    config_sha256 TEXT NOT NULL,
                    row_count INTEGER NOT NULL CHECK (row_count > 0),
                    status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed')),
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    analysis_result_json TEXT,
                    error TEXT,
                    UNIQUE (source, source_id)
                );

                CREATE INDEX runs_created_at ON runs(created_at DESC);
                CREATE INDEX runs_status ON runs(status);

                CREATE TABLE batches (
                    batch_id TEXT PRIMARY KEY,
                    payload_sha256 TEXT NOT NULL,
                    reference_id TEXT NOT NULL REFERENCES reference_versions(reference_id),
                    received_at TEXT NOT NULL,
                    records_json TEXT NOT NULL,
                    row_count INTEGER NOT NULL CHECK (row_count > 0),
                    state TEXT NOT NULL CHECK (
                        state IN ('accepted', 'processing', 'completed', 'failed')
                    ),
                    run_id TEXT REFERENCES runs(run_id)
                );

                CREATE TABLE deliveries (
                    delivery_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES runs(run_id),
                    channel TEXT NOT NULL CHECK (channel IN ('jsonl', 'webhook')),
                    status TEXT NOT NULL CHECK (status IN ('pending', 'succeeded', 'failed')),
                    attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
                    payload_json TEXT NOT NULL,
                    last_error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE (run_id, channel)
                );

                CREATE INDEX deliveries_status ON deliveries(status, created_at);

                CREATE TABLE audit_events (
                    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    details_json TEXT NOT NULL
                );

                PRAGMA user_version = 1;
                COMMIT;
                    """
                )
                version = 1

            if version == 1:
                connection.executescript(
                    """
                    BEGIN IMMEDIATE;

                    CREATE TABLE feedback (
                        event_id TEXT PRIMARY KEY REFERENCES events(event_id),
                        payload_sha256 TEXT NOT NULL,
                        y_true_json TEXT NOT NULL,
                        received_at TEXT NOT NULL
                    );

                    CREATE TABLE performance_evaluations (
                        run_id TEXT PRIMARY KEY REFERENCES runs(run_id),
                        status TEXT NOT NULL CHECK (
                            status IN ('evaluated', 'not_evaluated')
                        ),
                        result_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );

                    CREATE INDEX feedback_received_at
                        ON feedback(received_at, event_id);
                    CREATE INDEX performance_status
                        ON performance_evaluations(status, updated_at);

                    PRAGMA user_version = 2;
                    COMMIT;
                    """
                )
            connection.execute("PRAGMA journal_mode = WAL")

    @property
    def schema_version(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("PRAGMA user_version").fetchone()[0])

    @staticmethod
    def _reference_from_row(row: sqlite3.Row) -> ReferenceRecord:
        return ReferenceRecord(
            reference_id=row["reference_id"],
            name=row["name"],
            content_sha256=row["content_sha256"],
            config_sha256=row["config_sha256"],
            source_format=cast(ReferenceFormat, row["source_format"]),
            row_count=row["row_count"],
            columns=json.loads(row["columns_json"]),
            dtypes=json.loads(row["dtypes_json"]),
            content=bytes(row["content"]),
            created_at=row["created_at"],
            active=bool(row["active"]),
        )

    @staticmethod
    def _event_from_row(row: sqlite3.Row) -> EventRecord:
        return EventRecord(
            sequence=row["sequence"],
            event_id=row["event_id"],
            payload_sha256=row["payload_sha256"],
            reference_id=row["reference_id"],
            occurred_at=row["occurred_at"],
            received_at=row["received_at"],
            features=json.loads(row["features_json"]),
            prediction=json.loads(row["prediction_json"]),
            score=row["score"],
            state=cast(EventState, row["state"]),
            window_id=row["window_id"],
        )

    @staticmethod
    def _batch_from_row(row: sqlite3.Row) -> BatchRecord:
        return BatchRecord(
            batch_id=row["batch_id"],
            payload_sha256=row["payload_sha256"],
            reference_id=row["reference_id"],
            received_at=row["received_at"],
            records=json.loads(row["records_json"]),
            row_count=row["row_count"],
            state=cast(BatchState, row["state"]),
            run_id=row["run_id"],
        )

    @staticmethod
    def _run_from_row(row: sqlite3.Row) -> RunRecord:
        result_json = row["analysis_result_json"]
        return RunRecord(
            run_id=row["run_id"],
            source=cast(RunSource, row["source"]),
            source_id=row["source_id"],
            reference_id=row["reference_id"],
            config_sha256=row["config_sha256"],
            row_count=row["row_count"],
            status=cast(RunStatus, row["status"]),
            created_at=row["created_at"],
            completed_at=row["completed_at"],
            analysis_result=json.loads(result_json) if result_json is not None else None,
            error=row["error"],
        )

    @staticmethod
    def _delivery_from_row(row: sqlite3.Row) -> DeliveryRecord:
        return DeliveryRecord(
            delivery_id=row["delivery_id"],
            run_id=row["run_id"],
            channel=cast(DeliveryChannel, row["channel"]),
            status=cast(DeliveryStatus, row["status"]),
            attempts=row["attempts"],
            payload=json.loads(row["payload_json"]),
            last_error=row["last_error"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _feedback_from_row(row: sqlite3.Row) -> FeedbackRecord:
        return FeedbackRecord(
            event_id=row["event_id"],
            payload_sha256=row["payload_sha256"],
            y_true=json.loads(row["y_true_json"]),
            received_at=row["received_at"],
        )

    @staticmethod
    def _performance_from_row(row: sqlite3.Row) -> PerformanceRecord:
        return PerformanceRecord(
            run_id=row["run_id"],
            status=cast(PerformanceStatus, row["status"]),
            result=json.loads(row["result_json"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _require_reference(connection: sqlite3.Connection, reference_id: str) -> None:
        row = connection.execute(
            "SELECT 1 FROM reference_versions WHERE reference_id = ?",
            (reference_id,),
        ).fetchone()
        if row is None:
            raise RecordNotFoundError(f"Reference не найден: {reference_id}")

    def add_reference(
        self,
        *,
        reference_id: str,
        name: str,
        content: bytes,
        source_format: ReferenceFormat,
        config_sha256: str,
        row_count: int,
        columns: list[str],
        dtypes: Mapping[str, str],
    ) -> tuple[ReferenceRecord, bool]:
        """Добавить Reference; точный повтор вернуть без второй записи."""

        reference_id = _require_identifier(reference_id, "reference_id")
        name = _require_identifier(name, "name")
        source_format = cast(
            ReferenceFormat,
            _require_choice(source_format, _REFERENCE_FORMATS, "source_format"),
        )
        config_sha256 = _require_sha256(config_sha256, "config_sha256")
        row_count = _require_positive_int(row_count, "row_count")
        if not isinstance(content, bytes) or not content:
            raise ValueError("content должен быть непустым bytes")
        if (
            not isinstance(columns, list)
            or not columns
            or any(not isinstance(column, str) or not column.strip() for column in columns)
            or len(columns) != len(set(columns))
        ):
            raise ValueError("columns должен быть списком уникальных непустых строк")
        if not isinstance(dtypes, Mapping) or set(dtypes) != set(columns) or any(
            not isinstance(value, str) or not value.strip() for value in dtypes.values()
        ):
            raise ValueError("dtypes должен содержать строковый dtype для каждой колонки")

        content_sha256 = sha256_bytes(content)
        columns_json = canonical_json(columns)
        dtypes_json = canonical_json(dict(dtypes))
        created_at = _utc_now()

        with self._transaction() as connection:
            existing_row = connection.execute(
                "SELECT * FROM reference_versions WHERE reference_id = ?",
                (reference_id,),
            ).fetchone()
            if existing_row is not None:
                existing = self._reference_from_row(existing_row)
                same = (
                    existing.name == name
                    and existing.content_sha256 == content_sha256
                    and existing.config_sha256 == config_sha256
                    and existing.source_format == source_format
                    and existing.row_count == row_count
                    and existing.columns == columns
                    and existing.dtypes == dict(dtypes)
                )
                if not same:
                    raise IdempotencyConflictError(
                        f"reference_id {reference_id!r} уже связан с другим содержимым"
                    )
                return existing, False

            connection.execute(
                """
                INSERT INTO reference_versions (
                    reference_id, name, content_sha256, config_sha256,
                    source_format, row_count, columns_json, dtypes_json,
                    content, created_at, active
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    reference_id,
                    name,
                    content_sha256,
                    config_sha256,
                    source_format,
                    row_count,
                    columns_json,
                    dtypes_json,
                    content,
                    created_at,
                ),
            )
            row = connection.execute(
                "SELECT * FROM reference_versions WHERE reference_id = ?",
                (reference_id,),
            ).fetchone()
            return self._reference_from_row(row), True

    def get_reference(self, reference_id: str) -> ReferenceRecord:
        reference_id = _require_identifier(reference_id, "reference_id")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM reference_versions WHERE reference_id = ?",
                (reference_id,),
            ).fetchone()
        if row is None:
            raise RecordNotFoundError(f"Reference не найден: {reference_id}")
        return self._reference_from_row(row)

    def list_references(self) -> list[ReferenceRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM reference_versions ORDER BY created_at, reference_id"
            ).fetchall()
        return [self._reference_from_row(row) for row in rows]

    def get_active_reference(self) -> ReferenceRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM reference_versions WHERE active = 1"
            ).fetchone()
        return None if row is None else self._reference_from_row(row)

    def activate_reference(
        self,
        reference_id: str,
        *,
        discard_buffered: bool = False,
    ) -> ReferenceRecord:
        """Атомарно активировать Reference, не смешивая его с текущим буфером."""

        reference_id = _require_identifier(reference_id, "reference_id")
        if type(discard_buffered) is not bool:
            raise TypeError("discard_buffered должен иметь тип bool")

        with self._transaction() as connection:
            target_row = connection.execute(
                "SELECT * FROM reference_versions WHERE reference_id = ?",
                (reference_id,),
            ).fetchone()
            if target_row is None:
                raise RecordNotFoundError(f"Reference не найден: {reference_id}")
            if bool(target_row["active"]):
                return self._reference_from_row(target_row)

            claimed = int(
                connection.execute(
                    "SELECT COUNT(*) FROM events WHERE state = 'claimed'"
                ).fetchone()[0]
            )
            if claimed:
                raise PendingEventsError(
                    "Нельзя сменить Reference во время обработки окна: "
                    f"claimed events={claimed}"
                )

            buffered = int(
                connection.execute(
                    "SELECT COUNT(*) FROM events WHERE state = 'buffered'"
                ).fetchone()[0]
            )
            if buffered and not discard_buffered:
                raise PendingEventsError(
                    "Нельзя сменить Reference при непустом буфере: "
                    f"buffered events={buffered}"
                )
            if buffered:
                connection.execute(
                    "UPDATE events SET state = 'discarded' WHERE state = 'buffered'"
                )

            connection.execute("UPDATE reference_versions SET active = 0 WHERE active = 1")
            connection.execute(
                "UPDATE reference_versions SET active = 1 WHERE reference_id = ?",
                (reference_id,),
            )
            connection.execute(
                "INSERT INTO audit_events (event_type, created_at, details_json) "
                "VALUES (?, ?, ?)",
                (
                    "reference_activated",
                    _utc_now(),
                    canonical_json(
                        {
                            "reference_id": reference_id,
                            "discarded_buffered_events": buffered,
                        }
                    ),
                ),
            )
            updated = connection.execute(
                "SELECT * FROM reference_versions WHERE reference_id = ?",
                (reference_id,),
            ).fetchone()
            return self._reference_from_row(updated)

    def add_event(
        self,
        *,
        event_id: str,
        reference_id: str,
        occurred_at: str,
        features: Mapping[str, Any],
        prediction: Any | None = None,
        score: float | None = None,
    ) -> tuple[EventRecord, bool]:
        """Добавить событие с exactly-once семантикой внутри одной базы."""

        event_id = _require_identifier(event_id, "event_id")
        reference_id = _require_identifier(reference_id, "reference_id")
        occurred_at = _normalize_timestamp(occurred_at, "occurred_at")
        normalized_features = _normalize_features(features)
        normalized_prediction = _json_copy(prediction)
        normalized_score: float | None = None
        if score is not None:
            if isinstance(score, bool) or not isinstance(score, Real):
                raise ValueError("score должен быть конечным числом или null")
            normalized_score = float(score)
            if not isfinite(normalized_score):
                raise ValueError("score должен быть конечным числом или null")

        payload_sha256 = sha256_json(
            {
                "reference_id": reference_id,
                "occurred_at": occurred_at,
                "features": normalized_features,
                "prediction": normalized_prediction,
                "score": normalized_score,
            }
        )
        received_at = _utc_now()

        with self._transaction() as connection:
            existing_row = connection.execute(
                "SELECT * FROM events WHERE event_id = ?", (event_id,)
            ).fetchone()
            if existing_row is not None:
                existing = self._event_from_row(existing_row)
                if existing.payload_sha256 != payload_sha256:
                    raise IdempotencyConflictError(
                        f"event_id {event_id!r} уже связан с другим payload"
                    )
                return existing, False

            self._require_reference(connection, reference_id)
            connection.execute(
                """
                INSERT INTO events (
                    event_id, payload_sha256, reference_id, occurred_at,
                    received_at, features_json, prediction_json, score,
                    state, window_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'buffered', NULL)
                """,
                (
                    event_id,
                    payload_sha256,
                    reference_id,
                    occurred_at,
                    received_at,
                    canonical_json(normalized_features),
                    canonical_json(normalized_prediction),
                    normalized_score,
                ),
            )
            row = connection.execute(
                "SELECT * FROM events WHERE event_id = ?", (event_id,)
            ).fetchone()
            return self._event_from_row(row), True

    def get_event(self, event_id: str) -> EventRecord:
        event_id = _require_identifier(event_id, "event_id")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM events WHERE event_id = ?", (event_id,)
            ).fetchone()
        if row is None:
            raise RecordNotFoundError(f"Event не найден: {event_id}")
        return self._event_from_row(row)

    def count_events(self, *, state: EventState | None = None) -> int:
        with self._connect() as connection:
            if state is None:
                return int(connection.execute("SELECT COUNT(*) FROM events").fetchone()[0])
            normalized = _require_choice(state, _EVENT_STATES, "state")
            return int(
                connection.execute(
                    "SELECT COUNT(*) FROM events WHERE state = ?", (normalized,)
                ).fetchone()[0]
            )

    def list_buffered_events(
        self,
        reference_id: str,
        *,
        limit: int,
    ) -> list[EventRecord]:
        reference_id = _require_identifier(reference_id, "reference_id")
        limit = _require_positive_int(limit, "limit")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM events
                WHERE reference_id = ? AND state = 'buffered'
                ORDER BY sequence
                LIMIT ?
                """,
                (reference_id, limit),
            ).fetchall()
        return [self._event_from_row(row) for row in rows]

    def list_claimed_windows(self) -> list[tuple[str, list[EventRecord]]]:
        """Вернуть незавершённые окна в порядке их первых событий."""

        with self._connect() as connection:
            window_rows = connection.execute(
                """
                SELECT window_id, MIN(sequence) AS first_sequence
                FROM events
                WHERE state = 'claimed' AND window_id IS NOT NULL
                GROUP BY window_id
                ORDER BY first_sequence
                """
            ).fetchall()
            result: list[tuple[str, list[EventRecord]]] = []
            for window_row in window_rows:
                rows = connection.execute(
                    "SELECT * FROM events WHERE window_id = ? ORDER BY sequence",
                    (window_row["window_id"],),
                ).fetchall()
                result.append(
                    (
                        window_row["window_id"],
                        [self._event_from_row(row) for row in rows],
                    )
                )
        return result

    def claim_window(
        self,
        *,
        window_id: str,
        reference_id: str,
        size: int,
    ) -> list[EventRecord]:
        """Атомарно закрепить полное окно; при нехватке строк ничего не менять."""

        window_id = _require_identifier(window_id, "window_id")
        reference_id = _require_identifier(reference_id, "reference_id")
        size = _require_positive_int(size, "size")

        with self._transaction() as connection:
            existing_rows = connection.execute(
                "SELECT * FROM events WHERE window_id = ? ORDER BY sequence",
                (window_id,),
            ).fetchall()
            if existing_rows:
                existing = [self._event_from_row(row) for row in existing_rows]
                if len(existing) != size or any(
                    event.reference_id != reference_id for event in existing
                ):
                    raise IdempotencyConflictError(
                        f"window_id {window_id!r} уже связан с другим окном"
                    )
                return existing

            self._require_reference(connection, reference_id)
            rows = connection.execute(
                """
                SELECT * FROM events
                WHERE reference_id = ? AND state = 'buffered'
                ORDER BY sequence
                LIMIT ?
                """,
                (reference_id, size),
            ).fetchall()
            if len(rows) < size:
                return []

            sequences = [row["sequence"] for row in rows]
            placeholders = ",".join("?" for _ in sequences)
            connection.execute(
                f"UPDATE events SET state = 'claimed', window_id = ? "
                f"WHERE sequence IN ({placeholders}) AND state = 'buffered'",
                (window_id, *sequences),
            )
            claimed_rows = connection.execute(
                "SELECT * FROM events WHERE window_id = ? ORDER BY sequence",
                (window_id,),
            ).fetchall()
            if len(claimed_rows) != size:
                raise MonitoringStorageError("Не удалось атомарно закрепить полное окно")
            return [self._event_from_row(row) for row in claimed_rows]

    def finish_window(
        self,
        window_id: str,
        *,
        state: Literal["processed", "failed", "discarded"],
    ) -> int:
        window_id = _require_identifier(window_id, "window_id")
        state = cast(
            Literal["processed", "failed", "discarded"],
            _require_choice(
                state,
                frozenset({"processed", "failed", "discarded"}),
                "state",
            ),
        )
        with self._transaction() as connection:
            rows = connection.execute(
                "SELECT state FROM events WHERE window_id = ?", (window_id,)
            ).fetchall()
            if not rows:
                raise RecordNotFoundError(f"Window не найден: {window_id}")
            current_states = {row["state"] for row in rows}
            if current_states == {state}:
                return len(rows)
            if current_states != {"claimed"}:
                raise InvalidStateTransitionError(
                    f"Window {window_id!r} нельзя перевести из {sorted(current_states)} "
                    f"в {state}"
                )
            connection.execute(
                "UPDATE events SET state = ? WHERE window_id = ?",
                (state, window_id),
            )
            return len(rows)

    def add_batch(
        self,
        *,
        batch_id: str,
        reference_id: str,
        records: list[dict[str, Any]],
    ) -> tuple[BatchRecord, bool]:
        batch_id = _require_identifier(batch_id, "batch_id")
        reference_id = _require_identifier(reference_id, "reference_id")
        normalized_records = _normalize_records(records)
        payload_sha256 = sha256_json(
            {"reference_id": reference_id, "records": normalized_records}
        )
        received_at = _utc_now()

        with self._transaction() as connection:
            existing_row = connection.execute(
                "SELECT * FROM batches WHERE batch_id = ?", (batch_id,)
            ).fetchone()
            if existing_row is not None:
                existing = self._batch_from_row(existing_row)
                if existing.payload_sha256 != payload_sha256:
                    raise IdempotencyConflictError(
                        f"batch_id {batch_id!r} уже связан с другим payload"
                    )
                return existing, False

            self._require_reference(connection, reference_id)
            connection.execute(
                """
                INSERT INTO batches (
                    batch_id, payload_sha256, reference_id, received_at,
                    records_json, row_count, state, run_id
                ) VALUES (?, ?, ?, ?, ?, ?, 'accepted', NULL)
                """,
                (
                    batch_id,
                    payload_sha256,
                    reference_id,
                    received_at,
                    canonical_json(normalized_records),
                    len(normalized_records),
                ),
            )
            row = connection.execute(
                "SELECT * FROM batches WHERE batch_id = ?", (batch_id,)
            ).fetchone()
            return self._batch_from_row(row), True

    def get_batch(self, batch_id: str) -> BatchRecord:
        batch_id = _require_identifier(batch_id, "batch_id")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM batches WHERE batch_id = ?", (batch_id,)
            ).fetchone()
        if row is None:
            raise RecordNotFoundError(f"Batch не найден: {batch_id}")
        return self._batch_from_row(row)

    def start_run(
        self,
        *,
        run_id: str,
        source: RunSource,
        source_id: str,
        reference_id: str,
        config_sha256: str,
        row_count: int,
    ) -> tuple[RunRecord, bool]:
        run_id = _require_identifier(run_id, "run_id")
        source = cast(RunSource, _require_choice(source, _RUN_SOURCES, "source"))
        source_id = _require_identifier(source_id, "source_id")
        reference_id = _require_identifier(reference_id, "reference_id")
        config_sha256 = _require_sha256(config_sha256, "config_sha256")
        row_count = _require_positive_int(row_count, "row_count")
        created_at = _utc_now()

        with self._transaction() as connection:
            existing_row = connection.execute(
                "SELECT * FROM runs WHERE run_id = ? OR (source = ? AND source_id = ?)",
                (run_id, source, source_id),
            ).fetchone()
            if existing_row is not None:
                existing = self._run_from_row(existing_row)
                same = (
                    existing.source == source
                    and existing.source_id == source_id
                    and existing.reference_id == reference_id
                    and existing.config_sha256 == config_sha256
                    and existing.row_count == row_count
                )
                if not same:
                    raise IdempotencyConflictError(
                        f"run/source уже связан с другим запуском: {run_id!r}"
                    )
                return existing, False

            self._require_reference(connection, reference_id)
            connection.execute(
                """
                INSERT INTO runs (
                    run_id, source, source_id, reference_id, config_sha256,
                    row_count, status, created_at, completed_at,
                    analysis_result_json, error
                ) VALUES (?, ?, ?, ?, ?, ?, 'running', ?, NULL, NULL, NULL)
                """,
                (
                    run_id,
                    source,
                    source_id,
                    reference_id,
                    config_sha256,
                    row_count,
                    created_at,
                ),
            )
            row = connection.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            return self._run_from_row(row), True

    def get_run(self, run_id: str) -> RunRecord:
        run_id = _require_identifier(run_id, "run_id")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            raise RecordNotFoundError(f"Run не найден: {run_id}")
        return self._run_from_row(row)

    def get_run_for_source(self, source: RunSource, source_id: str) -> RunRecord | None:
        source = cast(RunSource, _require_choice(source, _RUN_SOURCES, "source"))
        source_id = _require_identifier(source_id, "source_id")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM runs WHERE source = ? AND source_id = ?",
                (source, source_id),
            ).fetchone()
        return None if row is None else self._run_from_row(row)

    def complete_run(self, run_id: str, result: Mapping[str, Any]) -> RunRecord:
        run_id = _require_identifier(run_id, "run_id")
        normalized_result = _json_copy(dict(result))
        if not isinstance(normalized_result, dict):  # pragma: no cover
            raise ValueError("result должен быть JSON-объектом")
        result_json = canonical_json(normalized_result)

        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise RecordNotFoundError(f"Run не найден: {run_id}")
            current = self._run_from_row(row)
            if current.status == "completed":
                if canonical_json(current.analysis_result) != result_json:
                    raise IdempotencyConflictError(
                        f"Run {run_id!r} уже завершён с другим результатом"
                    )
                return current
            if current.status != "running":
                raise InvalidStateTransitionError(
                    f"Run {run_id!r} нельзя завершить из статуса {current.status}"
                )

            connection.execute(
                """
                UPDATE runs
                SET status = 'completed', completed_at = ?,
                    analysis_result_json = ?, error = NULL
                WHERE run_id = ?
                """,
                (_utc_now(), result_json, run_id),
            )
            updated = connection.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            return self._run_from_row(updated)

    def fail_run(self, run_id: str, error: str) -> RunRecord:
        run_id = _require_identifier(run_id, "run_id")
        error = _require_identifier(error, "error")
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise RecordNotFoundError(f"Run не найден: {run_id}")
            current = self._run_from_row(row)
            if current.status == "failed" and current.error == error:
                return current
            if current.status != "running":
                raise InvalidStateTransitionError(
                    f"Run {run_id!r} нельзя завершить ошибкой из статуса {current.status}"
                )
            connection.execute(
                """
                UPDATE runs
                SET status = 'failed', completed_at = ?,
                    analysis_result_json = NULL, error = ?
                WHERE run_id = ?
                """,
                (_utc_now(), error, run_id),
            )
            updated = connection.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            return self._run_from_row(updated)

    def list_runs(
        self,
        *,
        status: RunStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[RunRecord]:
        limit = _require_positive_int(limit, "limit")
        if type(offset) is not int or offset < 0:
            raise ValueError("offset должен быть неотрицательным целым числом")
        with self._connect() as connection:
            if status is None:
                rows = connection.execute(
                    "SELECT * FROM runs ORDER BY created_at DESC, run_id DESC "
                    "LIMIT ? OFFSET ?",
                    (limit, offset),
                ).fetchall()
            else:
                normalized_status = _require_choice(status, _RUN_STATUSES, "status")
                rows = connection.execute(
                    "SELECT * FROM runs WHERE status = ? "
                    "ORDER BY created_at DESC, run_id DESC LIMIT ? OFFSET ?",
                    (normalized_status, limit, offset),
                ).fetchall()
        return [self._run_from_row(row) for row in rows]

    def attach_batch_run(self, batch_id: str, run_id: str) -> BatchRecord:
        batch_id = _require_identifier(batch_id, "batch_id")
        run_id = _require_identifier(run_id, "run_id")
        with self._transaction() as connection:
            batch_row = connection.execute(
                "SELECT * FROM batches WHERE batch_id = ?", (batch_id,)
            ).fetchone()
            if batch_row is None:
                raise RecordNotFoundError(f"Batch не найден: {batch_id}")
            batch = self._batch_from_row(batch_row)
            if batch.state == "processing" and batch.run_id == run_id:
                return batch
            if batch.state != "accepted":
                raise InvalidStateTransitionError(
                    f"Batch {batch_id!r} нельзя начать из статуса {batch.state}"
                )
            run_row = connection.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if run_row is None:
                raise RecordNotFoundError(f"Run не найден: {run_id}")
            run = self._run_from_row(run_row)
            if run.source != "batch" or run.source_id != batch_id:
                raise IdempotencyConflictError(
                    f"Run {run_id!r} не принадлежит Batch {batch_id!r}"
                )
            connection.execute(
                "UPDATE batches SET state = 'processing', run_id = ? WHERE batch_id = ?",
                (run_id, batch_id),
            )
            updated = connection.execute(
                "SELECT * FROM batches WHERE batch_id = ?", (batch_id,)
            ).fetchone()
            return self._batch_from_row(updated)

    def finish_batch(
        self,
        batch_id: str,
        *,
        state: Literal["completed", "failed"],
    ) -> BatchRecord:
        batch_id = _require_identifier(batch_id, "batch_id")
        state = cast(
            Literal["completed", "failed"],
            _require_choice(state, frozenset({"completed", "failed"}), "state"),
        )
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM batches WHERE batch_id = ?", (batch_id,)
            ).fetchone()
            if row is None:
                raise RecordNotFoundError(f"Batch не найден: {batch_id}")
            current = self._batch_from_row(row)
            if current.state == state:
                return current
            if current.state != "processing":
                raise InvalidStateTransitionError(
                    f"Batch {batch_id!r} нельзя перевести из {current.state} в {state}"
                )
            connection.execute(
                "UPDATE batches SET state = ? WHERE batch_id = ?",
                (state, batch_id),
            )
            updated = connection.execute(
                "SELECT * FROM batches WHERE batch_id = ?", (batch_id,)
            ).fetchone()
            return self._batch_from_row(updated)

    def add_delivery(
        self,
        *,
        delivery_id: str,
        run_id: str,
        channel: DeliveryChannel,
        payload: Mapping[str, Any],
    ) -> tuple[DeliveryRecord, bool]:
        delivery_id = _require_identifier(delivery_id, "delivery_id")
        run_id = _require_identifier(run_id, "run_id")
        channel = cast(
            DeliveryChannel,
            _require_choice(channel, _DELIVERY_CHANNELS, "channel"),
        )
        normalized_payload = _json_copy(dict(payload))
        payload_json = canonical_json(normalized_payload)
        now = _utc_now()

        with self._transaction() as connection:
            existing_row = connection.execute(
                "SELECT * FROM deliveries WHERE delivery_id = ? "
                "OR (run_id = ? AND channel = ?)",
                (delivery_id, run_id, channel),
            ).fetchone()
            if existing_row is not None:
                existing = self._delivery_from_row(existing_row)
                if (
                    existing.run_id != run_id
                    or existing.channel != channel
                    or canonical_json(existing.payload) != payload_json
                ):
                    raise IdempotencyConflictError(
                        f"delivery/run channel уже связан с другой доставкой: {delivery_id!r}"
                    )
                return existing, False

            run_row = connection.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if run_row is None:
                raise RecordNotFoundError(f"Run не найден: {run_id}")
            run = self._run_from_row(run_row)
            if run.status != "completed":
                raise InvalidStateTransitionError(
                    f"Доставку можно создать только для completed Run; получен {run.status}"
                )

            connection.execute(
                """
                INSERT INTO deliveries (
                    delivery_id, run_id, channel, status, attempts,
                    payload_json, last_error, created_at, updated_at
                ) VALUES (?, ?, ?, 'pending', 0, ?, NULL, ?, ?)
                """,
                (delivery_id, run_id, channel, payload_json, now, now),
            )
            row = connection.execute(
                "SELECT * FROM deliveries WHERE delivery_id = ?", (delivery_id,)
            ).fetchone()
            return self._delivery_from_row(row), True

    def get_delivery(self, delivery_id: str) -> DeliveryRecord:
        delivery_id = _require_identifier(delivery_id, "delivery_id")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM deliveries WHERE delivery_id = ?", (delivery_id,)
            ).fetchone()
        if row is None:
            raise RecordNotFoundError(f"Delivery не найден: {delivery_id}")
        return self._delivery_from_row(row)

    def update_delivery(
        self,
        delivery_id: str,
        *,
        status: DeliveryStatus,
        attempts: int,
        last_error: str | None,
    ) -> DeliveryRecord:
        delivery_id = _require_identifier(delivery_id, "delivery_id")
        status = cast(
            DeliveryStatus,
            _require_choice(status, _DELIVERY_STATUSES, "status"),
        )
        if type(attempts) is not int or attempts < 0:
            raise ValueError("attempts должен быть неотрицательным целым числом")
        if last_error is not None:
            last_error = _require_identifier(last_error, "last_error")
        if status == "succeeded" and last_error is not None:
            raise ValueError("Успешная доставка не должна содержать last_error")

        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM deliveries WHERE delivery_id = ?", (delivery_id,)
            ).fetchone()
            if row is None:
                raise RecordNotFoundError(f"Delivery не найден: {delivery_id}")
            connection.execute(
                """
                UPDATE deliveries
                SET status = ?, attempts = ?, last_error = ?, updated_at = ?
                WHERE delivery_id = ?
                """,
                (status, attempts, last_error, _utc_now(), delivery_id),
            )
            updated = connection.execute(
                "SELECT * FROM deliveries WHERE delivery_id = ?", (delivery_id,)
            ).fetchone()
            return self._delivery_from_row(updated)

    def list_deliveries(
        self,
        *,
        run_id: str | None = None,
        status: DeliveryStatus | None = None,
    ) -> list[DeliveryRecord]:
        if run_id is not None:
            run_id = _require_identifier(run_id, "run_id")
        with self._connect() as connection:
            if run_id is None and status is None:
                rows = connection.execute(
                    "SELECT * FROM deliveries ORDER BY created_at, delivery_id"
                ).fetchall()
            elif run_id is None:
                normalized = _require_choice(status, _DELIVERY_STATUSES, "status")
                rows = connection.execute(
                    "SELECT * FROM deliveries WHERE status = ? "
                    "ORDER BY created_at, delivery_id",
                    (normalized,),
                ).fetchall()
            elif status is None:
                rows = connection.execute(
                    "SELECT * FROM deliveries WHERE run_id = ? "
                    "ORDER BY created_at, delivery_id",
                    (run_id,),
                ).fetchall()
            else:
                normalized = _require_choice(status, _DELIVERY_STATUSES, "status")
                rows = connection.execute(
                    "SELECT * FROM deliveries WHERE run_id = ? AND status = ? "
                    "ORDER BY created_at, delivery_id",
                    (run_id, normalized),
                ).fetchall()
        return [self._delivery_from_row(row) for row in rows]

    def add_feedback(
        self,
        *,
        event_id: str,
        y_true: object,
    ) -> tuple[FeedbackRecord, bool]:
        """Сохранить фактическую метку; точный повтор вернуть идемпотентно."""

        event_id = _require_identifier(event_id, "event_id")
        if type(y_true) not in {bool, int, float, str}:
            raise ValueError(
                "y_true должен иметь тип bool, int, float или str"
            )
        normalized_y_true = _json_copy(y_true)
        payload = {"event_id": event_id, "y_true": normalized_y_true}
        payload_sha256 = sha256_json(payload)
        y_true_json = canonical_json(normalized_y_true)
        received_at = _utc_now()

        with self._transaction() as connection:
            event = connection.execute(
                "SELECT 1 FROM events WHERE event_id = ?",
                (event_id,),
            ).fetchone()
            if event is None:
                raise RecordNotFoundError(f"Event не найден: {event_id}")

            existing_row = connection.execute(
                "SELECT * FROM feedback WHERE event_id = ?",
                (event_id,),
            ).fetchone()
            if existing_row is not None:
                existing = self._feedback_from_row(existing_row)
                if existing.payload_sha256 != payload_sha256:
                    raise IdempotencyConflictError(
                        f"event_id {event_id!r} уже имеет другой Feedback"
                    )
                return existing, False

            connection.execute(
                """
                INSERT INTO feedback (
                    event_id, payload_sha256, y_true_json, received_at
                ) VALUES (?, ?, ?, ?)
                """,
                (event_id, payload_sha256, y_true_json, received_at),
            )
            row = connection.execute(
                "SELECT * FROM feedback WHERE event_id = ?",
                (event_id,),
            ).fetchone()
            return self._feedback_from_row(row), True

    def get_feedback(self, event_id: str) -> FeedbackRecord:
        event_id = _require_identifier(event_id, "event_id")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM feedback WHERE event_id = ?",
                (event_id,),
            ).fetchone()
        if row is None:
            raise RecordNotFoundError(f"Feedback не найден: {event_id}")
        return self._feedback_from_row(row)

    def list_feedback_samples(self, window_id: str) -> list[PerformanceSample]:
        window_id = _require_identifier(window_id, "window_id")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT e.event_id, e.prediction_json, e.score, f.y_true_json
                FROM events AS e
                JOIN feedback AS f ON f.event_id = e.event_id
                WHERE e.window_id = ?
                ORDER BY e.sequence
                """,
                (window_id,),
            ).fetchall()
        return [
            PerformanceSample(
                event_id=row["event_id"],
                prediction=json.loads(row["prediction_json"]),
                score=row["score"],
                y_true=json.loads(row["y_true_json"]),
            )
            for row in rows
        ]

    def save_performance(
        self,
        *,
        run_id: str,
        status: PerformanceStatus,
        result: Mapping[str, Any],
    ) -> PerformanceRecord:
        """Создать или обновить воспроизводимую оценку качества для Window Run."""

        run_id = _require_identifier(run_id, "run_id")
        status = cast(
            PerformanceStatus,
            _require_choice(status, _PERFORMANCE_STATUSES, "status"),
        )
        normalized_result = _json_copy(dict(result))
        result_json = canonical_json(normalized_result)
        now = _utc_now()

        with self._transaction() as connection:
            run_row = connection.execute(
                "SELECT * FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if run_row is None:
                raise RecordNotFoundError(f"Run не найден: {run_id}")
            run = self._run_from_row(run_row)
            if run.status != "completed" or run.source != "window":
                raise InvalidStateTransitionError(
                    "Performance можно сохранить только для completed Window Run"
                )

            existing = connection.execute(
                "SELECT created_at FROM performance_evaluations WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            created_at = now if existing is None else existing["created_at"]
            connection.execute(
                """
                INSERT INTO performance_evaluations (
                    run_id, status, result_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    status = excluded.status,
                    result_json = excluded.result_json,
                    updated_at = excluded.updated_at
                """,
                (run_id, status, result_json, created_at, now),
            )
            row = connection.execute(
                "SELECT * FROM performance_evaluations WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            return self._performance_from_row(row)

    def get_performance(self, run_id: str) -> PerformanceRecord | None:
        run_id = _require_identifier(run_id, "run_id")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM performance_evaluations WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        return None if row is None else self._performance_from_row(row)

    def list_audit_events(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM audit_events ORDER BY audit_id"
            ).fetchall()
        return [
            {
                "audit_id": row["audit_id"],
                "event_type": row["event_type"],
                "created_at": row["created_at"],
                "details": json.loads(row["details_json"]),
            }
            for row in rows
        ]
