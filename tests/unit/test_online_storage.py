"""Тесты транзакционного SQLite-состояния online-мониторинга."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from data_drift_guardian.online.storage import (
    SCHEMA_VERSION,
    IdempotencyConflictError,
    InvalidStateTransitionError,
    MonitoringStore,
    PendingEventsError,
    RecordNotFoundError,
    UnsupportedSchemaVersionError,
)

CONFIG_SHA = "a" * 64


@pytest.fixture
def store(tmp_path: Path) -> MonitoringStore:
    return MonitoringStore(tmp_path / "state" / "monitoring.sqlite3")


def add_reference(
    store: MonitoringStore,
    reference_id: str = "ref-1",
    *,
    content: bytes = b"age,income,region\n30,50000,north\n",
) -> None:
    store.add_reference(
        reference_id=reference_id,
        name=f"Reference {reference_id}",
        content=content,
        source_format="csv",
        config_sha256=CONFIG_SHA,
        row_count=1,
        columns=["age", "income", "region"],
        dtypes={"age": "float64", "income": "float64", "region": "str"},
    )


def add_event(
    store: MonitoringStore,
    event_id: str,
    *,
    reference_id: str = "ref-1",
    age: float = 30.0,
) -> None:
    store.add_event(
        event_id=event_id,
        reference_id=reference_id,
        occurred_at="2026-09-23T10:00:00+03:00",
        features={"age": age, "income": 50_000.0, "region": "north"},
        prediction=1,
        score=0.8,
    )


def test_store_creates_parent_database_and_schema(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "monitoring.sqlite3"

    store = MonitoringStore(path)

    assert path.is_file()
    assert store.schema_version == SCHEMA_VERSION


def test_store_rejects_future_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "future.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA user_version = 999")

    with pytest.raises(UnsupportedSchemaVersionError, match="999"):
        MonitoringStore(path)


def test_store_rejects_directory_path(tmp_path: Path) -> None:
    with pytest.raises(IsADirectoryError, match="SQLite-файлу"):
        MonitoringStore(tmp_path)


def test_existing_schema_v1_is_migrated_without_losing_state(tmp_path: Path) -> None:
    path = tmp_path / "monitoring.sqlite3"
    store = MonitoringStore(path)
    add_reference(store)
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TABLE performance_evaluations")
        connection.execute("DROP TABLE feedback")
        connection.execute("PRAGMA user_version = 1")

    migrated = MonitoringStore(path)

    assert migrated.schema_version == SCHEMA_VERSION
    assert migrated.get_reference("ref-1").name == "Reference ref-1"
    with sqlite3.connect(path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    assert {"feedback", "performance_evaluations"} <= tables


def test_reference_round_trip_and_exact_replay(store: MonitoringStore) -> None:
    record, created = store.add_reference(
        reference_id="ref-1",
        name="Baseline",
        content=b"age\n30\n",
        source_format="csv",
        config_sha256=CONFIG_SHA,
        row_count=1,
        columns=["age"],
        dtypes={"age": "float64"},
    )
    replay, replay_created = store.add_reference(
        reference_id="ref-1",
        name="Baseline",
        content=b"age\n30\n",
        source_format="csv",
        config_sha256=CONFIG_SHA,
        row_count=1,
        columns=["age"],
        dtypes={"age": "float64"},
    )

    assert created is True
    assert replay_created is False
    assert replay == record
    assert store.get_reference("ref-1") == record
    assert record.content_sha256
    assert record.active is False


def test_reference_identifier_conflict_is_rejected(store: MonitoringStore) -> None:
    add_reference(store)

    with pytest.raises(IdempotencyConflictError, match="reference_id"):
        store.add_reference(
            reference_id="ref-1",
            name="Changed",
            content=b"different",
            source_format="csv",
            config_sha256=CONFIG_SHA,
            row_count=1,
            columns=["age"],
            dtypes={"age": "float64"},
        )


def test_active_reference_survives_reopening(tmp_path: Path) -> None:
    path = tmp_path / "monitoring.sqlite3"
    first = MonitoringStore(path)
    add_reference(first)
    first.activate_reference("ref-1")

    reopened = MonitoringStore(path)

    assert reopened.get_active_reference() is not None
    assert reopened.get_active_reference().reference_id == "ref-1"  # type: ignore[union-attr]


def test_reference_switch_requires_explicit_buffer_discard(
    store: MonitoringStore,
) -> None:
    add_reference(store, "ref-1")
    add_reference(store, "ref-2", content=b"age\n40\n")
    store.activate_reference("ref-1")
    add_event(store, "event-1")

    with pytest.raises(PendingEventsError, match="непустом буфере"):
        store.activate_reference("ref-2")

    active = store.activate_reference("ref-2", discard_buffered=True)

    assert active.reference_id == "ref-2"
    assert store.get_event("event-1").state == "discarded"
    assert store.list_audit_events()[-1]["details"] == {
        "reference_id": "ref-2",
        "discarded_buffered_events": 1,
    }


def test_reference_switch_is_blocked_for_claimed_window(
    store: MonitoringStore,
) -> None:
    add_reference(store, "ref-1")
    add_reference(store, "ref-2", content=b"age\n40\n")
    store.activate_reference("ref-1")
    add_event(store, "event-1")
    assert store.claim_window(window_id="window-1", reference_id="ref-1", size=1)

    with pytest.raises(PendingEventsError, match="обработки окна"):
        store.activate_reference("ref-2", discard_buffered=True)


def test_event_round_trip_replay_and_conflict(store: MonitoringStore) -> None:
    add_reference(store)

    event, created = store.add_event(
        event_id="event-1",
        reference_id="ref-1",
        occurred_at="2026-09-23T10:00:00+03:00",
        features={"age": 30.0, "region": "north"},
        prediction=1,
        score=0.8,
    )
    replay, replay_created = store.add_event(
        event_id="event-1",
        reference_id="ref-1",
        occurred_at="2026-09-23T07:00:00Z",
        features={"region": "north", "age": 30.0},
        prediction=1,
        score=0.8,
    )

    assert created is True
    assert replay_created is False
    assert replay == event
    assert event.occurred_at == "2026-09-23T07:00:00Z"
    assert event.state == "buffered"
    assert store.count_events(state="buffered") == 1

    with pytest.raises(IdempotencyConflictError, match="event_id"):
        store.add_event(
            event_id="event-1",
            reference_id="ref-1",
            occurred_at="2026-09-23T07:00:00Z",
            features={"age": 99.0, "region": "north"},
            prediction=1,
            score=0.8,
        )


def test_event_requires_existing_reference(store: MonitoringStore) -> None:
    with pytest.raises(RecordNotFoundError, match="Reference"):
        add_event(store, "event-1", reference_id="missing")


def test_event_rejects_non_json_and_non_finite_values(store: MonitoringStore) -> None:
    add_reference(store)

    with pytest.raises(ValueError, match="JSON-безопасным"):
        store.add_event(
            event_id="event-1",
            reference_id="ref-1",
            occurred_at="2026-09-23T07:00:00Z",
            features={"age": float("nan")},
        )

    with pytest.raises(ValueError, match="score"):
        store.add_event(
            event_id="event-2",
            reference_id="ref-1",
            occurred_at="2026-09-23T07:00:00Z",
            features={"age": 30},
            score=float("inf"),
        )


def test_claim_window_is_atomic_and_keeps_surplus_buffered(
    store: MonitoringStore,
) -> None:
    add_reference(store)
    for index in range(3):
        add_event(store, f"event-{index}", age=30.0 + index)

    claimed = store.claim_window(
        window_id="window-1",
        reference_id="ref-1",
        size=2,
    )
    replay = store.claim_window(
        window_id="window-1",
        reference_id="ref-1",
        size=2,
    )

    assert [event.event_id for event in claimed] == ["event-0", "event-1"]
    assert replay == claimed
    assert {event.state for event in claimed} == {"claimed"}
    assert [event.event_id for event in store.list_buffered_events("ref-1", limit=10)] == [
        "event-2"
    ]
    assert store.finish_window("window-1", state="processed") == 2
    assert store.finish_window("window-1", state="processed") == 2

    with pytest.raises(InvalidStateTransitionError):
        store.finish_window("window-1", state="failed")


def test_claim_window_does_nothing_until_full(store: MonitoringStore) -> None:
    add_reference(store)
    add_event(store, "event-1")

    assert (
        store.claim_window(window_id="window-1", reference_id="ref-1", size=2)
        == []
    )
    assert store.get_event("event-1").state == "buffered"


def test_batch_run_and_delivery_lifecycle(store: MonitoringStore) -> None:
    add_reference(store)
    batch, batch_created = store.add_batch(
        batch_id="batch-1",
        reference_id="ref-1",
        records=[{"age": 30.0}, {"age": 31.0}],
    )
    batch_replay, replay_created = store.add_batch(
        batch_id="batch-1",
        reference_id="ref-1",
        records=[{"age": 30.0}, {"age": 31.0}],
    )

    assert batch_created is True
    assert replay_created is False
    assert batch_replay == batch

    run, run_created = store.start_run(
        run_id="run-1",
        source="batch",
        source_id="batch-1",
        reference_id="ref-1",
        config_sha256=CONFIG_SHA,
        row_count=2,
    )
    assert run_created is True
    assert run.status == "running"
    assert store.attach_batch_run("batch-1", "run-1").state == "processing"

    completed = store.complete_run(
        "run-1",
        {"contract_version": "0.2", "summary": {"has_alerts": True}},
    )
    assert completed.status == "completed"
    assert completed.analysis_result == {
        "contract_version": "0.2",
        "summary": {"has_alerts": True},
    }
    assert store.finish_batch("batch-1", state="completed").state == "completed"

    delivery, delivery_created = store.add_delivery(
        delivery_id="delivery-1",
        run_id="run-1",
        channel="webhook",
        payload={"run_id": "run-1", "alerts": ["drift"]},
    )
    assert delivery_created is True
    assert delivery.status == "pending"
    succeeded = store.update_delivery(
        "delivery-1",
        status="succeeded",
        attempts=1,
        last_error=None,
    )
    assert succeeded.status == "succeeded"
    assert store.list_deliveries(status="succeeded") == [succeeded]


def test_batch_identifier_conflict_is_rejected(store: MonitoringStore) -> None:
    add_reference(store)
    store.add_batch(
        batch_id="batch-1",
        reference_id="ref-1",
        records=[{"age": 30.0}],
    )

    with pytest.raises(IdempotencyConflictError, match="batch_id"):
        store.add_batch(
            batch_id="batch-1",
            reference_id="ref-1",
            records=[{"age": 99.0}],
        )


def test_run_failure_and_invalid_transition(store: MonitoringStore) -> None:
    add_reference(store)
    store.start_run(
        run_id="run-1",
        source="window",
        source_id="window-1",
        reference_id="ref-1",
        config_sha256=CONFIG_SHA,
        row_count=2,
    )

    failed = store.fail_run("run-1", "analysis failed")

    assert failed.status == "failed"
    assert failed.error == "analysis failed"
    assert store.fail_run("run-1", "analysis failed") == failed
    with pytest.raises(InvalidStateTransitionError):
        store.complete_run("run-1", {"contract_version": "0.2"})


def test_run_source_is_idempotent_and_history_is_filterable(
    store: MonitoringStore,
) -> None:
    add_reference(store)
    first, _ = store.start_run(
        run_id="run-1",
        source="window",
        source_id="window-1",
        reference_id="ref-1",
        config_sha256=CONFIG_SHA,
        row_count=2,
    )
    replay, created = store.start_run(
        run_id="another-generated-id",
        source="window",
        source_id="window-1",
        reference_id="ref-1",
        config_sha256=CONFIG_SHA,
        row_count=2,
    )
    store.complete_run("run-1", {"contract_version": "0.2"})

    assert created is False
    assert replay == first
    assert store.get_run_for_source("window", "window-1").run_id == "run-1"  # type: ignore[union-attr]
    assert [run.run_id for run in store.list_runs(status="completed")] == ["run-1"]
    assert store.list_runs(status="failed") == []


def test_result_must_be_json_safe(store: MonitoringStore) -> None:
    add_reference(store)
    store.start_run(
        run_id="run-1",
        source="window",
        source_id="window-1",
        reference_id="ref-1",
        config_sha256=CONFIG_SHA,
        row_count=1,
    )

    with pytest.raises(ValueError, match="JSON-безопасным"):
        store.complete_run("run-1", {"value": float("nan")})


def test_delivery_requires_completed_run(store: MonitoringStore) -> None:
    add_reference(store)
    store.start_run(
        run_id="run-1",
        source="window",
        source_id="window-1",
        reference_id="ref-1",
        config_sha256=CONFIG_SHA,
        row_count=1,
    )

    with pytest.raises(InvalidStateTransitionError, match="completed"):
        store.add_delivery(
            delivery_id="delivery-1",
            run_id="run-1",
            channel="jsonl",
            payload={"run_id": "run-1"},
        )


def test_feedback_is_idempotent_and_conflicting_label_is_rejected(
    store: MonitoringStore,
) -> None:
    add_reference(store)
    add_event(store, "event-1")

    first, created = store.add_feedback(event_id="event-1", y_true=1)
    replay, replay_created = store.add_feedback(event_id="event-1", y_true=1)

    assert created is True
    assert replay_created is False
    assert replay == first
    assert store.get_feedback("event-1") == first
    with pytest.raises(IdempotencyConflictError, match="другой Feedback"):
        store.add_feedback(event_id="event-1", y_true=0)
    with pytest.raises(RecordNotFoundError, match="missing"):
        store.add_feedback(event_id="missing", y_true=1)


def test_performance_result_is_updated_as_feedback_arrives(
    store: MonitoringStore,
) -> None:
    add_reference(store)
    add_event(store, "event-1")
    events = store.claim_window(
        window_id="window-1",
        reference_id="ref-1",
        size=1,
    )
    assert len(events) == 1
    store.start_run(
        run_id="run-1",
        source="window",
        source_id="window-1",
        reference_id="ref-1",
        config_sha256=CONFIG_SHA,
        row_count=1,
    )
    store.complete_run("run-1", {"contract_version": "0.2"})
    store.finish_window("window-1", state="processed")

    initial = store.save_performance(
        run_id="run-1",
        status="not_evaluated",
        result={"status": "not_evaluated", "feedback_rows": 0},
    )
    updated = store.save_performance(
        run_id="run-1",
        status="evaluated",
        result={"status": "evaluated", "feedback_rows": 1, "accuracy": 1.0},
    )

    assert updated.created_at == initial.created_at
    assert updated.status == "evaluated"
    assert updated.result["accuracy"] == 1.0
    assert store.get_performance("run-1") == updated
