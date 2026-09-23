"""Интеграционные тесты persistent оконного online-процессора."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import data_drift_guardian.online.monitor as monitor_module
from data_drift_guardian.online import (
    MonitoringStore,
    OnlineMonitor,
    load_monitoring_config,
)
from data_drift_guardian.online.monitor import (
    NoActiveReferenceError,
    ReferenceConfigMismatchError,
)
from data_drift_guardian.online.storage import IdempotencyConflictError

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def monitoring_config() -> dict[str, Any]:
    config = load_monitoring_config(PROJECT_ROOT / "configs" / "online.yaml")
    config["online"]["window"]["size"] = 3
    config["online"]["window"]["min_batch_rows"] = 2
    config["analysis"]["adversarial"]["enabled"] = False
    return config


@pytest.fixture
def reference_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [25.0, 30.0, 35.0, 40.0, 45.0, 50.0] * 5,
            "income": [40_000.0, 50_000.0, 60_000.0] * 10,
            "region": ["north", "central", "south"] * 10,
        }
    )


def write_reference(
    tmp_path: Path,
    frame: pd.DataFrame,
    *,
    extension: str = ".csv",
) -> Path:
    path = tmp_path / f"reference{extension}"
    if extension == ".csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_parquet(path, index=False)
    return path


def make_event(index: int, *, age: float | None = None) -> dict[str, Any]:
    return {
        "event_id": f"event-{index}",
        "occurred_at": f"2026-09-23T10:{index:02d}:00Z",
        "features": {
            "age": float(25 + index if age is None else age),
            "income": float(40_000 + index * 1_000),
            "region": ("north", "central", "south")[index % 3],
        },
        "prediction": index % 2,
        "score": 0.8 if index % 2 else 0.2,
    }


def configured_monitor(
    tmp_path: Path,
    monitoring_config: dict[str, Any],
    reference_frame: pd.DataFrame,
) -> OnlineMonitor:
    monitoring_config = deepcopy(monitoring_config)
    monitoring_config["online"]["alerts"]["jsonl_path"] = str(
        tmp_path / "alerts.jsonl"
    )
    monitoring_config["online"]["alerts"]["webhook_url_env"] = None
    store = MonitoringStore(tmp_path / "monitoring.sqlite3")
    monitor = OnlineMonitor(store, monitoring_config)
    monitor.register_reference(
        write_reference(tmp_path, reference_frame),
        reference_id="ref-1",
        activate=True,
    )
    return monitor


def test_monitor_requires_active_reference(
    tmp_path: Path,
    monitoring_config: dict[str, Any],
) -> None:
    monitor = OnlineMonitor(
        MonitoringStore(tmp_path / "monitoring.sqlite3"),
        monitoring_config,
    )

    with pytest.raises(NoActiveReferenceError, match="активный Reference"):
        monitor.ingest_events([make_event(0)])


@pytest.mark.parametrize("extension", [".csv", ".parquet"])
def test_reference_registration_persists_original_content_and_metadata(
    tmp_path: Path,
    monitoring_config: dict[str, Any],
    reference_frame: pd.DataFrame,
    extension: str,
) -> None:
    store = MonitoringStore(tmp_path / "monitoring.sqlite3")
    monitor = OnlineMonitor(store, monitoring_config)
    path = write_reference(tmp_path, reference_frame, extension=extension)

    reference = monitor.register_reference(
        path,
        name="Production baseline",
        reference_id="ref-1",
        activate=True,
    )

    assert reference.active is True
    assert reference.name == "Production baseline"
    assert reference.row_count == len(reference_frame)
    assert reference.content == path.read_bytes()
    assert reference.columns == ["age", "income", "region"]
    assert store.get_active_reference() == reference


def test_invalid_reference_schema_is_not_registered(
    tmp_path: Path,
    monitoring_config: dict[str, Any],
) -> None:
    store = MonitoringStore(tmp_path / "monitoring.sqlite3")
    monitor = OnlineMonitor(store, monitoring_config)
    invalid = pd.DataFrame({"unknown": [1, 2, 3]})

    with pytest.raises(ValueError, match="не соответствует"):
        monitor.register_reference(write_reference(tmp_path, invalid))

    assert store.list_references() == []


def test_events_wait_for_full_window_then_create_completed_run(
    tmp_path: Path,
    monitoring_config: dict[str, Any],
    reference_frame: pd.DataFrame,
) -> None:
    monitor = configured_monitor(tmp_path, monitoring_config, reference_frame)

    partial = monitor.ingest_events([make_event(0), make_event(1)])
    completed = monitor.ingest_events([make_event(2)])

    assert partial == {
        "accepted": 2,
        "duplicates": 0,
        "buffered_rows": 2,
        "run_ids": [],
    }
    assert completed["accepted"] == 1
    assert completed["buffered_rows"] == 0
    assert len(completed["run_ids"]) == 1
    run = monitor.store.get_run(completed["run_ids"][0])
    assert run.status == "completed"
    assert run.row_count == 3
    assert run.analysis_result is not None
    assert run.analysis_result["metadata"]["current_rows"] == 3
    assert monitor.store.count_events(state="processed") == 3


def test_one_request_can_create_multiple_windows_and_leave_surplus(
    tmp_path: Path,
    monitoring_config: dict[str, Any],
    reference_frame: pd.DataFrame,
) -> None:
    monitor = configured_monitor(tmp_path, monitoring_config, reference_frame)

    result = monitor.ingest_events([make_event(index) for index in range(7)])

    assert result["accepted"] == 7
    assert len(result["run_ids"]) == 2
    assert result["buffered_rows"] == 1
    assert all(monitor.store.get_run(run_id).status == "completed" for run_id in result["run_ids"])


def test_exact_event_replay_does_not_create_second_run(
    tmp_path: Path,
    monitoring_config: dict[str, Any],
    reference_frame: pd.DataFrame,
) -> None:
    monitor = configured_monitor(tmp_path, monitoring_config, reference_frame)
    events = [make_event(index) for index in range(3)]
    first = monitor.ingest_events(events)

    replay = monitor.ingest_events(events)

    assert len(first["run_ids"]) == 1
    assert replay == {
        "accepted": 0,
        "duplicates": 3,
        "buffered_rows": 0,
        "run_ids": [],
    }
    assert len(monitor.store.list_runs()) == 1


def test_changed_event_replay_is_rejected(
    tmp_path: Path,
    monitoring_config: dict[str, Any],
    reference_frame: pd.DataFrame,
) -> None:
    monitor = configured_monitor(tmp_path, monitoring_config, reference_frame)
    monitor.ingest_events([make_event(0)])

    with pytest.raises(IdempotencyConflictError, match="event_id"):
        monitor.ingest_events([make_event(0, age=99.0)])


def test_partial_window_survives_store_and_monitor_restart(
    tmp_path: Path,
    monitoring_config: dict[str, Any],
    reference_frame: pd.DataFrame,
) -> None:
    monitor = configured_monitor(tmp_path, monitoring_config, reference_frame)
    monitor.ingest_events([make_event(0), make_event(1)])

    reopened = OnlineMonitor(
        MonitoringStore(tmp_path / "monitoring.sqlite3"),
        monitoring_config,
    )
    result = reopened.ingest_events([make_event(2)])

    assert len(result["run_ids"]) == 1
    assert reopened.store.get_run(result["run_ids"][0]).status == "completed"


def test_claimed_window_is_recovered_after_restart(
    tmp_path: Path,
    monitoring_config: dict[str, Any],
    reference_frame: pd.DataFrame,
) -> None:
    monitor = configured_monitor(tmp_path, monitoring_config, reference_frame)
    active = monitor.store.get_active_reference()
    assert active is not None
    for index in range(3):
        event = make_event(index)
        monitor.store.add_event(
            event_id=event["event_id"],
            reference_id=active.reference_id,
            occurred_at=event["occurred_at"],
            features=event["features"],
            prediction=event["prediction"],
            score=event["score"],
        )
    claimed = monitor.store.claim_window(
        window_id="window-before-crash",
        reference_id=active.reference_id,
        size=3,
    )
    assert len(claimed) == 3

    reopened = OnlineMonitor(
        MonitoringStore(tmp_path / "monitoring.sqlite3"),
        monitoring_config,
    )
    runs = reopened.process_ready_windows()

    assert len(runs) == 1
    assert runs[0].source_id == "window-before-crash"
    assert runs[0].status == "completed"
    assert reopened.store.count_events(state="processed") == 3


def test_batch_is_processed_immediately_and_replay_returns_same_run(
    tmp_path: Path,
    monitoring_config: dict[str, Any],
    reference_frame: pd.DataFrame,
) -> None:
    monitor = configured_monitor(tmp_path, monitoring_config, reference_frame)
    records = [make_event(0)["features"], make_event(1)["features"]]

    first = monitor.process_batch(batch_id="batch-1", records=records)
    replay = monitor.process_batch(batch_id="batch-1", records=deepcopy(records))

    assert first.status == "completed"
    assert replay == first
    assert monitor.store.get_batch("batch-1").state == "completed"
    assert len(monitor.store.list_runs()) == 1

    changed = deepcopy(records)
    changed[0]["age"] = 99.0
    with pytest.raises(IdempotencyConflictError, match="batch_id"):
        monitor.process_batch(batch_id="batch-1", records=changed)


def test_batch_smaller_than_configured_minimum_is_rejected(
    tmp_path: Path,
    monitoring_config: dict[str, Any],
    reference_frame: pd.DataFrame,
) -> None:
    monitor = configured_monitor(tmp_path, monitoring_config, reference_frame)

    with pytest.raises(ValueError, match="не меньше 2"):
        monitor.process_batch(
            batch_id="too-small",
            records=[make_event(0)["features"]],
        )


def test_unexpected_analysis_error_is_persisted_for_batch(
    tmp_path: Path,
    monitoring_config: dict[str, Any],
    reference_frame: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monitor = configured_monitor(tmp_path, monitoring_config, reference_frame)

    def broken_analyze(*args: object, **kwargs: object) -> None:
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(monitor_module, "analyze", broken_analyze)
    run = monitor.process_batch(
        batch_id="batch-1",
        records=[make_event(0)["features"], make_event(1)["features"]],
    )

    assert run.status == "failed"
    assert run.error == "RuntimeError: synthetic failure"
    assert monitor.store.get_batch("batch-1").state == "failed"


def test_strong_shift_creates_alerted_analysis_result(
    tmp_path: Path,
    monitoring_config: dict[str, Any],
    reference_frame: pd.DataFrame,
) -> None:
    monitoring_config["online"]["window"]["size"] = 20
    monitor = configured_monitor(tmp_path, monitoring_config, reference_frame)
    events = [make_event(index, age=90.0 + index / 100) for index in range(20)]

    result = monitor.ingest_events(events)
    run = monitor.store.get_run(result["run_ids"][0])

    assert run.analysis_result is not None
    assert run.analysis_result["summary"]["has_alerts"] is True
    assert run.analysis_result["drift"]["features"]["age"]["alert"] is True


def test_reference_registered_with_other_config_cannot_be_used_silently(
    tmp_path: Path,
    monitoring_config: dict[str, Any],
    reference_frame: pd.DataFrame,
) -> None:
    monitor = configured_monitor(tmp_path, monitoring_config, reference_frame)
    changed_config = deepcopy(monitoring_config)
    changed_config["analysis"]["random_seed"] = 777
    changed = OnlineMonitor(monitor.store, changed_config)

    with pytest.raises(ReferenceConfigMismatchError, match="другой"):
        changed.ingest_events([make_event(0)])


def test_run_envelope_is_json_safe(
    tmp_path: Path,
    monitoring_config: dict[str, Any],
    reference_frame: pd.DataFrame,
) -> None:
    monitor = configured_monitor(tmp_path, monitoring_config, reference_frame)
    run = monitor.process_batch(
        batch_id="batch-1",
        records=[make_event(0)["features"], make_event(1)["features"]],
    )

    envelope = monitor.run_envelope(run)

    assert envelope["online_contract_version"] == "1.0"
    assert envelope["run_id"] == run.run_id
    assert envelope["analysis_result"]["contract_version"] == "0.2"
