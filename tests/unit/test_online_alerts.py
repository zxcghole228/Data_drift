"""Тесты JSONL- и webhook-доставки online-алертов."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from data_drift_guardian.online.alerts import AlertDeliveryService
from data_drift_guardian.online.contracts import OnlineAlertConfig, RunRecord
from data_drift_guardian.online.storage import MonitoringStore

CONFIG_SHA = "a" * 64


def completed_run(
    store: MonitoringStore,
    *,
    run_id: str = "run-1",
    alerts: list[dict[str, Any]] | None = None,
) -> RunRecord:
    store.add_reference(
        reference_id="ref-1",
        name="Baseline",
        content=b"age\n30\n",
        source_format="csv",
        config_sha256=CONFIG_SHA,
        row_count=1,
        columns=["age"],
        dtypes={"age": "float64"},
    )
    store.start_run(
        run_id=run_id,
        source="batch",
        source_id="batch-1",
        reference_id="ref-1",
        config_sha256=CONFIG_SHA,
        row_count=2,
    )
    return store.complete_run(
        run_id,
        {
            "contract_version": "0.2",
            "summary": {"status": "warning", "has_alerts": bool(alerts)},
            "alerts": [] if alerts is None else alerts,
        },
    )


def alert_config(
    *,
    jsonl_path: Path | None,
    webhook_url_env: str | None = None,
    max_attempts: int = 3,
) -> OnlineAlertConfig:
    return {
        "jsonl_path": None if jsonl_path is None else str(jsonl_path),
        "webhook_url_env": webhook_url_env,
        "timeout_seconds": 0.25,
        "max_attempts": max_attempts,
    }


def drift_alert() -> dict[str, Any]:
    return {
        "source": "drift",
        "feature": "age",
        "check": "wasserstein",
        "message": "Обнаружен числовой дрейф",
    }


def test_run_without_alerts_creates_no_deliveries(tmp_path: Path) -> None:
    store = MonitoringStore(tmp_path / "state.sqlite3")
    run = completed_run(store)
    service = AlertDeliveryService(
        store,
        alert_config(jsonl_path=tmp_path / "alerts.jsonl"),
    )

    assert service.dispatch_run(run) == []
    assert store.list_deliveries() == []
    assert not (tmp_path / "alerts.jsonl").exists()


def test_jsonl_delivery_is_durable_and_idempotent(tmp_path: Path) -> None:
    store = MonitoringStore(tmp_path / "state.sqlite3")
    run = completed_run(store, alerts=[drift_alert()])
    path = tmp_path / "nested" / "alerts.jsonl"
    service = AlertDeliveryService(store, alert_config(jsonl_path=path))

    first = service.dispatch_run(run)
    replay = service.dispatch_run(run)

    assert first == replay
    assert first[0].channel == "jsonl"
    assert first[0].status == "succeeded"
    assert first[0].attempts == 1
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["delivery_id"] == first[0].delivery_id
    assert payload["run_id"] == run.run_id
    assert payload["alerts"] == [drift_alert()]


def test_webhook_retries_with_stable_delivery_id_then_succeeds(
    tmp_path: Path,
) -> None:
    store = MonitoringStore(tmp_path / "state.sqlite3")
    run = completed_run(store, alerts=[drift_alert()])
    calls: list[tuple[str, dict[str, Any], float, str]] = []
    sleeps: list[float] = []

    def sender(url: str, body: bytes, timeout: float, delivery_id: str) -> None:
        calls.append((url, json.loads(body), timeout, delivery_id))
        if len(calls) < 3:
            raise TimeoutError("synthetic timeout")

    service = AlertDeliveryService(
        store,
        alert_config(
            jsonl_path=None,
            webhook_url_env="TEST_WEBHOOK_URL",
        ),
        environment={"TEST_WEBHOOK_URL": "https://example.test/hook?secret=hidden"},
        webhook_sender=sender,
        sleep=sleeps.append,
    )

    deliveries = service.dispatch_run(run)

    assert len(calls) == 3
    assert sleeps == [1.0, 2.0]
    assert len({call[3] for call in calls}) == 1
    assert all(call[1]["delivery_id"] == call[3] for call in calls)
    assert deliveries[0].status == "succeeded"
    assert deliveries[0].attempts == 3
    assert deliveries[0].last_error is None


def test_exhausted_webhook_does_not_change_completed_run(tmp_path: Path) -> None:
    store = MonitoringStore(tmp_path / "state.sqlite3")
    run = completed_run(store, alerts=[drift_alert()])
    calls = 0

    def sender(url: str, body: bytes, timeout: float, delivery_id: str) -> None:
        nonlocal calls
        del url, body, timeout, delivery_id
        calls += 1
        raise TimeoutError("https://secret.example.test/hook")

    service = AlertDeliveryService(
        store,
        alert_config(
            jsonl_path=None,
            webhook_url_env="TEST_WEBHOOK_URL",
            max_attempts=2,
        ),
        environment={"TEST_WEBHOOK_URL": "https://secret.example.test/hook"},
        webhook_sender=sender,
        sleep=lambda _: None,
    )

    delivery = service.dispatch_run(run)[0]
    service.dispatch_run(run)

    assert calls == 2
    assert delivery.status == "failed"
    assert delivery.attempts == 2
    assert delivery.last_error == "TimeoutError: webhook timeout"
    assert "secret.example" not in delivery.last_error
    assert store.get_run(run.run_id).status == "completed"


def test_jsonl_failure_is_persisted_separately_from_run(tmp_path: Path) -> None:
    store = MonitoringStore(tmp_path / "state.sqlite3")
    run = completed_run(store, alerts=[drift_alert()])
    invalid_path = tmp_path / "alerts.jsonl"
    invalid_path.mkdir()
    service = AlertDeliveryService(store, alert_config(jsonl_path=invalid_path))

    delivery = service.dispatch_run(run)[0]

    assert delivery.status == "failed"
    assert delivery.attempts == 1
    assert delivery.last_error is not None
    assert store.get_run(run.run_id).status == "completed"
