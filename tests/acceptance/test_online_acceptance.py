"""Приёмочные сценарии публичного online-контракта."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient

from data_drift_guardian.online.api import create_app
from scripts.generate_demo_data import generate_demo_data

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _write_config(
    tmp_path: Path,
    *,
    window_size: int = 12,
    minimum_batch_rows: int = 10,
    webhook_env: str | None = None,
    jsonl_enabled: bool = True,
) -> Path:
    config: dict[str, Any] = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "online.yaml").read_text(encoding="utf-8")
    )
    config["adversarial"]["enabled"] = False
    config["online"]["state_path"] = str(tmp_path / "monitoring.sqlite3")
    config["online"]["window"].update(
        {
            "size": window_size,
            "min_batch_rows": minimum_batch_rows,
        }
    )
    config["online"]["alerts"].update(
        {
            "jsonl_path": (
                str(tmp_path / "alerts.jsonl") if jsonl_enabled else None
            ),
            "webhook_url_env": webhook_env,
            "timeout_seconds": 0.05,
            "max_attempts": 2,
        }
    )
    config["online"]["performance"]["enabled"] = False
    config_path = tmp_path / "online.yaml"
    config_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return config_path


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Вернуть JSON-безопасные записи, заменяя pandas NaN на null."""

    return json.loads(frame.to_json(orient="records"))


def _events(frame: pd.DataFrame, *, prefix: str) -> list[dict[str, Any]]:
    started_at = datetime(2026, 9, 23, 10, 0, tzinfo=timezone.utc)
    return [
        {
            "event_id": f"{prefix}-{index}",
            "occurred_at": (started_at + timedelta(seconds=index)).isoformat().replace(
                "+00:00", "Z"
            ),
            "features": record,
        }
        for index, record in enumerate(_records(frame))
    ]


def _register_reference(
    client: TestClient,
    reference: pd.DataFrame,
    *,
    reference_id: str,
) -> None:
    response = client.post(
        "/api/v1/references",
        params={
            "format": "csv",
            "reference_id": reference_id,
            "name": reference_id,
            "activate": "true",
        },
        content=reference.to_csv(index=False).encode("utf-8"),
        headers={"content-type": "text/csv"},
    )
    assert response.status_code == 201, response.text
    assert response.json()["active"] is True


@pytest.mark.online_smoke
def test_partial_window_survives_api_restart_and_replay_is_idempotent(
    tmp_path: Path,
) -> None:
    generated = generate_demo_data(
        seed=42,
        n_reference=300,
        n_current=12,
        scenario="none",
    )
    config_path = _write_config(tmp_path, window_size=12)
    events = _events(generated.current, prefix="restart")

    with TestClient(create_app(config_path), raise_server_exceptions=False) as client:
        _register_reference(client, generated.reference, reference_id="ref-restart")
        partial = client.post("/api/v1/events", json={"events": events[:7]})
        assert partial.status_code == 202
        assert partial.json()["buffered_rows"] == 7
        assert partial.json()["run_ids"] == []

    with TestClient(create_app(config_path), raise_server_exceptions=False) as client:
        readiness = client.get("/health/ready")
        assert readiness.status_code == 200
        assert readiness.json()["buffered_rows"] == 7

        completed = client.post("/api/v1/events", json={"events": events[7:]})
        assert completed.status_code == 201
        assert completed.json()["buffered_rows"] == 0
        assert len(completed.json()["run_ids"]) == 1
        run_id = completed.json()["run_ids"][0]

        first_run = client.get(f"/api/v1/runs/{run_id}")
        replay = client.post("/api/v1/events", json={"events": events})
        replayed_run = client.get(f"/api/v1/runs/{run_id}")
        history = client.get("/api/v1/runs").json()["items"]

    assert first_run.status_code == 200
    assert first_run.json()["status"] == "completed"
    assert first_run.json()["row_count"] == 12
    assert replay.status_code == 202
    assert replay.json() == {
        "accepted": 0,
        "duplicates": 12,
        "buffered_rows": 0,
        "run_ids": [],
    }
    assert replayed_run.json() == first_run.json()
    assert [item["run_id"] for item in history] == [run_id]


def test_none_batch_completes_without_quality_or_drift_alerts(
    tmp_path: Path,
) -> None:
    generated = generate_demo_data(
        seed=42,
        n_reference=5_000,
        n_current=3_000,
        scenario="none",
    )
    config_path = _write_config(tmp_path)

    with TestClient(create_app(config_path), raise_server_exceptions=False) as client:
        _register_reference(client, generated.reference, reference_id="ref-none")
        response = client.post(
            "/api/v1/batches",
            json={"batch_id": "none-batch", "records": _records(generated.current)},
        )

    assert response.status_code == 201, response.text
    envelope = response.json()
    result = envelope["analysis_result"]
    assert envelope["status"] == "completed"
    assert result["summary"]["status"] == "ok", result["alerts"]
    assert result["summary"]["has_alerts"] is False, result["alerts"]
    assert result["alerts"] == []
    assert envelope["deliveries"] == []


def test_combined_batch_creates_expected_alerts_and_exact_replay(
    tmp_path: Path,
) -> None:
    generated = generate_demo_data(
        seed=42,
        n_reference=5_000,
        n_current=1_200,
        scenario="combined",
    )
    config_path = _write_config(tmp_path)
    payload = {
        "batch_id": "combined-batch",
        "records": _records(generated.current),
    }

    with TestClient(create_app(config_path), raise_server_exceptions=False) as client:
        _register_reference(client, generated.reference, reference_id="ref-combined")
        first = client.post("/api/v1/batches", json=payload)
        replay = client.post("/api/v1/batches", json=payload)
        history = client.get("/api/v1/runs").json()["items"]

    assert first.status_code == 201, first.text
    assert replay.status_code == 201, replay.text
    envelope = first.json()
    result = envelope["analysis_result"]
    alerts = result["alerts"]
    assert envelope["status"] == "completed"
    assert result["summary"]["has_alerts"] is True
    assert {alert["source"] for alert in alerts} >= {"quality", "drift"}
    assert any(
        alert["source"] == "quality" and alert["feature"] == "income"
        for alert in alerts
    )
    assert any(
        alert["source"] == "drift" and alert["feature"] == "age"
        for alert in alerts
    )
    assert replay.json() == envelope
    assert len(history) == 1
    assert history[0]["run_id"] == envelope["run_id"]
    assert envelope["deliveries"][0]["channel"] == "jsonl"
    assert envelope["deliveries"][0]["status"] == "succeeded"

    lines = (tmp_path / "alerts.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["run_id"] == envelope["run_id"]


def test_unavailable_webhook_keeps_completed_alerted_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated = generate_demo_data(
        seed=17,
        n_reference=1_000,
        n_current=200,
        scenario="combined",
        age_shift_years=20.0,
        drifted_income_missing_fraction=0.20,
    )
    webhook_env = "DDG_ACCEPTANCE_WEBHOOK_URL"
    monkeypatch.setenv(webhook_env, "https://unavailable.example.test/hook")
    config_path = _write_config(
        tmp_path,
        webhook_env=webhook_env,
        jsonl_enabled=False,
    )
    app = create_app(config_path)

    def unavailable_webhook(
        url: str,
        body: bytes,
        timeout: float,
        delivery_id: str,
    ) -> None:
        del url, body, timeout, delivery_id
        raise TimeoutError("synthetic unavailable webhook")

    app.state.monitor.alert_delivery.webhook_sender = unavailable_webhook
    app.state.monitor.alert_delivery.sleep = lambda _: None

    with TestClient(app, raise_server_exceptions=False) as client:
        _register_reference(client, generated.reference, reference_id="ref-webhook")
        response = client.post(
            "/api/v1/batches",
            json={
                "batch_id": "webhook-batch",
                "records": _records(generated.current),
            },
        )
        assert response.status_code == 201, response.text
        run = client.get(f"/api/v1/runs/{response.json()['run_id']}")

    envelope = response.json()
    assert envelope["status"] == "completed"
    assert envelope["analysis_result"]["summary"]["has_alerts"] is True
    assert len(envelope["deliveries"]) == 1
    delivery = envelope["deliveries"][0]
    assert delivery["channel"] == "webhook"
    assert delivery["status"] == "failed"
    assert delivery["attempts"] == 2
    assert delivery["last_error"] == "TimeoutError: webhook timeout"
    assert run.status_code == 200
    assert run.json()["status"] == "completed"
    assert run.json()["deliveries"] == [delivery]
