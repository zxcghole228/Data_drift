"""HTTP-контракт FastAPI online-мониторинга."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient

from data_drift_guardian.online.api import create_app

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def raw_config(
    tmp_path: Path,
    *,
    max_request_bytes: int = 1_000_000,
    performance_enabled: bool = False,
) -> dict[str, Any]:
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "online.yaml").read_text(encoding="utf-8")
    )
    config["online"]["state_path"] = str(tmp_path / "monitoring.sqlite3")
    config["online"]["max_request_bytes"] = max_request_bytes
    config["online"]["window"]["size"] = 3
    config["online"]["window"]["min_batch_rows"] = 2
    config["online"]["alerts"]["jsonl_path"] = str(tmp_path / "alerts.jsonl")
    config["online"]["alerts"]["webhook_url_env"] = None
    if performance_enabled:
        config["online"]["performance"].update(
            {
                "enabled": True,
                "min_feedback_rows": 3,
                "baseline_accuracy": 0.9,
                "max_accuracy_drop": 0.1,
                "baseline_roc_auc": 0.9,
                "max_roc_auc_drop": 0.1,
            }
        )
    config["adversarial"]["enabled"] = False
    return config


def write_config(
    tmp_path: Path,
    *,
    max_request_bytes: int = 1_000_000,
    performance_enabled: bool = False,
) -> Path:
    path = tmp_path / "online.yaml"
    path.write_text(
        yaml.safe_dump(
            raw_config(
                tmp_path,
                max_request_bytes=max_request_bytes,
                performance_enabled=performance_enabled,
            ),
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    app = create_app(write_config(tmp_path))
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def reference_csv(*, age_shift: float = 0.0) -> bytes:
    frame = pd.DataFrame(
        {
            "age": [25.0 + age_shift, 30.0 + age_shift, 35.0 + age_shift] * 10,
            "income": [40_000.0, 50_000.0, 60_000.0] * 10,
            "region": ["north", "central", "south"] * 10,
        }
    )
    return frame.to_csv(index=False).encode("utf-8")


def event(index: int, *, age: float | None = None) -> dict[str, Any]:
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


def register_reference(
    client: TestClient,
    *,
    reference_id: str = "ref-1",
    activate: bool = True,
    age_shift: float = 0.0,
) -> Any:
    return client.post(
        "/api/v1/references",
        params={
            "format": "csv",
            "reference_id": reference_id,
            "name": reference_id,
            "activate": str(activate).lower(),
        },
        content=reference_csv(age_shift=age_shift),
        headers={"content-type": "text/csv"},
    )


def test_liveness_is_available_before_reference_and_readiness_is_not(
    client: TestClient,
) -> None:
    live = client.get("/health/live")
    ready = client.get("/health/ready")

    assert live.status_code == 200
    assert live.json() == {"status": "alive"}
    assert ready.status_code == 503
    assert ready.json()["reason"] == "active_reference_missing"


def test_reference_registration_activation_and_readiness(client: TestClient) -> None:
    created = register_reference(client)
    current = client.get("/api/v1/references/current")
    ready = client.get("/health/ready")

    assert created.status_code == 201
    assert created.json()["reference_id"] == "ref-1"
    assert created.json()["active"] is True
    assert "content" not in created.json()
    assert current.status_code == 200
    assert current.json() == created.json()
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"
    assert ready.json()["buffered_rows"] == 0


def test_events_return_202_for_buffer_and_201_for_completed_window(
    client: TestClient,
) -> None:
    assert register_reference(client).status_code == 201

    partial = client.post("/api/v1/events", json={"events": [event(0), event(1)]})
    completed = client.post("/api/v1/events", json={"events": [event(2)]})

    assert partial.status_code == 202
    assert partial.json()["buffered_rows"] == 2
    assert partial.json()["run_ids"] == []
    assert completed.status_code == 201
    assert completed.json()["buffered_rows"] == 0
    run_id = completed.json()["run_ids"][0]

    run = client.get(f"/api/v1/runs/{run_id}")
    history = client.get("/api/v1/runs", params={"status": "completed"})
    assert run.status_code == 200
    assert run.json()["online_contract_version"] == "1.0"
    assert run.json()["analysis_result"]["metadata"]["current_rows"] == 3
    assert history.status_code == 200
    assert history.json()["items"][0]["run_id"] == run_id
    assert "analysis_result" not in history.json()["items"][0]


def test_exact_event_replay_is_idempotent(client: TestClient) -> None:
    register_reference(client)
    events = [event(index) for index in range(3)]
    first = client.post("/api/v1/events", json={"events": events})

    replay = client.post("/api/v1/events", json={"events": events})

    assert first.status_code == 201
    assert replay.status_code == 202
    assert replay.json() == {
        "accepted": 0,
        "duplicates": 3,
        "buffered_rows": 0,
        "run_ids": [],
    }
    assert len(client.get("/api/v1/runs").json()["items"]) == 1


def test_changed_event_replay_returns_conflict(client: TestClient) -> None:
    register_reference(client)
    assert client.post("/api/v1/events", json={"events": [event(0)]}).status_code == 202

    conflict = client.post(
        "/api/v1/events",
        json={"events": [event(0, age=99.0)]},
    )

    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "idempotency_conflict"


def test_ready_batch_is_processed_and_replayed(client: TestClient) -> None:
    register_reference(client)
    payload = {
        "batch_id": "batch-1",
        "records": [event(0)["features"], event(1)["features"]],
    }

    first = client.post("/api/v1/batches", json=payload)
    replay = client.post("/api/v1/batches", json=payload)

    assert first.status_code == 201
    assert first.json()["status"] == "completed"
    assert replay.status_code == 201
    assert replay.json()["run_id"] == first.json()["run_id"]


def test_alert_delivery_state_is_visible_in_run_and_history(
    client: TestClient,
) -> None:
    register_reference(client)
    response = client.post(
        "/api/v1/batches",
        json={
            "batch_id": "shifted-batch",
            "records": [
                event(0, age=90.0)["features"],
                event(1, age=91.0)["features"],
            ],
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["analysis_result"]["summary"]["has_alerts"] is True
    assert body["deliveries"][0]["channel"] == "jsonl"
    assert body["deliveries"][0]["status"] == "succeeded"

    history = client.get("/api/v1/runs").json()["items"]
    item = next(item for item in history if item["run_id"] == body["run_id"])
    assert item["deliveries"] == body["deliveries"]


def test_delayed_feedback_updates_performance_and_is_idempotent(
    tmp_path: Path,
) -> None:
    app = create_app(write_config(tmp_path, performance_enabled=True))
    with TestClient(app, raise_server_exceptions=False) as feedback_client:
        assert register_reference(feedback_client).status_code == 201
        events = [event(index) for index in range(3)]
        created_run = feedback_client.post(
            "/api/v1/events",
            json={"events": events},
        )
        run_id = created_run.json()["run_ids"][0]
        feedback = {
            "feedback": [
                {"event_id": "event-0", "y_true": 1},
                {"event_id": "event-1", "y_true": 0},
                {"event_id": "event-2", "y_true": 1},
            ]
        }

        first = feedback_client.post("/api/v1/feedback", json=feedback)
        replay = feedback_client.post("/api/v1/feedback", json=feedback)
        conflict = feedback_client.post(
            "/api/v1/feedback",
            json={"feedback": [{"event_id": "event-0", "y_true": 0}]},
        )
        run = feedback_client.get(f"/api/v1/runs/{run_id}")

    assert first.status_code == 201
    assert first.json()["accepted"] == 3
    evaluation = first.json()["evaluations"][0]
    assert evaluation["status"] == "evaluated"
    assert evaluation["accuracy"]["value"] == 0.0
    assert evaluation["roc_auc"]["value"] == 0.0
    assert {alert["metric"] for alert in evaluation["alerts"]} == {
        "accuracy",
        "roc_auc",
    }
    assert replay.status_code == 200
    assert replay.json()["duplicates"] == 3
    assert conflict.status_code == 409
    assert run.json()["performance"]["status"] == "evaluated"
    assert run.json()["performance"]["alerts"] == evaluation["alerts"]


def test_batch_smaller_than_minimum_returns_422(client: TestClient) -> None:
    register_reference(client)

    response = client.post(
        "/api/v1/batches",
        json={"batch_id": "small", "records": [event(0)["features"]]},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_input"


def test_reference_switch_requires_explicit_buffer_discard(client: TestClient) -> None:
    register_reference(client)
    client.post("/api/v1/events", json={"events": [event(0)]})
    second = register_reference(
        client,
        reference_id="ref-2",
        activate=False,
        age_shift=1.0,
    )
    assert second.status_code == 201

    refused = client.post("/api/v1/references/ref-2/activate")
    activated = client.post(
        "/api/v1/references/ref-2/activate",
        params={"discard_buffered": "true"},
    )

    assert refused.status_code == 409
    assert refused.json()["error"]["code"] == "pending_events"
    assert activated.status_code == 200
    assert activated.json()["active"] is True


def test_request_validation_has_stable_error_shape(client: TestClient) -> None:
    response = client.post(
        "/api/v1/events",
        json={"events": [{"event_id": "x", "unexpected": 1}]},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "request_validation_error"
    assert response.json()["error"]["details"]


def test_events_without_reference_return_service_unavailable(client: TestClient) -> None:
    response = client.post("/api/v1/events", json={"events": [event(0)]})

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "no_active_reference"


def test_unknown_run_returns_404(client: TestClient) -> None:
    response = client.get("/api/v1/runs/missing")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_request_size_limit_returns_413(tmp_path: Path) -> None:
    app = create_app(write_config(tmp_path, max_request_bytes=32))
    with TestClient(app, raise_server_exceptions=False) as limited_client:
        response = register_reference(limited_client)

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_too_large"


def test_unexpected_error_does_not_expose_traceback(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    register_reference(client)

    def broken_ingestion(events: object) -> None:
        del events
        raise RuntimeError("secret internal details")

    monkeypatch.setattr(client.app.state.monitor, "ingest_events", broken_ingestion)
    response = client.post("/api/v1/events", json={"events": [event(0)]})

    assert response.status_code == 500
    body = response.json()
    assert body["error"]["code"] == "internal_error"
    assert body["error"]["details"]["incident_id"].startswith("incident_")
    assert "secret" not in response.text
    assert "Traceback" not in response.text


def test_openapi_contains_public_online_routes(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]

    assert {
        "/health/live",
        "/health/ready",
        "/api/v1/references",
        "/api/v1/references/{reference_id}/activate",
        "/api/v1/references/current",
        "/api/v1/events",
        "/api/v1/batches",
        "/api/v1/feedback",
        "/api/v1/runs",
        "/api/v1/runs/{run_id}",
    } <= set(paths)


def test_online_module_help_is_available() -> None:
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    source_path = str(PROJECT_ROOT / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (source_path, environment.get("PYTHONPATH")))
    )

    completed = subprocess.run(
        [sys.executable, "-m", "data_drift_guardian.online", "--help"],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert completed.returncode == 0
    assert "--config" in completed.stdout
    assert "--state" in completed.stdout
    assert "--host" in completed.stdout
    assert "--port" in completed.stdout
    assert completed.stderr == ""
