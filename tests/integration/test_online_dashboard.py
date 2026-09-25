"""Интеграционные проверки read-only online-панели Streamlit."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from streamlit.testing.v1 import AppTest

from data_drift_guardian.online.storage import MonitoringStore

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_PATH = PROJECT_ROOT / "app" / "streamlit_app.py"
APP_TEST_TIMEOUT = 30
CONFIG_SHA = "a" * 64


def _write_config(
    tmp_path: Path,
    *,
    performance_enabled: bool = False,
) -> tuple[Path, Path]:
    config: dict[str, Any] = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "online.yaml").read_text(encoding="utf-8")
    )
    state_path = tmp_path / "monitoring.sqlite3"
    config["online"]["state_path"] = str(state_path)
    config["online"]["window"]["size"] = 4
    config["online"]["window"]["min_batch_rows"] = 2
    config["online"]["performance"]["enabled"] = performance_enabled
    config_path = tmp_path / "online.yaml"
    config_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return config_path, state_path


def _online_app(monkeypatch: Any, config_path: Path) -> AppTest:
    monkeypatch.setenv("DDG_ONLINE_CONFIG", str(config_path))
    app = AppTest.from_file(str(APP_PATH), default_timeout=APP_TEST_TIMEOUT)
    app.run()
    app.radio(key="application_mode").set_value("Онлайн-мониторинг").run()
    return app


def _seed_state(state_path: Path) -> None:
    store = MonitoringStore(state_path)
    store.add_reference(
        reference_id="ref-dashboard",
        name="Dashboard Reference",
        content=b"age,income,region\n30,50000,north\n",
        source_format="csv",
        config_sha256=CONFIG_SHA,
        row_count=1,
        columns=["age", "income", "region"],
        dtypes={"age": "float64", "income": "float64", "region": "str"},
    )
    store.activate_reference("ref-dashboard")
    store.add_event(
        event_id="buffered-event",
        reference_id="ref-dashboard",
        occurred_at="2026-09-23T10:00:00Z",
        features={"age": 31.0, "income": 51_000.0, "region": "north"},
        prediction=1,
        score=0.8,
    )
    store.start_run(
        run_id="run-dashboard",
        source="window",
        source_id="window-dashboard",
        reference_id="ref-dashboard",
        config_sha256=CONFIG_SHA,
        row_count=4,
    )
    store.complete_run(
        "run-dashboard",
        {
            "summary": {
                "status": "warning",
                "has_alerts": True,
                "n_alerts": 1,
                "analyzed_features": 3,
                "skipped_features": 0,
            },
            "alerts": [
                {
                    "source": "drift",
                    "check": "wasserstein",
                    "feature": "age",
                    "severity": "warning",
                    "message": "Сдвиг age",
                }
            ],
        },
    )
    store.add_delivery(
        delivery_id="delivery-dashboard",
        run_id="run-dashboard",
        channel="jsonl",
        payload={"run_id": "run-dashboard"},
    )
    store.update_delivery(
        "delivery-dashboard",
        status="succeeded",
        attempts=1,
        last_error=None,
    )
    store.save_performance(
        run_id="run-dashboard",
        status="evaluated",
        result={
            "status": "evaluated",
            "feedback_rows": 4,
            "reason": None,
            "accuracy": {"value": 0.75, "alert": False},
            "roc_auc": {"value": 0.8, "alert": False},
            "alerts": [],
        },
    )


def test_online_mode_reports_uninitialized_database(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    config_path, _ = _write_config(tmp_path)

    app = _online_app(monkeypatch, config_path)

    assert not app.exception
    assert any("ещё не создана" in warning.value for warning in app.warning)
    assert len(app.metric) == 0


def test_online_mode_shows_reference_history_alerts_and_performance(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    config_path, state_path = _write_config(
        tmp_path,
        performance_enabled=True,
    )
    _seed_state(state_path)

    app = _online_app(monkeypatch, config_path)

    assert not app.exception
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["Готовность данных"] == "Готово"
    assert metrics["Буфер"] == "1"
    assert metrics["Размер окна"] == "4"
    assert metrics["Run в снимке"] == "1"
    assert any(metric.value == "ref-dashboard" for metric in app.metric)
    assert len(app.dataframe) == 4

    tables = [dataframe.value for dataframe in app.dataframe]
    assert any("Статус запуска" in table.columns for table in tables)
    assert any("Статус" in table.columns for table in tables)
    assert any("Канал" in table.columns for table in tables)
    assert any("Accuracy" in table.columns for table in tables)
    rendered_tables = "\n".join(table.astype(str).to_string() for table in tables)
    assert "✅ Завершён" in rendered_tables
    assert "⚠️ Предупреждение" in rendered_tables
    assert "✅ Успешно" in rendered_tables
    assert "warning" not in rendered_tables
    assert "completed" not in rendered_tables
    assert "succeeded" not in rendered_tables
    assert any("Сдвиг age" in table.astype(str).to_string() for table in tables)
    assert any("0.75" in table.astype(str).to_string() for table in tables)


def test_online_mode_explains_disabled_performance(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    config_path, state_path = _write_config(tmp_path)
    _seed_state(state_path)

    app = _online_app(monkeypatch, config_path)

    assert not app.exception
    assert len(app.dataframe) == 3
    assert any(
        "Проверка качества модели отключена" in message.value
        for message in app.info
    )
    assert all(
        "Accuracy" not in dataframe.value.columns
        for dataframe in app.dataframe
    )
