"""Интеграционные проверки пользовательского интерфейса Streamlit."""

from __future__ import annotations

import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[2] / "app" / "streamlit_app.py"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
APP_TEST_TIMEOUT = 30

REFERENCE_CSV = b"""age,income,region
20,100,north
30,200,south
40,300,east
50,400,north
60,500,south
70,600,east
"""

CURRENT_CSV = b"""age,income,region
21,110,north
31,210,south
41,310,east
51,410,north
61,510,south
71,610,east
"""


def _loaded_app() -> AppTest:
    app = AppTest.from_file(str(APP_PATH), default_timeout=APP_TEST_TIMEOUT)
    app.run()
    app.file_uploader(key="reference_file").set_value(
        ("reference.csv", REFERENCE_CSV, "text/csv")
    )
    app.file_uploader(key="current_file").set_value(
        ("current.csv", CURRENT_CSV, "text/csv")
    )
    app.run()
    return app


def test_app_waits_for_explicit_analysis_request() -> None:
    app = _loaded_app()

    assert not app.exception
    assert len(app.metric) == 0
    assert any("явно запустите анализ" in message.value.lower() for message in app.info)


def test_app_runs_real_pipeline_and_keeps_result_across_reruns() -> None:
    app = _loaded_app()

    app.button(key="run_analysis").click().run()

    assert not app.exception
    assert len(app.metric) == 4
    assert app.metric[0].value == "6"
    assert app.metric[1].value == "6"
    assert app.metric[2].value == "3"
    assert any("Итоговый статус" in message.value for message in app.success)
    assert len(app.dataframe) >= 3
    assert len(app.get("plotly_chart")) == 1
    assert len(app.get("download_button")) == 2
    assert any(
        "загрузка и анализ" in caption.value.lower() for caption in app.caption
    )
    assert any(
        "Источник порога" in dataframe.value.columns
        for dataframe in app.dataframe
    )

    payload = app.session_state["analysis_payload"]
    downloaded_result = json.loads(payload["json_content"].decode("utf-8"))
    assert downloaded_result == payload["result"]
    assert payload["html_content"].startswith(b"<!doctype html>")
    assert b"Plotly.newPlot" in payload["html_content"]

    app.selectbox(key="distribution_feature").select("region").run()

    assert not app.exception
    assert len(app.metric) == 4
    assert len(app.get("plotly_chart")) == 1


def test_app_shows_input_error_instead_of_crashing() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=APP_TEST_TIMEOUT)
    app.run()
    app.file_uploader(key="reference_file").set_value(
        ("reference.csv", b"", "text/csv")
    )
    app.file_uploader(key="current_file").set_value(
        ("current.csv", CURRENT_CSV, "text/csv")
    )
    app.run()

    app.button(key="run_analysis").click().run()

    assert not app.exception
    assert len(app.error) == 1
    assert "Не удалось выполнить анализ" in app.error[0].value
    assert "EmptyDataError" in app.error[0].value
    assert len(app.metric) == 0


def test_app_accepts_uploaded_yaml_configuration() -> None:
    app = _loaded_app()
    app.radio(key="config_source").set_value("Загрузить YAML").run()
    app.file_uploader(key="config_file").set_value(
        ("custom.yaml", DEFAULT_CONFIG_PATH.read_bytes(), "application/yaml")
    )
    app.run()

    app.button(key="run_analysis").click().run()

    assert not app.exception
    assert len(app.error) == 0
    assert len(app.metric) == 4
    assert len(app.get("plotly_chart")) == 1


def test_changing_input_marks_previous_result_as_stale() -> None:
    app = _loaded_app()
    app.button(key="run_analysis").click().run()
    assert len(app.metric) == 4

    app.file_uploader(key="current_file").set_value(
        ("changed.csv", REFERENCE_CSV, "text/csv")
    )
    app.run()

    assert not app.exception
    assert len(app.metric) == 0
    assert any(
        "помечен как неактуальный" in message.value.lower()
        for message in app.warning
    )


def test_app_shows_oof_auc_folds_and_feature_importance_when_enabled() -> None:
    app = _loaded_app()
    app.radio(key="config_source").set_value("Загрузить YAML").run()
    enabled_config = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8").replace(
        "enabled: false", "enabled: true"
    )
    app.file_uploader(key="config_file").set_value(
        ("adversarial.yaml", enabled_config.encode(), "application/yaml")
    )
    app.run()

    app.button(key="run_analysis").click().run()

    assert not app.exception
    assert any(metric.label == "Отложенный ROC-AUC" for metric in app.metric)
    assert len(app.dataframe) >= 5
    assert any("причинный эффект" in caption.value.lower() for caption in app.caption)
    assert any("Стратегия split" in caption.value for caption in app.caption)


def test_app_applies_adversarial_override_and_shows_effective_config() -> None:
    app = _loaded_app()
    app.checkbox(key="override_adversarial").set_value(True).run()
    app.checkbox(key="adversarial_enabled").set_value(True)
    app.checkbox(key="use_auc_threshold").set_value(True)
    app.number_input(key="auc_threshold").set_value(0.60)
    app.run()

    app.button(key="run_analysis").click().run()

    assert not app.exception
    effective_config = app.session_state["analysis_payload"]["result"][
        "effective_config"
    ]
    assert effective_config["adversarial"]["enabled"] is True
    assert effective_config["adversarial"]["roc_auc_threshold"] == 0.60
    assert any(
        expander.label == "Фактически применённая конфигурация"
        for expander in app.expander
    )


def test_result_filters_do_not_discard_saved_analysis() -> None:
    app = _loaded_app()
    app.button(key="run_analysis").click().run()
    saved_result = app.session_state["analysis_payload"]["result"]
    saved_elapsed = app.session_state["analysis_payload"]["elapsed_seconds"]

    app.multiselect(key="result_status_filter").set_value(["warning"]).run()

    assert not app.exception
    assert app.session_state["analysis_payload"]["result"] == saved_result
    assert app.session_state["analysis_payload"]["elapsed_seconds"] == saved_elapsed
    assert len(app.metric) == 4
    assert any("фильтрам" in message.value.lower() for message in app.info)


def test_changing_override_marks_result_as_stale() -> None:
    app = _loaded_app()
    app.button(key="run_analysis").click().run()
    assert len(app.metric) == 4

    app.checkbox(key="override_adversarial").set_value(True).run()

    assert not app.exception
    assert app.session_state["analysis_payload"]["stale"] is True
    assert len(app.metric) == 0
    assert any(
        "помечен как неактуальный" in message.value.lower()
        for message in app.warning
    )
