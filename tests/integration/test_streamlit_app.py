"""Интеграционные проверки пользовательского интерфейса Streamlit."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[2] / "app" / "streamlit_app.py"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"

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
    app = AppTest.from_file(str(APP_PATH), default_timeout=10)
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
    assert any(
        "явно запустите анализ" in message.value.lower() for message in app.info
    )


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

    app.selectbox(key="distribution_feature").select("region").run()

    assert not app.exception
    assert len(app.metric) == 4
    assert len(app.get("plotly_chart")) == 1


def test_app_shows_input_error_instead_of_crashing() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=10)
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


def test_changing_input_clears_previous_result() -> None:
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
        "явно запустите анализ" in message.value.lower() for message in app.info
    )
