"""Read-only Streamlit-представление persistent online-состояния."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from .config import load_monitoring_config
from .contracts import RunRecord
from .storage import MonitoringStore

ONLINE_CONFIG_ENV = "DDG_ONLINE_CONFIG"
ONLINE_STATE_ENV = "DDG_ONLINE_STATE"


def _summary(run: RunRecord) -> dict[str, Any]:
    if not isinstance(run.analysis_result, dict):
        return {}
    summary = run.analysis_result.get("summary")
    return summary if isinstance(summary, dict) else {}


def _table_height(row_count: int, *, maximum_rows: int = 10) -> int:
    """Ограничить таблицу содержимым и избежать больших пустых областей."""

    visible_rows = min(max(row_count, 1), maximum_rows)
    return 39 + visible_rows * 35


def _show_table(rows: list[dict[str, object]]) -> None:
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
        height=_table_height(len(rows)),
    )


def collect_dashboard_snapshot(
    state_path: str | Path,
    *,
    window_size: int,
    run_limit: int = 100,
) -> dict[str, Any]:
    """Собрать один согласованный read-only снимок состояния для интерфейса."""

    if not isinstance(state_path, (str, Path)):
        raise TypeError("state_path должен иметь тип str или pathlib.Path")
    if type(window_size) is not int or window_size < 2:
        raise ValueError("window_size должен быть целым числом не меньше 2")
    if type(run_limit) is not int or run_limit < 1:
        raise ValueError("run_limit должен быть положительным целым числом")

    database_path = Path(state_path).expanduser()
    if not database_path.exists():
        return {
            "initialized": False,
            "state_path": str(database_path),
            "schema_version": None,
            "readiness": "not_initialized",
            "reference": None,
            "buffered_rows": 0,
            "claimed_rows": 0,
            "window_size": window_size,
            "runs": [],
            "alerts": [],
            "deliveries": [],
            "performance": [],
            "run_records": {},
        }

    store = MonitoringStore(database_path)
    reference = store.get_active_reference()
    runs = store.list_runs(limit=run_limit)
    run_rows: list[dict[str, object]] = []
    alert_rows: list[dict[str, object]] = []
    delivery_rows: list[dict[str, object]] = []
    performance_rows: list[dict[str, object]] = []

    for run in runs:
        summary = _summary(run)
        performance = store.get_performance(run.run_id)
        deliveries = store.list_deliveries(run_id=run.run_id)
        run_rows.append(
            {
                "Run": run.run_id,
                "Создан": run.created_at,
                "Источник": run.source,
                "Source ID": run.source_id,
                "Reference": run.reference_id,
                "Строк": run.row_count,
                "Run status": run.status,
                "Analysis status": summary.get("status", "—"),
                "Алертов": summary.get("n_alerts", "—"),
                "Delivery": ", ".join(
                    f"{delivery.channel}:{delivery.status}"
                    for delivery in deliveries
                )
                or "—",
                "Performance": performance.status if performance else "—",
                "Ошибка": run.error or "—",
            }
        )

        if isinstance(run.analysis_result, dict):
            alerts = run.analysis_result.get("alerts")
            if isinstance(alerts, list):
                for alert in alerts:
                    if not isinstance(alert, dict):
                        continue
                    alert_rows.append(
                        {
                            "Run": run.run_id,
                            "Завершён": run.completed_at or "—",
                            "Источник": alert.get("source", "—"),
                            "Проверка": alert.get("check", "—"),
                            "Признак": alert.get("feature") or "—",
                            "Severity": alert.get("severity", "—"),
                            "Сообщение": alert.get("message", "—"),
                        }
                    )

        for delivery in deliveries:
            delivery_rows.append(
                {
                    "Run": run.run_id,
                    "Delivery": delivery.delivery_id,
                    "Канал": delivery.channel,
                    "Статус": delivery.status,
                    "Попыток": delivery.attempts,
                    "Ошибка": delivery.last_error or "—",
                    "Обновлён": delivery.updated_at,
                }
            )

        if performance is not None:
            result = performance.result
            accuracy = result.get("accuracy")
            roc_auc = result.get("roc_auc")
            accuracy = accuracy if isinstance(accuracy, dict) else {}
            roc_auc = roc_auc if isinstance(roc_auc, dict) else {}
            performance_rows.append(
                {
                    "Run": run.run_id,
                    "Статус": performance.status,
                    "Feedback rows": result.get("feedback_rows", "—"),
                    "Accuracy": accuracy.get("value", "—"),
                    "Accuracy alert": accuracy.get("alert", "—"),
                    "ROC-AUC": roc_auc.get("value", "—"),
                    "ROC-AUC alert": roc_auc.get("alert", "—"),
                    "Причина": result.get("reason") or "—",
                    "Обновлён": performance.updated_at,
                }
            )

    reference_view = None
    if reference is not None:
        reference_view = {
            "reference_id": reference.reference_id,
            "name": reference.name,
            "rows": reference.row_count,
            "format": reference.source_format,
            "columns": list(reference.columns),
            "created_at": reference.created_at,
        }

    return {
        "initialized": True,
        "state_path": str(database_path),
        "schema_version": store.schema_version,
        "readiness": "ready" if reference is not None else "not_ready",
        "reference": reference_view,
        "buffered_rows": store.count_events(state="buffered"),
        "claimed_rows": store.count_events(state="claimed"),
        "window_size": window_size,
        "runs": run_rows,
        "alerts": alert_rows,
        "deliveries": delivery_rows,
        "performance": performance_rows,
        "run_records": {run.run_id: run.analysis_result for run in runs},
    }


def _render_reference(snapshot: dict[str, Any]) -> None:
    reference = snapshot["reference"]
    st.subheader("Активный Reference")
    if reference is None:
        st.warning("Активный Reference отсутствует; online-сервис не готов к анализу.")
        return

    first, second, third = st.columns(3)
    first.metric("Reference ID", reference["reference_id"])
    second.metric("Строк", reference["rows"])
    third.metric("Формат", reference["format"])
    st.caption(
        f"{reference['name']} · создан {reference['created_at']} · "
        f"колонки: {', '.join(reference['columns'])}"
    )


def _render_section(
    title: str,
    rows: list[dict[str, object]],
    *,
    empty_message: str,
) -> None:
    st.subheader(title)
    if rows:
        _show_table(rows)
    else:
        st.info(empty_message)


def render_online_dashboard(default_config_path: str | Path) -> None:
    """Показать online-историю без повторного запуска анализа."""

    st.title("🛡️ Data Drift Guardian · Online")
    st.write(
        "Read-only панель persistent-состояния: Reference, буфер, Run, алерты, "
        "доставка и запаздывающий Feedback."
    )

    configured_path = os.environ.get(ONLINE_CONFIG_ENV, str(default_config_path))
    with st.sidebar:
        st.header("Online-панель")
        config_path = st.text_input(
            "Monitoring config",
            value=configured_path,
            key="online_config_path",
        )

    try:
        config = load_monitoring_config(config_path)
    except Exception as exc:
        st.error(
            "Не удалось прочитать online-конфигурацию: "
            f"{type(exc).__name__}: {exc}"
        )
        return

    default_state_path = os.environ.get(
        ONLINE_STATE_ENV,
        config["online"]["state_path"],
    )
    with st.sidebar:
        state_path = st.text_input(
            "SQLite state",
            value=default_state_path,
            key="online_state_path",
        )
        run_limit = int(
            st.number_input(
                "Последних Run",
                min_value=1,
                max_value=500,
                value=100,
                step=10,
                key="online_run_limit",
            )
        )
        st.button("Обновить состояние", key="online_refresh", width="stretch")
        st.caption("Любое действие перезагружает снимок SQLite; pipeline не запускается.")

    try:
        snapshot = collect_dashboard_snapshot(
            state_path,
            window_size=config["online"]["window"]["size"],
            run_limit=run_limit,
        )
    except Exception as exc:
        st.error(
            "Не удалось прочитать online-состояние: "
            f"{type(exc).__name__}: {exc}"
        )
        return

    if not snapshot["initialized"]:
        st.warning(
            "SQLite-база ещё не создана. Запустите Online API и зарегистрируйте "
            f"Reference. Ожидаемый путь: `{snapshot['state_path']}`."
        )
        return

    first, second, third, fourth = st.columns(4)
    first.metric("Readiness", snapshot["readiness"])
    second.metric("Буфер", snapshot["buffered_rows"])
    third.metric("Размер окна", snapshot["window_size"])
    fourth.metric("Run в снимке", len(snapshot["runs"]))
    st.caption(
        f"SQLite schema v{snapshot['schema_version']} · `{snapshot['state_path']}` · "
        f"claimed rows: {snapshot['claimed_rows']}. Readiness здесь отражает "
        "доступность состояния и Reference, а не liveness процесса API."
    )

    _render_reference(snapshot)
    _render_section(
        "История Run",
        snapshot["runs"],
        empty_message="Run ещё не создавались.",
    )
    _render_section(
        "Алерты",
        snapshot["alerts"],
        empty_message="В сохранённых Run алертов нет.",
    )
    _render_section(
        "Доставка алертов",
        snapshot["deliveries"],
        empty_message="Попытки доставки ещё не создавались.",
    )
    _render_section(
        "Performance / delayed Feedback",
        snapshot["performance"],
        empty_message="Оценок качества по Feedback пока нет.",
    )

    if snapshot["run_records"]:
        with st.expander("Полный AnalysisResult выбранного Run", expanded=False):
            selected = st.selectbox(
                "Run",
                options=list(snapshot["run_records"]),
                key="online_selected_run",
            )
            result = snapshot["run_records"][selected]
            if result is None:
                st.info("У этого Run ещё нет сохранённого результата.")
            else:
                st.json(result)
