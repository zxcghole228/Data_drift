"""FastAPI transport для persistent online-мониторинга."""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Literal, cast
from uuid import uuid4

from fastapi import FastAPI, Query, Request, Response, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .config import load_monitoring_config
from .contracts import ReferenceRecord, RunRecord, RunStatus
from .monitor import (
    NoActiveReferenceError,
    OnlineMonitor,
    ReferenceConfigMismatchError,
)
from .storage import (
    IdempotencyConflictError,
    InvalidStateTransitionError,
    MonitoringStore,
    PendingEventsError,
    RecordNotFoundError,
)

DEFAULT_CONFIG_PATH = Path("configs/online.yaml")
CONFIG_ENV = "DDG_ONLINE_CONFIG"
STATE_ENV = "DDG_ONLINE_STATE"


class EventRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(min_length=1, max_length=256)
    occurred_at: str = Field(min_length=1, max_length=128)
    features: dict[str, Any] = Field(min_length=1)
    prediction: Any | None = None
    score: float | None = None


class EventsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    events: list[EventRequest] = Field(min_length=1, max_length=10_000)


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_id: str = Field(min_length=1, max_length=256)
    records: list[dict[str, Any]] = Field(min_length=1, max_length=100_000)

    @field_validator("records")
    @classmethod
    def records_must_not_contain_empty_rows(
        cls,
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if any(not record for record in records):
            raise ValueError("records не должен содержать пустые строки")
        return records


class FeedbackItem(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    event_id: str = Field(min_length=1, max_length=256)
    y_true: bool | int | float | str


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    feedback: list[FeedbackItem] = Field(min_length=1, max_length=10_000)


class RequestBodyTooLargeError(ValueError):
    pass


def _error_response(
    *,
    status_code: int,
    code: str,
    message: str,
    details: object | None = None,
) -> JSONResponse:
    error: dict[str, Any] = {"code": code, "message": message}
    if details is not None:
        error["details"] = details
    return JSONResponse(status_code=status_code, content={"error": error})


def _reference_response(reference: ReferenceRecord) -> dict[str, Any]:
    return {
        "reference_id": reference.reference_id,
        "name": reference.name,
        "content_sha256": reference.content_sha256,
        "config_sha256": reference.config_sha256,
        "source_format": reference.source_format,
        "row_count": reference.row_count,
        "columns": list(reference.columns),
        "dtypes": dict(reference.dtypes),
        "created_at": reference.created_at,
        "active": reference.active,
    }


def _run_summary(
    run: RunRecord,
    monitor: OnlineMonitor,
) -> dict[str, Any]:
    analysis_summary = (
        run.analysis_result.get("summary")
        if isinstance(run.analysis_result, dict)
        else None
    )
    return {
        "run_id": run.run_id,
        "source": run.source,
        "source_id": run.source_id,
        "status": run.status,
        "reference_id": run.reference_id,
        "row_count": run.row_count,
        "created_at": run.created_at,
        "completed_at": run.completed_at,
        "summary": analysis_summary,
        "error": run.error,
        "deliveries": [
            monitor.delivery_summary(delivery)
            for delivery in monitor.store.list_deliveries(run_id=run.run_id)
        ],
        "performance": (
            None
            if (performance := monitor.store.get_performance(run.run_id)) is None
            else monitor.performance_summary(performance)
        ),
    }


async def _read_limited_body(request: Request, limit: int) -> bytes:
    content = bytearray()
    async for chunk in request.stream():
        content.extend(chunk)
        if len(content) > limit:
            raise RequestBodyTooLargeError(
                f"Размер тела запроса превышает допустимые {limit} байт"
            )
    if not content:
        raise ValueError("Тело Reference не должно быть пустым")
    return bytes(content)


def create_app(
    config_path: str | Path | None = None,
    *,
    state_path: str | Path | None = None,
) -> FastAPI:
    """Создать изолированное приложение; используется Uvicorn factory mode."""

    configured_path = config_path or os.environ.get(CONFIG_ENV) or DEFAULT_CONFIG_PATH
    config = load_monitoring_config(configured_path)
    if not config["online"]["enabled"]:
        raise ValueError("Online API нельзя запустить при online.enabled=false")

    configured_state = state_path or os.environ.get(STATE_ENV)
    if configured_state is not None:
        config["online"]["state_path"] = str(configured_state)

    store = MonitoringStore(config["online"]["state_path"])
    monitor = OnlineMonitor(store, config)
    max_request_bytes = config["online"]["max_request_bytes"]

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        del app
        if store.get_active_reference() is not None:
            monitor.process_ready_windows()
        yield

    app = FastAPI(
        title="Data Drift Guardian Online API",
        version="1.0",
        description=(
            "Приём production-событий и micro-batch анализ через общее "
            "статистическое ядро Data Drift Guardian."
        ),
        lifespan=lifespan,
    )
    app.state.store = store
    app.state.monitor = monitor
    app.state.config_path = str(configured_path)

    @app.middleware("http")
    async def reject_oversized_content_length(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        raw_length = request.headers.get("content-length")
        if raw_length is not None:
            try:
                content_length = int(raw_length)
            except ValueError:
                return _error_response(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    code="invalid_content_length",
                    message="Заголовок Content-Length должен быть целым числом",
                )
            if content_length > max_request_bytes:
                return _error_response(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    code="request_too_large",
                    message=(
                        "Размер тела запроса превышает допустимые "
                        f"{max_request_bytes} байт"
                    ),
                )
        return await call_next(request)

    @app.exception_handler(RequestValidationError)
    async def request_validation_error_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        del request
        return _error_response(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            code="request_validation_error",
            message="HTTP-запрос не соответствует контракту",
            details=jsonable_encoder(exc.errors()),
        )

    @app.exception_handler(RequestBodyTooLargeError)
    async def request_too_large_handler(
        request: Request,
        exc: RequestBodyTooLargeError,
    ) -> JSONResponse:
        del request
        return _error_response(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            code="request_too_large",
            message=str(exc),
        )

    @app.exception_handler(IdempotencyConflictError)
    async def idempotency_error_handler(
        request: Request,
        exc: IdempotencyConflictError,
    ) -> JSONResponse:
        del request
        return _error_response(
            status_code=status.HTTP_409_CONFLICT,
            code="idempotency_conflict",
            message=str(exc),
        )

    @app.exception_handler(PendingEventsError)
    async def pending_events_handler(
        request: Request,
        exc: PendingEventsError,
    ) -> JSONResponse:
        del request
        return _error_response(
            status_code=status.HTTP_409_CONFLICT,
            code="pending_events",
            message=str(exc),
        )

    @app.exception_handler(InvalidStateTransitionError)
    async def state_transition_handler(
        request: Request,
        exc: InvalidStateTransitionError,
    ) -> JSONResponse:
        del request
        return _error_response(
            status_code=status.HTTP_409_CONFLICT,
            code="invalid_state_transition",
            message=str(exc),
        )

    @app.exception_handler(RecordNotFoundError)
    async def not_found_handler(
        request: Request,
        exc: RecordNotFoundError,
    ) -> JSONResponse:
        del request
        return _error_response(
            status_code=status.HTTP_404_NOT_FOUND,
            code="not_found",
            message=str(exc),
        )

    @app.exception_handler(NoActiveReferenceError)
    async def no_reference_handler(
        request: Request,
        exc: NoActiveReferenceError,
    ) -> JSONResponse:
        del request
        return _error_response(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            code="no_active_reference",
            message=str(exc),
        )

    @app.exception_handler(ReferenceConfigMismatchError)
    async def config_mismatch_handler(
        request: Request,
        exc: ReferenceConfigMismatchError,
    ) -> JSONResponse:
        del request
        return _error_response(
            status_code=status.HTTP_409_CONFLICT,
            code="reference_config_mismatch",
            message=str(exc),
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(
        request: Request,
        exc: ValueError,
    ) -> JSONResponse:
        del request
        return _error_response(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            code="invalid_input",
            message=str(exc),
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        del request, exc
        incident_id = f"incident_{uuid4().hex}"
        return _error_response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code="internal_error",
            message="Непредвиденная внутренняя ошибка",
            details={"incident_id": incident_id},
        )

    @app.get("/health/live")
    def liveness() -> dict[str, str]:
        return {"status": "alive"}

    @app.get("/health/ready")
    def readiness(response: Response) -> dict[str, Any]:
        reference = store.get_active_reference()
        if reference is None:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return {
                "status": "not_ready",
                "reason": "active_reference_missing",
                "schema_version": store.schema_version,
            }
        try:
            monitor.get_active_reference()
        except ReferenceConfigMismatchError:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return {
                "status": "not_ready",
                "reason": "reference_config_mismatch",
                "schema_version": store.schema_version,
                "reference_id": reference.reference_id,
            }
        return {
            "status": "ready",
            "schema_version": store.schema_version,
            "reference_id": reference.reference_id,
            "buffered_rows": store.count_events(state="buffered"),
        }

    @app.post("/api/v1/references", status_code=status.HTTP_201_CREATED)
    async def register_reference(
        request: Request,
        file_format: Literal["csv", "parquet"] = Query(alias="format"),
        name: str | None = Query(default=None, min_length=1, max_length=200),
        reference_id: str | None = Query(default=None, min_length=1, max_length=256),
        activate: bool = Query(default=False),
        discard_buffered: bool = Query(default=False),
    ) -> dict[str, Any]:
        content = await _read_limited_body(request, max_request_bytes)
        temporary_path: Path | None = None
        try:
            with NamedTemporaryFile(delete=False, suffix=f".{file_format}") as stream:
                temporary_path = Path(stream.name)
                stream.write(content)
            reference = monitor.register_reference(
                temporary_path,
                name=name or reference_id or "reference",
                reference_id=reference_id,
                activate=activate,
                discard_buffered=discard_buffered,
            )
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return _reference_response(reference)

    @app.post("/api/v1/references/{reference_id}/activate")
    def activate_reference(
        reference_id: str,
        discard_buffered: bool = Query(default=False),
    ) -> dict[str, Any]:
        return _reference_response(
            monitor.activate_reference(
                reference_id,
                discard_buffered=discard_buffered,
            )
        )

    @app.get("/api/v1/references/current")
    def current_reference() -> dict[str, Any]:
        reference = store.get_active_reference()
        if reference is None:
            raise RecordNotFoundError("Активный Reference отсутствует")
        return _reference_response(reference)

    @app.post("/api/v1/events")
    def ingest_events(
        payload: EventsRequest,
        response: Response,
    ) -> dict[str, Any]:
        result = monitor.ingest_events(
            [event.model_dump(mode="json") for event in payload.events]
        )
        response.status_code = (
            status.HTTP_201_CREATED
            if result["run_ids"]
            else status.HTTP_202_ACCEPTED
        )
        return dict(result)

    @app.post("/api/v1/batches", status_code=status.HTTP_201_CREATED)
    def process_batch(payload: BatchRequest) -> dict[str, Any]:
        run = monitor.process_batch(
            batch_id=payload.batch_id,
            records=payload.records,
        )
        return monitor.run_envelope(run)

    @app.post("/api/v1/feedback")
    def ingest_feedback(
        payload: FeedbackRequest,
        response: Response,
    ) -> dict[str, Any]:
        result = monitor.ingest_feedback(
            [item.model_dump(mode="json") for item in payload.feedback]
        )
        response.status_code = (
            status.HTTP_201_CREATED
            if result["accepted"]
            else status.HTTP_200_OK
        )
        return dict(result)

    @app.get("/api/v1/runs")
    def list_runs(
        run_status: Literal["running", "completed", "failed"] | None = Query(
            default=None,
            alias="status",
        ),
        limit: int = Query(default=100, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        runs = store.list_runs(
            status=cast(RunStatus | None, run_status),
            limit=limit,
            offset=offset,
        )
        return {
            "items": [_run_summary(run, monitor) for run in runs],
            "limit": limit,
            "offset": offset,
        }

    @app.get("/api/v1/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        return monitor.run_envelope(store.get_run(run_id))

    return app
