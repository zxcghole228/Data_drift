"""Сохранение и доставка алертов завершённых online-запусков."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .contracts import DeliveryRecord, OnlineAlertConfig, RunRecord
from .storage import MonitoringStore, canonical_json, sha256_bytes

WebhookSender = Callable[[str, bytes, float, str], None]


def _delivery_id(run_id: str, channel: str) -> str:
    digest = sha256_bytes(f"{run_id}:{channel}".encode("utf-8"))[:32]
    return f"delivery_{digest}"


def _validate_webhook_url(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Webhook URL должен использовать http или https")
    return url


def _default_webhook_sender(
    url: str,
    body: bytes,
    timeout_seconds: float,
    delivery_id: str,
) -> None:
    request = Request(
        _validate_webhook_url(url),
        data=body,
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "Idempotency-Key": delivery_id,
            "User-Agent": "data-drift-guardian/0.1",
        },
        method="POST",
    )
    with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
        status = int(getattr(response, "status", response.getcode()))
        if not 200 <= status < 300:
            raise RuntimeError(f"Webhook вернул HTTP {status}")


def _safe_error(exc: Exception) -> str:
    """Вернуть диагностическое сообщение без URL и секретов webhook."""

    if isinstance(exc, HTTPError):
        return f"HTTPError: status {exc.code}"
    if isinstance(exc, TimeoutError):
        return "TimeoutError: webhook timeout"
    if isinstance(exc, URLError):
        return f"URLError: {type(exc.reason).__name__}"
    if isinstance(exc, ValueError):
        return f"ValueError: {exc}"
    return f"{type(exc).__name__}: delivery failed"


class AlertDeliveryService:
    """Идемпотентно доставить сохранённые в Run алерты по включённым каналам."""

    def __init__(
        self,
        store: MonitoringStore,
        config: OnlineAlertConfig,
        *,
        environment: Mapping[str, str] | None = None,
        webhook_sender: WebhookSender | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.store = store
        self.config = dict(config)
        self.environment = os.environ if environment is None else environment
        self.webhook_sender = webhook_sender or _default_webhook_sender
        self.sleep = sleep

    @staticmethod
    def _alert_payload(run: RunRecord) -> dict[str, Any] | None:
        result = run.analysis_result
        if run.status != "completed" or not isinstance(result, dict):
            return None
        alerts = result.get("alerts")
        if not isinstance(alerts, list) or not alerts:
            return None
        return {
            "online_contract_version": "1.0",
            "run_id": run.run_id,
            "reference_id": run.reference_id,
            "source": run.source,
            "source_id": run.source_id,
            "completed_at": run.completed_at,
            "summary": result.get("summary"),
            "alerts": alerts,
        }

    @staticmethod
    def _payload_for_channel(
        payload: Mapping[str, Any],
        *,
        delivery_id: str,
        channel: str,
    ) -> dict[str, Any]:
        return {
            "delivery_id": delivery_id,
            "channel": channel,
            **dict(payload),
        }

    @staticmethod
    def _jsonl_contains(path: Path, delivery_id: str) -> bool:
        if not path.exists():
            return False
        if not path.is_file():
            raise IsADirectoryError(f"Ожидался JSONL-файл: {path}")
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Некорректная JSONL-строка {line_number}: {path}"
                    ) from exc
                if isinstance(record, dict) and record.get("delivery_id") == delivery_id:
                    return True
        return False

    def _deliver_jsonl(
        self,
        run: RunRecord,
        payload: Mapping[str, Any],
        path: Path,
    ) -> DeliveryRecord:
        delivery_id = _delivery_id(run.run_id, "jsonl")
        channel_payload = self._payload_for_channel(
            payload,
            delivery_id=delivery_id,
            channel="jsonl",
        )
        delivery, _ = self.store.add_delivery(
            delivery_id=delivery_id,
            run_id=run.run_id,
            channel="jsonl",
            payload=channel_payload,
        )
        if delivery.status == "succeeded":
            return delivery
        if delivery.attempts >= self.config["max_attempts"]:
            return delivery

        attempt = delivery.attempts + 1
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if not self._jsonl_contains(path, delivery_id):
                with path.open("a", encoding="utf-8", newline="\n") as stream:
                    stream.write(canonical_json(channel_payload) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
        except Exception as exc:
            return self.store.update_delivery(
                delivery_id,
                status="failed",
                attempts=attempt,
                last_error=_safe_error(exc),
            )
        return self.store.update_delivery(
            delivery_id,
            status="succeeded",
            attempts=attempt,
            last_error=None,
        )

    def _deliver_webhook(
        self,
        run: RunRecord,
        payload: Mapping[str, Any],
        url: str,
    ) -> DeliveryRecord:
        delivery_id = _delivery_id(run.run_id, "webhook")
        channel_payload = self._payload_for_channel(
            payload,
            delivery_id=delivery_id,
            channel="webhook",
        )
        delivery, _ = self.store.add_delivery(
            delivery_id=delivery_id,
            run_id=run.run_id,
            channel="webhook",
            payload=channel_payload,
        )
        if delivery.status == "succeeded":
            return delivery

        max_attempts = self.config["max_attempts"]
        timeout = self.config["timeout_seconds"]
        attempts = delivery.attempts
        if attempts >= max_attempts:
            return delivery

        body = canonical_json(channel_payload).encode("utf-8")
        while attempts < max_attempts:
            attempts += 1
            try:
                self.webhook_sender(url, body, timeout, delivery_id)
            except Exception as exc:
                final = attempts >= max_attempts
                delivery = self.store.update_delivery(
                    delivery_id,
                    status="failed" if final else "pending",
                    attempts=attempts,
                    last_error=_safe_error(exc),
                )
                if not final:
                    self.sleep(min(2.0 ** (attempts - 1), 5.0))
                continue
            return self.store.update_delivery(
                delivery_id,
                status="succeeded",
                attempts=attempts,
                last_error=None,
            )
        return delivery

    def dispatch_run(self, run: RunRecord) -> list[DeliveryRecord]:
        """Доставить алерты Run; отсутствие алертов не создаёт Delivery."""

        payload = self._alert_payload(run)
        if payload is None:
            return []

        deliveries: list[DeliveryRecord] = []
        jsonl_path = self.config["jsonl_path"]
        if jsonl_path is not None:
            deliveries.append(self._deliver_jsonl(run, payload, Path(jsonl_path)))

        env_name = self.config["webhook_url_env"]
        webhook_url = self.environment.get(env_name, "").strip() if env_name else ""
        if webhook_url:
            deliveries.append(self._deliver_webhook(run, payload, webhook_url))
        return deliveries
