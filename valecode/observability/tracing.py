from __future__ import annotations

import hashlib
import logging
import os
import secrets
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from valecode.observability.attributes import sanitize_attributes

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TracingConfig:
    enabled: bool = False
    service_name: str = "valecode"
    exporter: str = "none"  # none | console | file | otlp
    endpoint: str | None = None
    file_path: str | None = None
    capture_content: bool = False


class SpanHandle:
    def __init__(self, span: Any = None, *, capture_content: bool = False) -> None:
        self._span = span
        self._capture_content = capture_content

    def set_attributes(self, attributes: dict[str, Any]) -> None:
        if self._span is None:
            return
        for key, value in sanitize_attributes(
            attributes, capture_content=self._capture_content
        ).items():
            try:
                self._span.set_attribute(key, value)
            except Exception as exc:
                log.debug("Unable to set span attribute %s: %s", key, exc)

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        if self._span is not None:
            try:
                self._span.add_event(
                    name,
                    sanitize_attributes(
                        attributes, capture_content=self._capture_content
                    ),
                )
            except Exception as exc:
                log.debug("Unable to add span event %s: %s", name, exc)

    def set_error(self, description: str) -> None:
        if self._span is None:
            return
        try:
            from opentelemetry.trace import Status, StatusCode

            safe_description = description if self._capture_content else "[REDACTED]"
            self._span.set_status(Status(StatusCode.ERROR, safe_description))
        except Exception as exc:
            log.debug("Unable to set span error status: %s", exc)

    def record_exception(self, exception: BaseException) -> None:
        if self._span is None:
            return
        try:
            self._span.add_event(
                "exception",
                sanitize_attributes(
                    {
                        "exception.type": type(exception).__name__,
                        "exception.message": (
                            str(exception) if self._capture_content else "[REDACTED]"
                        ),
                    },
                    capture_content=self._capture_content,
                ),
            )
        except Exception as exc:
            log.debug("Unable to record span exception: %s", exc)
        self.set_error(str(exception))


class Tracing:
    def __init__(
        self,
        tracer: Any = None,
        *,
        enabled: bool = False,
        capture_content: bool = False,
        provider: Any = None,
    ) -> None:
        self.tracer = tracer
        self.enabled = enabled and tracer is not None
        self.capture_content = capture_content
        self.provider = provider

    @contextmanager
    def span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
        *,
        trace_id: str | None = None,
    ) -> Iterator[SpanHandle]:
        if not self.enabled:
            yield SpanHandle()
            return
        try:
            context = _parent_context(trace_id) if trace_id else None
            manager = self.tracer.start_as_current_span(name, context=context)
            raw_span = manager.__enter__()
        except Exception as exc:
            log.warning("Unable to start span %s: %s", name, exc)
            yield SpanHandle()
            return

        handle = SpanHandle(raw_span, capture_content=self.capture_content)
        try:
            handle.set_attributes(attributes or {})
        except Exception as exc:
            log.debug("Unable to set span attributes: %s", exc)
        error_info = (None, None, None)
        try:
            yield handle
        except BaseException as exc:
            error_info = sys.exc_info()
            try:
                handle.record_exception(exc)
            except Exception as trace_exc:
                log.debug("Unable to record span exception: %s", trace_exc)
            raise
        finally:
            try:
                manager.__exit__(*error_info)
            except Exception as exc:
                # Never replace an application exception with a tracing failure.
                log.warning("Unable to finish span %s: %s", name, exc)

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        if self.provider is None:
            return True
        try:
            return bool(self.provider.force_flush(timeout_millis))
        except Exception as exc:
            log.warning("OpenTelemetry flush failed: %s", exc)
            return False

    def shutdown(self) -> None:
        if self.provider is None:
            return
        try:
            self.provider.shutdown()
        except Exception as exc:
            log.warning("OpenTelemetry shutdown failed: %s", exc)


_tracing = Tracing()


def get_tracing() -> Tracing:
    return _tracing


def configure_tracing(config: TracingConfig) -> Tracing:
    global _tracing
    if not config.enabled:
        _tracing = Tracing()
        return _tracing
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            ConsoleSpanExporter,
            SimpleSpanProcessor,
        )

        provider = TracerProvider(
            resource=Resource.create({"service.name": config.service_name})
        )
        if config.exporter == "console":
            provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        elif config.exporter == "file":
            from valecode.observability.exporters import JsonlSpanExporter

            path = config.file_path or ".valecode/traces.jsonl"
            provider.add_span_processor(BatchSpanProcessor(JsonlSpanExporter(path)))
        elif config.exporter == "otlp":
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            options = {"endpoint": config.endpoint} if config.endpoint else {}
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(**options)))
        elif config.exporter != "none":
            raise ValueError(f"Unsupported trace exporter: {config.exporter}")
        tracer = provider.get_tracer("valecode")
        _tracing = Tracing(
            tracer,
            enabled=True,
            capture_content=config.capture_content,
            provider=provider,
        )
    except Exception as exc:
        log.warning("OpenTelemetry setup failed; tracing disabled: %s", exc)
        _tracing = Tracing()
    return _tracing


def configure_tracing_from_env(work_dir: str | Path = ".") -> Tracing:
    from dotenv import dotenv_values

    values: dict[str, str] = {}
    for path in (
        Path.home() / ".valecode" / ".env",
        Path(work_dir) / ".env",
        Path(work_dir) / ".env.local",
    ):
        if path.is_file():
            values.update(
                {key: value for key, value in dotenv_values(path).items() if value is not None}
            )
    values.update(os.environ)
    enabled = values.get("VALECODE_OTEL_ENABLED", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    capture_content = values.get("VALECODE_OTEL_CAPTURE_CONTENT", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    file_path = values.get("VALECODE_OTEL_FILE")
    if file_path is None:
        file_path = str(Path(work_dir) / ".valecode" / "traces.jsonl")
    return configure_tracing(
        TracingConfig(
            enabled=enabled,
            service_name=values.get("VALECODE_OTEL_SERVICE_NAME", "valecode"),
            exporter=values.get("VALECODE_OTEL_EXPORTER", "none").lower(),
            endpoint=values.get("VALECODE_OTEL_ENDPOINT") or None,
            file_path=file_path,
            capture_content=capture_content,
        )
    )


def _parent_context(trace_id: str):
    from opentelemetry import trace

    normalized = trace_id.replace("-", "")
    try:
        numeric_trace_id = int(normalized, 16)
    except ValueError:
        numeric_trace_id = int(hashlib.sha256(trace_id.encode()).hexdigest()[:32], 16)
    numeric_trace_id &= (1 << 128) - 1
    if numeric_trace_id == 0:
        numeric_trace_id = 1
    span_context = trace.SpanContext(
        trace_id=numeric_trace_id,
        span_id=secrets.randbits(64) or 1,
        is_remote=True,
        trace_flags=trace.TraceFlags(trace.TraceFlags.SAMPLED),
        trace_state=trace.TraceState(),
    )
    return trace.set_span_in_context(trace.NonRecordingSpan(span_context))
