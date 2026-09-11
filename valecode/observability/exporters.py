from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any


class JsonlSpanExporter:
    """Small local exporter implementing the OpenTelemetry SpanExporter API."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def export(self, spans: list[Any]):
        from opentelemetry.sdk.trace.export import SpanExportResult

        try:
            lines = [json.dumps(self._serialize(span), ensure_ascii=False) for span in spans]
            with self._lock, self.path.open("a", encoding="utf-8") as handle:
                for line in lines:
                    handle.write(line + "\n")
                handle.flush()
            return SpanExportResult.SUCCESS
        except Exception:
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        return

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True

    @staticmethod
    def _serialize(span: Any) -> dict[str, Any]:
        context = span.context
        parent = span.parent
        return {
            "name": span.name,
            "trace_id": f"{context.trace_id:032x}",
            "span_id": f"{context.span_id:016x}",
            "parent_span_id": f"{parent.span_id:016x}" if parent else None,
            "start_time_unix_nano": span.start_time,
            "end_time_unix_nano": span.end_time,
            "status": span.status.status_code.name,
            "attributes": dict(span.attributes or {}),
            "events": [
                {
                    "name": event.name,
                    "timestamp": event.timestamp,
                    "attributes": dict(event.attributes or {}),
                }
                for event in span.events
            ],
        }
