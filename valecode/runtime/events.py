from __future__ import annotations

import asyncio
import uuid
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


@dataclass(frozen=True)
class EventEnvelope:
    event_id: str
    event_type: str
    sequence: int
    emitted_at: str
    session_id: str | None
    run_id: str | None
    step_id: str | None
    tool_call_id: str | None
    trace_id: str | None
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "sequence": self.sequence,
            "emitted_at": self.emitted_at,
            "session_id": self.session_id,
            "run_id": self.run_id,
            "step_id": self.step_id,
            "tool_call_id": self.tool_call_id,
            "trace_id": self.trace_id,
            "payload": self.payload,
        }


@dataclass(kw_only=True)
class RuntimeEvent:
    """Common metadata shared by every event emitted from the Agent runtime."""

    EVENT_TYPE: ClassVar[str] = "runtime.event"

    event_id: str = field(default_factory=lambda: f"evt_{uuid.uuid4().hex}")
    sequence: int = 0
    emitted_at: str = field(default_factory=_utc_now)
    session_id: str | None = None
    run_id: str | None = None
    step_id: str | None = None
    tool_call_id: str | None = None
    trace_id: str | None = None

    def stamp(
        self,
        *,
        sequence: int,
        session_id: str | None,
        run_id: str | None,
        step_id: str | None,
        tool_call_id: str | None,
        trace_id: str | None,
    ) -> RuntimeEvent:
        if not self.event_id:
            self.event_id = f"evt_{uuid.uuid4().hex}"
        if not self.emitted_at:
            self.emitted_at = _utc_now()
        self.sequence = sequence
        self.session_id = session_id
        self.run_id = run_id
        self.step_id = step_id
        self.tool_call_id = tool_call_id
        self.trace_id = trace_id
        return self

    def to_envelope(self) -> EventEnvelope:
        metadata = {
            "event_id",
            "sequence",
            "emitted_at",
            "session_id",
            "run_id",
            "step_id",
            "tool_call_id",
            "trace_id",
        }
        payload: dict[str, Any] = {}
        for item in fields(self):
            if item.name in metadata:
                continue
            value = getattr(self, item.name)
            if isinstance(value, asyncio.Future):
                continue
            payload[item.name] = _json_value(value)
        return EventEnvelope(
            event_id=self.event_id,
            event_type=self.EVENT_TYPE,
            sequence=self.sequence,
            emitted_at=self.emitted_at,
            session_id=self.session_id,
            run_id=self.run_id,
            step_id=self.step_id,
            tool_call_id=self.tool_call_id,
            trace_id=self.trace_id,
            payload=payload,
        )
