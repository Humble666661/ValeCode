from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from valecode.persistence.models import InvalidTransitionError


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def dump_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def load_json(value: str | None, default: Any) -> Any:
    if value is None:
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


def require_transition(
    entity: str,
    current: StrEnum,
    target: StrEnum,
    transitions: dict[Any, frozenset[Any]],
) -> None:
    if current == target:
        return
    if target not in transitions[current]:
        raise InvalidTransitionError(entity, current.value, target.value)
