from __future__ import annotations

import json
import re
from typing import Any

_SENSITIVE_KEY = re.compile(
    r"(api[_-]?key|authorization|token|secret|password|cookie|credential)", re.I
)
_CONTENT_KEY = re.compile(r"(prompt|input|arguments|content|output|result)", re.I)


def _safe_value(value: Any) -> str | bool | int | float:
    if isinstance(value, (str, bool, int, float)):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return repr(value)


def sanitize_attributes(
    attributes: dict[str, Any] | None,
    *,
    capture_content: bool = False,
    max_length: int = 2_000,
) -> dict[str, str | bool | int | float]:
    sanitized: dict[str, str | bool | int | float] = {}
    for key, raw_value in (attributes or {}).items():
        name = str(key)
        if _SENSITIVE_KEY.search(name):
            sanitized[name] = "[REDACTED]"
            continue
        if _CONTENT_KEY.search(name) and not capture_content:
            sanitized[name] = "[REDACTED]"
            continue
        value = _safe_value(raw_value)
        if isinstance(value, str) and len(value) > max_length:
            value = value[:max_length] + "…"
        sanitized[name] = value
    return sanitized
