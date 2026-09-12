
from __future__ import annotations

import re

MAX_SLUG_LENGTH = 64
_SEGMENT_RE = re.compile(r"^[a-zA-Z0-9._-]+$")
_WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def validate_slug(name: str) -> str | None:
    if not name:
        return "name cannot be empty"
    if len(name) > MAX_SLUG_LENGTH:
        return f"name too long (max {MAX_SLUG_LENGTH} characters)"


    segments = name.split("/")
    for seg in segments:
        if not seg:
            return "name contains empty segment"
        if seg in (".", ".."):
            return "name must not contain '.' or '..' as a segment"
        if not _SEGMENT_RE.match(seg):
            return f"invalid segment: {seg!r} (allowed: letters, digits, '.', '-', '_')"
        if seg.endswith((".", " ")):
            return "name segments must not end with a dot or space"
        if seg.split(".", 1)[0].upper() in _WINDOWS_RESERVED:
            return f"invalid Windows reserved name: {seg!r}"


    return None


def flatten_slug(name: str) -> str:
    return name.replace("/", "+")
