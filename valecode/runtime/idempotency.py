from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


def canonical_arguments(arguments: dict[str, Any]) -> str:
    """Return a stable representation independent of dictionary insertion order."""
    return json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def make_tool_idempotency_key(
    run_id: str,
    provider_tool_call_id: str,
    tool_name: str,
    arguments: dict[str, Any],
) -> str:
    payload = "\0".join(
        (run_id, provider_tool_call_id, tool_name, canonical_arguments(arguments))
    )
    return "tool:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


class FileEffectState(StrEnum):
    APPLIED = "applied"
    NOT_APPLIED = "not_applied"
    AMBIGUOUS = "ambiguous"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class FileEffectInspection:
    state: FileEffectState
    reason: str
    path: str | None = None


def _resolve_path(raw_path: str, work_dir: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = Path(work_dir) / path
    return path.absolute()


def inspect_file_effect(
    tool_name: str,
    arguments: dict[str, Any],
    work_dir: str | Path,
) -> FileEffectInspection:
    """Inspect deterministic file tools without changing the filesystem."""
    raw_path = arguments.get("file_path")
    if not isinstance(raw_path, str) or not raw_path:
        return FileEffectInspection(
            FileEffectState.UNSUPPORTED, "Tool has no inspectable file_path"
        )
    path = _resolve_path(raw_path, work_dir)

    if tool_name == "WriteFile":
        expected = arguments.get("content")
        if not isinstance(expected, str):
            return FileEffectInspection(
                FileEffectState.UNSUPPORTED,
                "WriteFile content is not a string",
                str(path),
            )
        try:
            actual = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return FileEffectInspection(
                FileEffectState.NOT_APPLIED, "Target file does not exist", str(path)
            )
        except OSError as exc:
            return FileEffectInspection(
                FileEffectState.AMBIGUOUS, f"Cannot inspect target file: {exc}", str(path)
            )
        state = FileEffectState.APPLIED if actual == expected else FileEffectState.NOT_APPLIED
        reason = "Target content matches" if state == FileEffectState.APPLIED else "Target content differs"
        return FileEffectInspection(state, reason, str(path))

    if tool_name == "EditFile":
        old = arguments.get("old_string")
        new = arguments.get("new_string")
        if not isinstance(old, str) or not isinstance(new, str):
            return FileEffectInspection(
                FileEffectState.UNSUPPORTED,
                "EditFile strings are not inspectable",
                str(path),
            )
        try:
            actual = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return FileEffectInspection(
                FileEffectState.NOT_APPLIED, "Target file does not exist", str(path)
            )
        except OSError as exc:
            return FileEffectInspection(
                FileEffectState.AMBIGUOUS, f"Cannot inspect target file: {exc}", str(path)
            )
        old_count = actual.count(old)
        new_count = actual.count(new)
        if old_count == 0 and new_count > 0:
            return FileEffectInspection(
                FileEffectState.APPLIED,
                "Replacement is present and original text is absent",
                str(path),
            )
        if old_count == 1 and new_count == 0:
            return FileEffectInspection(
                FileEffectState.NOT_APPLIED,
                "Original text is still present",
                str(path),
            )
        return FileEffectInspection(
            FileEffectState.AMBIGUOUS,
            "File content cannot prove whether the edit completed",
            str(path),
        )

    return FileEffectInspection(
        FileEffectState.UNSUPPORTED,
        f"No file-effect inspector for {tool_name}",
        str(path),
    )
