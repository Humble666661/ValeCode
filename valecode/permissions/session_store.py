"""Durable, exact-match approvals scoped to one conversation session."""

from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path


_SESSION_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
_FINGERPRINT = re.compile(r"[0-9a-f]{64}\Z")


class SessionAllowStore:
    def __init__(self, work_dir: str | Path, session_id: str) -> None:
        if not _SESSION_ID.fullmatch(session_id):
            raise ValueError("Invalid session ID for permission grants")
        root = Path(work_dir).resolve()
        state_dir = (root / ".valecode" / "session-permissions").resolve()
        if not state_dir.is_relative_to(root):
            raise ValueError("Session permission directory escapes the project")
        self.path = state_dir / f"{session_id}.json"

    def load(self) -> set[str]:
        if self.path.is_symlink():
            raise ValueError("Session permission file must not be a symlink")
        if not self.path.exists():
            return set()
        if self.path.stat().st_size > 100_000:
            raise ValueError("Session permission state is too large")
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        grants = raw.get("grants") if isinstance(raw, dict) and raw.get("version") == 1 else None
        if not isinstance(grants, list) or len(grants) > 500 or any(
            not isinstance(grant, str) or not _FINGERPRINT.fullmatch(grant)
            for grant in grants
        ):
            raise ValueError("Invalid session permission state")
        return set(grants)

    def save(self, grants: set[str]) -> None:
        if self.path.is_symlink():
            raise ValueError("Session permission file must not be a symlink")
        if len(grants) > 500 or any(not _FINGERPRINT.fullmatch(grant) for grant in grants):
            raise ValueError("Invalid session permission grants")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(
                json.dumps({"version": 1, "grants": sorted(grants)}),
                encoding="utf-8",
            )
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def delete(self) -> bool:
        if not self.path.exists() and not self.path.is_symlink():
            return False
        self.path.unlink()
        return True
