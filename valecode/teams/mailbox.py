from __future__ import annotations

import json
import os
import random
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable


class MailboxLockTimeout(TimeoutError):
    """Raised when an inbox remains owned by another process."""


class MailboxDataError(ValueError):
    """Raised when an existing inbox cannot be decoded safely."""


_SAFE_AGENT_ID = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9_-]{0,126}[A-Za-z0-9])?$")
MAILBOX_LOCK_ATTEMPTS = 50


@dataclass
class MailboxMessage:
    id: str
    from_agent: str
    to_agent: str
    content: str
    summary: str = ""
    message_type: str = "text"  # text | shutdown_request | shutdown_response
    timestamp: float = 0.0
    read: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MailboxMessage:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class Mailbox:
    """Single-file mailbox with file locking, one JSON array per agent.

    Each agent's inbox is stored as ``{agent_id}.json`` under *base_dir*.
    A companion ``.lock`` file is used for mutual exclusion (matching the
    Go/Java/TS implementation).
    """

    def __init__(self, base_dir: str | Path) -> None:
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)

    # ── path helpers ─────────────────────────────────────────────

    @staticmethod
    def _validate_agent_id(agent_id: str) -> str:
        if not isinstance(agent_id, str) or not _SAFE_AGENT_ID.fullmatch(agent_id):
            raise ValueError("Invalid mailbox agent ID")
        return agent_id

    def _inbox_path(self, agent_id: str) -> Path:
        agent_id = self._validate_agent_id(agent_id)
        return self._base_dir / f"{agent_id}.json"

    def _lock_path(self, agent_id: str) -> Path:
        agent_id = self._validate_agent_id(agent_id)
        return self._base_dir / f"{agent_id}.json.lock"

    # ── file lock ────────────────────────────────────────────────

    def _with_lock(
        self,
        agent_id: str,
        fn: Callable[[list[MailboxMessage]], Any],
        *,
        write_back: bool = True,
    ) -> Any:
        """Acquire one inbox lock and never continue after acquisition failure."""
        lock_file = self._lock_path(agent_id)
        owner_token = uuid.uuid4().hex

        # Bounded retries avoid both unprotected fallback and indefinite waits.
        last_err: Exception | None = None
        for _ in range(MAILBOX_LOCK_ATTEMPTS):
            try:
                fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
                try:
                    os.write(fd, owner_token.encode("ascii"))
                    os.fsync(fd)
                finally:
                    os.close(fd)
                break
            except FileExistsError:
                # Lock exists — check if stale (> 10s old)
                try:
                    info = lock_file.stat()
                    if time.time() - info.st_mtime > 10:
                        stale_token = lock_file.read_text(
                            encoding="ascii", errors="replace"
                        )
                        latest = lock_file.stat()
                        if (
                            latest.st_mtime_ns == info.st_mtime_ns
                            and lock_file.read_text(
                                encoding="ascii", errors="replace"
                            ) == stale_token
                        ):
                            lock_file.unlink(missing_ok=True)
                except OSError:
                    pass
                sleep_ms = 5 + random.randint(0, 95)  # 5–100ms
                time.sleep(sleep_ms / 1000)
            except OSError as e:
                last_err = e
                break
        else:
            raise MailboxLockTimeout(
                f"Timed out acquiring mailbox lock for '{agent_id}'"
            )

        if last_err is not None:
            raise last_err

        try:
            messages = self._read_inbox(agent_id)
            result = fn(messages)
            if write_back:
                if not isinstance(result, list):
                    raise TypeError("Mailbox mutation must return the message list")
                self._write_inbox(agent_id, result)
            return result
        finally:
            try:
                if lock_file.read_text(encoding="ascii") == owner_token:
                    lock_file.unlink(missing_ok=True)
            except OSError:
                pass

    # ── inbox I/O ────────────────────────────────────────────────

    def _read_inbox(self, agent_id: str) -> list[MailboxMessage]:
        path = self._inbox_path(agent_id)
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                raise TypeError("Inbox root must be a JSON array")
            if not all(isinstance(item, dict) for item in data):
                raise TypeError("Inbox messages must be JSON objects")
            return [MailboxMessage.from_dict(item) for item in data]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise MailboxDataError(f"Corrupt mailbox for '{agent_id}'") from exc

    def _write_inbox(self, agent_id: str, messages: list[MailboxMessage]) -> None:
        path = self._inbox_path(agent_id)
        data = json.dumps(
            [m.to_dict() for m in messages],
            ensure_ascii=False,
            indent=2,
        )
        temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temp_path.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
        finally:
            temp_path.unlink(missing_ok=True)

    # ── public API ───────────────────────────────────────────────

    def write(self, agent_id: str, message: MailboxMessage) -> None:
        """Append a message to *agent_id*'s inbox (thread-safe)."""
        def _append(msgs: list[MailboxMessage]) -> list[MailboxMessage]:
            message.read = False
            if message.timestamp == 0.0:
                message.timestamp = time.time()
            msgs.append(message)
            return msgs
        self._with_lock(agent_id, _append)

    def read(self, agent_id: str) -> list[MailboxMessage]:
        """Return all unread messages without marking them as read."""
        return self._with_lock(
            agent_id,
            lambda messages: [m for m in messages if not m.read],
            write_back=False,
        )

    def consume(self, agent_id: str) -> list[MailboxMessage]:
        """Return all unread messages and mark them as read (thread-safe)."""
        result: list[MailboxMessage] = []

        def _mark_read(msgs: list[MailboxMessage]) -> list[MailboxMessage]:
            for m in msgs:
                if not m.read:
                    result.append(m)
                    m.read = True
            return msgs
        self._with_lock(agent_id, _mark_read)
        return result

    def broadcast(
        self,
        team_members: list[str],
        message: MailboxMessage,
        exclude: str = "",
    ) -> None:
        for agent_id in team_members:
            if agent_id == exclude:
                continue
            self.write(agent_id, message)

    def cleanup(self, agent_id: str) -> None:
        """Remove an agent's inbox file."""
        self._inbox_path(agent_id).unlink(missing_ok=True)
        self._lock_path(agent_id).unlink(missing_ok=True)

    def cleanup_all(self) -> None:
        """Remove all inbox files."""
        if not self._base_dir.exists():
            return
        for f in self._base_dir.iterdir():
            f.unlink(missing_ok=True)


def create_message(
    from_agent: str,
    to_agent: str,
    content: str,
    summary: str = "",
    message_type: str = "text",
    metadata: dict[str, Any] | None = None,
) -> MailboxMessage:
    return MailboxMessage(
        id=uuid.uuid4().hex[:12],
        from_agent=from_agent,
        to_agent=to_agent,
        content=content,
        summary=summary,
        message_type=message_type,
        timestamp=time.time(),
        metadata=metadata or {},
    )
