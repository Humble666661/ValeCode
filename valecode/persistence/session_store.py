from __future__ import annotations

import sqlite3
from typing import Any

from valecode.persistence._common import dump_json, load_json, utc_now
from valecode.persistence.database import Database
from valecode.persistence.models import SessionState


class SessionStore:
    def __init__(self, database: Database) -> None:
        self.database = database

    @staticmethod
    def _from_row(row: sqlite3.Row) -> SessionState:
        return SessionState(
            id=row["id"],
            title=row["title"],
            summary=row["summary"],
            message_count=row["message_count"],
            total_tokens=row["total_tokens"],
            metadata=load_json(row["metadata_json"], {}),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def upsert(
        self,
        session_id: str,
        *,
        title: str = "",
        summary: str = "",
        message_count: int = 0,
        total_tokens: int = 0,
        metadata: dict[str, Any] | None = None,
        created_at: str | None = None,
    ) -> SessionState:
        now = utc_now()
        with self.database.transaction(immediate=True) as connection:
            connection.execute(
                """
                INSERT INTO sessions(
                    id, title, summary, message_count, total_tokens,
                    metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title = excluded.title,
                    summary = excluded.summary,
                    message_count = excluded.message_count,
                    total_tokens = excluded.total_tokens,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                (
                    session_id,
                    title,
                    summary,
                    message_count,
                    total_tokens,
                    dump_json(metadata or {}),
                    created_at or now,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
        return self._from_row(row)

    def get(self, session_id: str) -> SessionState | None:
        with self.database.reader() as connection:
            row = connection.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
        return self._from_row(row) if row is not None else None

    def list(self, *, limit: int = 100) -> list[SessionState]:
        with self.database.reader() as connection:
            rows = connection.execute(
                "SELECT * FROM sessions ORDER BY updated_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def delete(self, session_id: str) -> bool:
        with self.database.transaction(immediate=True) as connection:
            cursor = connection.execute(
                "DELETE FROM sessions WHERE id = ?", (session_id,)
            )
        return cursor.rowcount > 0
