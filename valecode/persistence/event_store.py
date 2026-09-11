from __future__ import annotations

import sqlite3
from typing import Any

from valecode.persistence._common import dump_json, load_json, utc_now
from valecode.persistence.database import Database
from valecode.persistence.models import RunEvent


class EventStore:
    def __init__(self, database: Database) -> None:
        self.database = database

    @staticmethod
    def _from_row(row: sqlite3.Row) -> RunEvent:
        return RunEvent(
            id=row["id"],
            session_id=row["session_id"],
            run_id=row["run_id"],
            step_id=row["step_id"],
            tool_call_id=row["tool_call_id"],
            task_id=row["task_id"],
            event_type=row["event_type"],
            payload=load_json(row["payload_json"], {}),
            idempotency_key=row["idempotency_key"],
            created_at=row["created_at"],
        )

    def _append(
        self,
        connection: sqlite3.Connection,
        event_type: str,
        *,
        payload: dict[str, Any] | None = None,
        session_id: str | None = None,
        run_id: str | None = None,
        step_id: str | None = None,
        tool_call_id: str | None = None,
        task_id: str | None = None,
        idempotency_key: str | None = None,
        created_at: str | None = None,
    ) -> RunEvent:
        encoded = dump_json(payload or {})
        timestamp = created_at or utc_now()
        try:
            cursor = connection.execute(
                """
                INSERT INTO run_events(
                    session_id, run_id, step_id, tool_call_id, task_id,
                    event_type, payload_json, idempotency_key, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    run_id,
                    step_id,
                    tool_call_id,
                    task_id,
                    event_type,
                    encoded,
                    idempotency_key,
                    timestamp,
                ),
            )
        except sqlite3.IntegrityError:
            if idempotency_key is None:
                raise
            row = connection.execute(
                "SELECT * FROM run_events WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
            if row is None:
                raise
            return self._from_row(row)
        row = connection.execute(
            "SELECT * FROM run_events WHERE id = ?", (cursor.lastrowid,)
        ).fetchone()
        return self._from_row(row)

    def append(self, event_type: str, **kwargs: Any) -> RunEvent:
        with self.database.transaction(immediate=True) as connection:
            return self._append(connection, event_type, **kwargs)

    def list(
        self,
        *,
        session_id: str | None = None,
        run_id: str | None = None,
        task_id: str | None = None,
        after_id: int | None = None,
        limit: int = 500,
    ) -> list[RunEvent]:
        if limit <= 0:
            return []
        clauses: list[str] = []
        params: list[Any] = []
        for column, value in (
            ("session_id", session_id),
            ("run_id", run_id),
            ("task_id", task_id),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        if after_id is not None:
            clauses.append("id > ?")
            params.append(after_id)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        params.append(limit)
        with self.database.reader() as connection:
            rows = connection.execute(
                f"SELECT * FROM run_events{where} ORDER BY id ASC LIMIT ?",
                params,
            ).fetchall()
        return [self._from_row(row) for row in rows]
