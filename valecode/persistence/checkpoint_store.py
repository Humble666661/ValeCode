from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any

from valecode.persistence._common import dump_json, load_json, utc_now
from valecode.persistence.database import Database


@dataclass(frozen=True)
class CheckpointState:
    id: str
    session_id: str
    kind: str
    tail_id: str
    payload: dict[str, Any]
    transcript_offset: int
    run_id: str | None
    step_id: str | None
    created_at: str


class CheckpointStore:
    """Queryable index aligned with authoritative JSONL checkpoints."""

    def __init__(self, database: Database) -> None:
        self.database = database

    @staticmethod
    def _from_row(row: sqlite3.Row) -> CheckpointState:
        return CheckpointState(
            id=row["id"],
            session_id=row["session_id"],
            kind=row["kind"],
            tail_id=row["tail_id"],
            payload=load_json(row["payload_json"], {}),
            transcript_offset=row["transcript_offset"],
            run_id=row["run_id"],
            step_id=row["step_id"],
            created_at=row["created_at"],
        )

    def upsert(
        self,
        checkpoint_id: str,
        session_id: str,
        *,
        kind: str,
        tail_id: str,
        payload: dict[str, Any],
        transcript_offset: int,
        run_id: str | None = None,
        step_id: str | None = None,
        created_at: str | None = None,
    ) -> CheckpointState:
        with self.database.transaction(immediate=True) as connection:
            connection.execute(
                """
                INSERT INTO checkpoints(
                    id, session_id, run_id, step_id, kind, tail_id,
                    payload_json, transcript_offset, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    session_id = excluded.session_id,
                    run_id = excluded.run_id,
                    step_id = excluded.step_id,
                    kind = excluded.kind,
                    tail_id = excluded.tail_id,
                    payload_json = excluded.payload_json,
                    transcript_offset = excluded.transcript_offset
                """,
                (
                    checkpoint_id,
                    session_id,
                    run_id,
                    step_id,
                    kind,
                    tail_id,
                    dump_json(payload),
                    transcript_offset,
                    created_at or utc_now(),
                ),
            )
            row = connection.execute(
                "SELECT * FROM checkpoints WHERE id = ?", (checkpoint_id,)
            ).fetchone()
        return self._from_row(row)

    def get(self, checkpoint_id: str) -> CheckpointState | None:
        with self.database.reader() as connection:
            row = connection.execute(
                "SELECT * FROM checkpoints WHERE id = ?", (checkpoint_id,)
            ).fetchone()
        return self._from_row(row) if row is not None else None

    def list_for_session(self, session_id: str) -> list[CheckpointState]:
        with self.database.reader() as connection:
            rows = connection.execute(
                """
                SELECT * FROM checkpoints
                WHERE session_id = ?
                ORDER BY transcript_offset, created_at
                """,
                (session_id,),
            ).fetchall()
        return [self._from_row(row) for row in rows]
