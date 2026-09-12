from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from valecode.persistence._common import utc_now
from valecode.persistence.database import Database


@dataclass(frozen=True)
class ResultArtifactState:
    id: str
    session_id: str | None
    run_id: str | None
    step_id: str | None
    tool_call_id: str | None
    tool_use_id: str
    path: str
    sha256: str
    size_bytes: int
    state: str
    checkpoint_id: str | None
    created_at: str
    updated_at: str
    deleted_at: str | None


class ResultArtifactStore:
    """SQLite index and recoverable lifecycle for spilled tool results."""

    def __init__(self, database: Database) -> None:
        self.database = database

    @staticmethod
    def _from_row(row: sqlite3.Row) -> ResultArtifactState:
        return ResultArtifactState(**dict(row))

    @staticmethod
    def _artifact_id(path: Path) -> str:
        digest = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()
        return f"artifact_{digest}"

    def register(
        self,
        path: str | Path,
        *,
        tool_use_id: str,
        session_id: str | None = None,
        run_id: str | None = None,
        step_id: str | None = None,
        tool_call_id: str | None = None,
    ) -> ResultArtifactState:
        file_path = Path(path).resolve()
        data = file_path.read_bytes()
        now = utc_now()
        artifact_id = self._artifact_id(file_path)
        with self.database.transaction(immediate=True) as connection:
            connection.execute(
                """
                INSERT INTO result_artifacts(
                    id, session_id, run_id, step_id, tool_call_id,
                    tool_use_id, path, sha256, size_bytes, state,
                    checkpoint_id, created_at, updated_at, deleted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', NULL, ?, ?, NULL)
                ON CONFLICT(path) DO UPDATE SET
                    session_id = excluded.session_id,
                    run_id = excluded.run_id,
                    step_id = excluded.step_id,
                    tool_call_id = excluded.tool_call_id,
                    tool_use_id = excluded.tool_use_id,
                    sha256 = excluded.sha256,
                    size_bytes = excluded.size_bytes,
                    state = 'active',
                    checkpoint_id = NULL,
                    updated_at = excluded.updated_at,
                    deleted_at = NULL
                """,
                (
                    artifact_id,
                    session_id,
                    run_id,
                    step_id,
                    tool_call_id,
                    tool_use_id,
                    str(file_path),
                    hashlib.sha256(data).hexdigest(),
                    len(data),
                    now,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM result_artifacts WHERE path = ?", (str(file_path),)
            ).fetchone()
        return self._from_row(row)

    def get(self, artifact_id: str) -> ResultArtifactState | None:
        with self.database.reader() as connection:
            row = connection.execute(
                "SELECT * FROM result_artifacts WHERE id = ?", (artifact_id,)
            ).fetchone()
        return self._from_row(row) if row is not None else None

    def get_by_path(self, path: str | Path) -> ResultArtifactState | None:
        normalized = str(Path(path).resolve())
        with self.database.reader() as connection:
            row = connection.execute(
                "SELECT * FROM result_artifacts WHERE path = ?", (normalized,)
            ).fetchone()
        return self._from_row(row) if row is not None else None

    def list_for_session(self, session_id: str) -> list[ResultArtifactState]:
        with self.database.reader() as connection:
            rows = connection.execute(
                """
                SELECT * FROM result_artifacts
                WHERE session_id = ? ORDER BY created_at, id
                """,
                (session_id,),
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def reconcile_references(
        self,
        session_id: str,
        referenced_tool_use_ids: set[str],
        *,
        root_dir: str | Path,
        checkpoint_id: str | None = None,
        referenced_paths: set[str] | None = None,
    ) -> list[ResultArtifactState]:
        """Retain referenced files and safely reclaim the rest for one session.

        Rows are first marked ``released`` in SQLite, then file deletion is
        attempted, and finally each row becomes ``deleted`` or ``missing``.
        A crash between those phases is recoverable through ``sweep_released``.
        """
        now = utc_now()
        normalized_paths = {
            str(Path(path).resolve()) for path in (referenced_paths or set())
        }
        with self.database.transaction(immediate=True) as connection:
            rows = connection.execute(
                """
                SELECT * FROM result_artifacts
                WHERE session_id = ? AND state IN ('active', 'released')
                """,
                (session_id,),
            ).fetchall()
            for row in rows:
                referenced = row["path"] in normalized_paths or (
                    not normalized_paths
                    and row["tool_use_id"] in referenced_tool_use_ids
                )
                connection.execute(
                    """
                    UPDATE result_artifacts
                    SET state = ?, checkpoint_id = ?, updated_at = ?, deleted_at = NULL
                    WHERE id = ?
                    """,
                    (
                        "active" if referenced else "released",
                        checkpoint_id if referenced else None,
                        now,
                        row["id"],
                    ),
                )
        self.sweep_released(root_dir=root_dir, session_id=session_id)
        return self.list_for_session(session_id)

    def sweep_released(
        self, *, root_dir: str | Path, session_id: str | None = None
    ) -> int:
        root = Path(root_dir).resolve()
        query = "SELECT * FROM result_artifacts WHERE state = 'released'"
        params: tuple[str, ...] = ()
        if session_id is not None:
            query += " AND session_id = ?"
            params = (session_id,)
        with self.database.reader() as connection:
            rows = connection.execute(query, params).fetchall()

        updated = 0
        for row in rows:
            path = Path(row["path"]).resolve()
            # Never let a corrupt index turn cleanup into an arbitrary delete.
            if path == root or root not in path.parents:
                continue
            state = "missing"
            try:
                if path.exists():
                    path.unlink()
                    state = "deleted"
            except OSError:
                continue
            now = utc_now()
            with self.database.transaction(immediate=True) as connection:
                connection.execute(
                    """
                    UPDATE result_artifacts
                    SET state = ?, updated_at = ?, deleted_at = ?
                    WHERE id = ? AND state = 'released'
                    """,
                    (state, now, now, row["id"]),
                )
            updated += 1
        return updated
