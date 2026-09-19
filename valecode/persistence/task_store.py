from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from typing import Any

from valecode.persistence._common import (
    dump_json,
    load_json,
    new_id,
    require_transition,
    utc_now,
)
from valecode.persistence.database import Database
from valecode.persistence.event_store import EventStore
from valecode.persistence.models import (
    TASK_TRANSITIONS,
    TaskAttemptState,
    TaskState,
    TaskStatus,
)


_TASK_FINISHED = {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED}
_UNSET = object()


class TaskStore:
    def __init__(self, database: Database, events: EventStore | None = None) -> None:
        self.database = database
        self.events = events or EventStore(database)

    @staticmethod
    def _from_row(row: sqlite3.Row) -> TaskState:
        return TaskState(
            id=row["id"],
            session_id=row["session_id"],
            run_id=row["run_id"],
            parent_task_id=row["parent_task_id"],
            team_name=row["team_name"],
            status=TaskStatus(row["status"]),
            input=load_json(row["input_json"], {}),
            result=load_json(row["result_json"], None),
            result_path=row["result_path"],
            error=row["error"],
            lease_owner=row["lease_owner"],
            lease_expires_at=row["lease_expires_at"],
            heartbeat_at=row["heartbeat_at"],
            attempt_count=row["attempt_count"],
            max_attempts=row["max_attempts"],
            next_retry_at=row["next_retry_at"],
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            metadata=load_json(row["metadata_json"], {}),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            completed_at=row["completed_at"],
            version=row["version"],
        )

    @staticmethod
    def _attempt_from_row(row: sqlite3.Row) -> TaskAttemptState:
        return TaskAttemptState(
            id=row["id"],
            task_id=row["task_id"],
            attempt=row["attempt"],
            worker_id=row["worker_id"],
            status=row["status"],
            error=row["error"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            metadata=load_json(row["metadata_json"], {}),
        )

    def create(
        self,
        input: dict[str, Any],
        *,
        task_id: str | None = None,
        session_id: str | None = None,
        run_id: str | None = None,
        parent_task_id: str | None = None,
        team_name: str | None = None,
        max_attempts: int = 1,
        dependencies: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TaskState:
        task_id = task_id or new_id("task")
        now = utc_now()
        with self.database.transaction(immediate=True) as connection:
            connection.execute(
                """
                INSERT INTO tasks(
                    id, session_id, run_id, parent_task_id, team_name, status,
                    input_json, max_attempts, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'queued', ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    session_id,
                    run_id,
                    parent_task_id,
                    team_name,
                    dump_json(input),
                    max_attempts,
                    dump_json(metadata or {}),
                    now,
                    now,
                ),
            )
            for dependency_id in dependencies or []:
                connection.execute(
                    """
                    INSERT INTO task_dependencies(task_id, depends_on_task_id, created_at)
                    VALUES (?, ?, ?)
                    """,
                    (task_id, dependency_id, now),
                )
            self.events._append(
                connection,
                "task.created",
                session_id=session_id,
                run_id=run_id,
                task_id=task_id,
                payload={"status": TaskStatus.QUEUED.value},
            )
            row = connection.execute(
                "SELECT * FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
        return self._from_row(row)

    def get(self, task_id: str) -> TaskState | None:
        with self.database.reader() as connection:
            row = connection.execute(
                "SELECT * FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
        return self._from_row(row) if row is not None else None

    def list(
        self,
        *,
        statuses: set[TaskStatus] | None = None,
        run_id: str | None = None,
        session_id: str | None = None,
        team_name: str | None = None,
        limit: int = 500,
    ) -> list[TaskState]:
        clauses: list[str] = []
        params: list[Any] = []
        if statuses:
            placeholders = ", ".join("?" for _ in statuses)
            clauses.append(f"status IN ({placeholders})")
            params.extend(status.value for status in statuses)
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)
        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(session_id)
        if team_name is not None:
            clauses.append("team_name = ?")
            params.append(team_name)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        params.append(limit)
        with self.database.reader() as connection:
            rows = connection.execute(
                f"SELECT * FROM tasks{where} ORDER BY created_at ASC LIMIT ?", params
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def transition(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        result: Any = _UNSET,
        result_path: str | None | object = _UNSET,
        error: str | None | object = _UNSET,
        lease_owner: str | None | object = _UNSET,
        lease_expires_at: str | None | object = _UNSET,
        heartbeat_at: str | None | object = _UNSET,
        next_retry_at: str | None | object = _UNSET,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        increment_attempt: bool = False,
        event_payload: dict[str, Any] | None = None,
    ) -> TaskState:
        target = TaskStatus(status)
        with self.database.transaction(immediate=True) as connection:
            row = connection.execute(
                "SELECT * FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Task not found: {task_id}")
            current = TaskStatus(row["status"])
            require_transition("task", current, target, TASK_TRANSITIONS)
            if current == target:
                return self._from_row(row)
            now = utc_now()
            completed_at = now if target in _TASK_FINISHED else None
            attempt_count = row["attempt_count"] + int(increment_attempt)
            connection.execute(
                """
                UPDATE tasks SET status = ?, result_json = ?, result_path = ?,
                    error = ?, lease_owner = ?, lease_expires_at = ?, heartbeat_at = ?,
                    attempt_count = ?, next_retry_at = ?, input_tokens = ?,
                    output_tokens = ?, updated_at = ?, completed_at = ?,
                    version = version + 1
                WHERE id = ?
                """,
                (
                    target.value,
                    row["result_json"] if result is _UNSET else dump_json(result),
                    row["result_path"] if result_path is _UNSET else result_path,
                    row["error"] if error is _UNSET else error,
                    row["lease_owner"] if lease_owner is _UNSET else lease_owner,
                    row["lease_expires_at"] if lease_expires_at is _UNSET else lease_expires_at,
                    row["heartbeat_at"] if heartbeat_at is _UNSET else heartbeat_at,
                    attempt_count,
                    row["next_retry_at"] if next_retry_at is _UNSET else next_retry_at,
                    row["input_tokens"] if input_tokens is None else input_tokens,
                    row["output_tokens"] if output_tokens is None else output_tokens,
                    now,
                    completed_at,
                    task_id,
                ),
            )
            self.events._append(
                connection,
                "task.status_changed",
                session_id=row["session_id"],
                run_id=row["run_id"],
                task_id=task_id,
                payload={"from": current.value, "to": target.value, **(event_payload or {})},
            )
            updated = connection.execute(
                "SELECT * FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
        return self._from_row(updated)

    def dependencies(self, task_id: str) -> list[str]:
        with self.database.reader() as connection:
            rows = connection.execute(
                """
                SELECT depends_on_task_id FROM task_dependencies
                WHERE task_id = ? ORDER BY created_at ASC
                """,
                (task_id,),
            ).fetchall()
        return [row["depends_on_task_id"] for row in rows]

    def update_details(
        self,
        task_id: str,
        *,
        input: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        add_dependencies: list[str] | None = None,
    ) -> TaskState:
        with self.database.transaction(immediate=True) as connection:
            row = connection.execute(
                "SELECT * FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Task not found: {task_id}")
            now = utc_now()
            connection.execute(
                """
                UPDATE tasks SET input_json = ?, metadata_json = ?,
                    updated_at = ?, version = version + 1 WHERE id = ?
                """,
                (
                    dump_json(input) if input is not None else row["input_json"],
                    dump_json(metadata) if metadata is not None else row["metadata_json"],
                    now,
                    task_id,
                ),
            )
            for dependency_id in add_dependencies or []:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO task_dependencies(
                        task_id, depends_on_task_id, created_at
                    ) VALUES (?, ?, ?)
                    """,
                    (task_id, dependency_id, now),
                )
            self.events._append(
                connection,
                "task.details_updated",
                session_id=row["session_id"],
                run_id=row["run_id"],
                task_id=task_id,
                payload={"dependencies_added": add_dependencies or []},
            )
            updated = connection.execute(
                "SELECT * FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
        return self._from_row(updated)

    def update_board_status(self, task_id: str, board_status: str) -> TaskState:
        status_map = {
            "pending": TaskStatus.QUEUED,
            "blocked": TaskStatus.QUEUED,
            "in_progress": TaskStatus.RUNNING,
            "completed": TaskStatus.SUCCEEDED,
        }
        if board_status not in status_map:
            raise ValueError(f"Unsupported board status: {board_status}")
        with self.database.transaction(immediate=True) as connection:
            row = connection.execute(
                "SELECT * FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Task not found: {task_id}")
            metadata = load_json(row["metadata_json"], {})
            if metadata.get("kind") != "shared_team_task":
                raise ValueError("Board status is only valid for shared team tasks")
            data = load_json(row["input_json"], {})
            previous_board_status = data.get("board_status", "pending")
            data["board_status"] = board_status
            target = status_map[board_status]
            now = utc_now()
            completed_at = now if target == TaskStatus.SUCCEEDED else None
            connection.execute(
                """
                UPDATE tasks SET status = ?, input_json = ?, updated_at = ?,
                    completed_at = ?, version = version + 1 WHERE id = ?
                """,
                (target.value, dump_json(data), now, completed_at, task_id),
            )
            self.events._append(
                connection,
                "task.board_status_changed",
                task_id=task_id,
                payload={"from": previous_board_status, "to": board_status},
            )
            updated = connection.execute(
                "SELECT * FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
        return self._from_row(updated)

    def dependencies_ready(self, task_id: str) -> bool:
        with self.database.reader() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS pending
                FROM task_dependencies d
                JOIN tasks dependency ON dependency.id = d.depends_on_task_id
                WHERE d.task_id = ? AND dependency.status != 'succeeded'
                """,
                (task_id,),
            ).fetchone()
        return int(row["pending"]) == 0

    def claim(
        self,
        task_id: str,
        worker_id: str,
        *,
        lease_seconds: float = 30.0,
    ) -> TaskState | None:
        now_dt = datetime.now(UTC)
        now = now_dt.isoformat(timespec="milliseconds")
        expires = (now_dt + timedelta(seconds=lease_seconds)).isoformat(
            timespec="milliseconds"
        )
        with self.database.transaction(immediate=True) as connection:
            row = connection.execute(
                "SELECT * FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Task not found: {task_id}")
            if TaskStatus(row["status"]) != TaskStatus.QUEUED:
                return None
            if row["next_retry_at"] is not None and row["next_retry_at"] > now:
                return None
            if row["attempt_count"] >= row["max_attempts"]:
                return None
            blocked = connection.execute(
                """
                SELECT 1
                FROM task_dependencies d
                JOIN tasks dependency ON dependency.id = d.depends_on_task_id
                WHERE d.task_id = ? AND dependency.status != 'succeeded'
                LIMIT 1
                """,
                (task_id,),
            ).fetchone()
            if blocked is not None:
                return None
            attempt = row["attempt_count"] + 1
            connection.execute(
                """
                UPDATE tasks SET status = 'leased', lease_owner = ?,
                    lease_expires_at = ?, heartbeat_at = ?, attempt_count = ?,
                    next_retry_at = NULL, updated_at = ?, version = version + 1
                WHERE id = ?
                """,
                (worker_id, expires, now, attempt, now, task_id),
            )
            connection.execute(
                """
                INSERT INTO task_attempts(
                    task_id, attempt, worker_id, status, started_at, metadata_json
                ) VALUES (?, ?, ?, 'leased', ?, '{}')
                """,
                (task_id, attempt, worker_id, now),
            )
            self.events._append(
                connection,
                "task.leased",
                session_id=row["session_id"],
                run_id=row["run_id"],
                task_id=task_id,
                payload={
                    "worker_id": worker_id,
                    "attempt": attempt,
                    "lease_expires_at": expires,
                },
            )
            updated = connection.execute(
                "SELECT * FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
        return self._from_row(updated)

    def mark_running(self, task_id: str, worker_id: str) -> TaskState:
        with self.database.transaction(immediate=True) as connection:
            row = connection.execute(
                "SELECT * FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Task not found: {task_id}")
            if row["status"] != TaskStatus.LEASED.value or row["lease_owner"] != worker_id:
                raise ValueError(f"Task {task_id} is not leased by {worker_id}")
            now = utc_now()
            connection.execute(
                "UPDATE tasks SET status = 'running', updated_at = ?, version = version + 1 WHERE id = ?",
                (now, task_id),
            )
            connection.execute(
                "UPDATE task_attempts SET status = 'running' WHERE task_id = ? AND attempt = ?",
                (task_id, row["attempt_count"]),
            )
            self.events._append(
                connection,
                "task.status_changed",
                session_id=row["session_id"],
                run_id=row["run_id"],
                task_id=task_id,
                payload={"from": "leased", "to": "running", "worker_id": worker_id},
            )
            updated = connection.execute(
                "SELECT * FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
        return self._from_row(updated)

    def heartbeat(
        self, task_id: str, worker_id: str, *, lease_seconds: float = 30.0
    ) -> bool:
        now_dt = datetime.now(UTC)
        now = now_dt.isoformat(timespec="milliseconds")
        expires = (now_dt + timedelta(seconds=lease_seconds)).isoformat(
            timespec="milliseconds"
        )
        with self.database.transaction(immediate=True) as connection:
            cursor = connection.execute(
                """
                UPDATE tasks SET heartbeat_at = ?, lease_expires_at = ?,
                    updated_at = ?, version = version + 1
                WHERE id = ? AND lease_owner = ? AND status IN ('leased', 'running')
                """,
                (now, expires, now, task_id, worker_id),
            )
        return cursor.rowcount == 1

    def recover_expired_leases(self, *, now: str | None = None) -> list[TaskState]:
        current = now or utc_now()
        recovered: list[TaskState] = []
        with self.database.transaction(immediate=True) as connection:
            rows = connection.execute(
                """
                SELECT * FROM tasks
                WHERE status IN ('leased', 'running')
                  AND lease_expires_at IS NOT NULL AND lease_expires_at <= ?
                ORDER BY created_at ASC
                """,
                (current,),
            ).fetchall()
            for row in rows:
                can_retry = row["attempt_count"] < row["max_attempts"]
                target = TaskStatus.QUEUED if can_retry else TaskStatus.FAILED
                completed_at = None if can_retry else current
                error = "Worker lease expired"
                connection.execute(
                    """
                    UPDATE tasks SET status = ?, error = ?, lease_owner = NULL,
                        lease_expires_at = NULL, heartbeat_at = NULL,
                        updated_at = ?, completed_at = ?, version = version + 1
                    WHERE id = ?
                    """,
                    (target.value, error, current, completed_at, row["id"]),
                )
                connection.execute(
                    """
                    UPDATE task_attempts SET status = 'expired', error = ?, completed_at = ?
                    WHERE task_id = ? AND attempt = ? AND completed_at IS NULL
                    """,
                    (error, current, row["id"], row["attempt_count"]),
                )
                self.events._append(
                    connection,
                    "task.lease_expired",
                    session_id=row["session_id"],
                    run_id=row["run_id"],
                    task_id=row["id"],
                    payload={"attempt": row["attempt_count"], "to": target.value},
                )
                updated = connection.execute(
                    "SELECT * FROM tasks WHERE id = ?", (row["id"],)
                ).fetchone()
                recovered.append(self._from_row(updated))
        return recovered

    def start_attempt(
        self,
        task_id: str,
        attempt: int,
        *,
        worker_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TaskAttemptState:
        now = utc_now()
        with self.database.transaction(immediate=True) as connection:
            cursor = connection.execute(
                """
                INSERT INTO task_attempts(
                    task_id, attempt, worker_id, status, started_at, metadata_json
                ) VALUES (?, ?, ?, 'running', ?, ?)
                """,
                (task_id, attempt, worker_id, now, dump_json(metadata or {})),
            )
            row = connection.execute(
                "SELECT * FROM task_attempts WHERE id = ?", (cursor.lastrowid,)
            ).fetchone()
        return self._attempt_from_row(row)

    def finish_attempt(
        self,
        task_id: str,
        attempt: int,
        *,
        status: str,
        error: str | None = None,
    ) -> TaskAttemptState:
        with self.database.transaction(immediate=True) as connection:
            cursor = connection.execute(
                """
                UPDATE task_attempts SET status = ?, error = ?, completed_at = ?
                WHERE task_id = ? AND attempt = ?
                """,
                (status, error, utc_now(), task_id, attempt),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"Task attempt not found: {task_id}/{attempt}")
            row = connection.execute(
                "SELECT * FROM task_attempts WHERE task_id = ? AND attempt = ?",
                (task_id, attempt),
            ).fetchone()
        return self._attempt_from_row(row)

    def list_attempts(self, task_id: str) -> list[TaskAttemptState]:
        with self.database.reader() as connection:
            rows = connection.execute(
                "SELECT * FROM task_attempts WHERE task_id = ? ORDER BY attempt ASC",
                (task_id,),
            ).fetchall()
        return [self._attempt_from_row(row) for row in rows]
