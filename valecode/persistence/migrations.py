from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass

from valecode.persistence.database import Database


class MigrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    apply: Callable[[sqlite3.Connection], None]


def _migration_001_initial_control_plane(connection: sqlite3.Connection) -> None:
    statements = [
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL DEFAULT '',
            summary TEXT NOT NULL DEFAULT '',
            message_count INTEGER NOT NULL DEFAULT 0 CHECK (message_count >= 0),
            total_tokens INTEGER NOT NULL DEFAULT 0 CHECK (total_tokens >= 0),
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE runs (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            status TEXT NOT NULL CHECK (
                status IN ('pending', 'running', 'completed', 'failed', 'interrupted', 'cancelled', 'blocked')
            ),
            input TEXT NOT NULL DEFAULT '',
            agent_id TEXT,
            parent_run_id TEXT REFERENCES runs(id) ON DELETE SET NULL,
            trace_id TEXT,
            error TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            started_at TEXT,
            completed_at TEXT,
            version INTEGER NOT NULL DEFAULT 0
        )
        """,
        """
        CREATE TABLE steps (
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
            sequence INTEGER NOT NULL CHECK (sequence > 0),
            kind TEXT NOT NULL DEFAULT 'llm',
            status TEXT NOT NULL CHECK (
                status IN ('pending', 'running', 'completed', 'failed', 'interrupted', 'cancelled', 'blocked')
            ),
            provider TEXT,
            model TEXT,
            input_tokens INTEGER NOT NULL DEFAULT 0 CHECK (input_tokens >= 0),
            output_tokens INTEGER NOT NULL DEFAULT 0 CHECK (output_tokens >= 0),
            error TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            started_at TEXT,
            completed_at TEXT,
            version INTEGER NOT NULL DEFAULT 0,
            UNIQUE (run_id, sequence),
            UNIQUE (id, run_id)
        )
        """,
        """
        CREATE TABLE tool_calls (
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            step_id TEXT NOT NULL,
            tool_name TEXT NOT NULL,
            arguments_json TEXT NOT NULL DEFAULT '{}',
            status TEXT NOT NULL CHECK (
                status IN ('pending', 'running', 'completed', 'failed', 'uncertain', 'cancelled', 'denied')
            ),
            result_json TEXT,
            is_error INTEGER NOT NULL DEFAULT 0 CHECK (is_error IN (0, 1)),
            error TEXT,
            elapsed_ms INTEGER CHECK (elapsed_ms IS NULL OR elapsed_ms >= 0),
            idempotency_key TEXT,
            side_effect_class TEXT NOT NULL DEFAULT 'unknown' CHECK (
                side_effect_class IN ('read', 'write', 'external', 'unknown')
            ),
            result_path TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            started_at TEXT,
            completed_at TEXT,
            version INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY (step_id, run_id) REFERENCES steps(id, run_id) ON DELETE CASCADE
        )
        """,
        """
        CREATE TABLE tasks (
            id TEXT PRIMARY KEY,
            session_id TEXT REFERENCES sessions(id) ON DELETE SET NULL,
            run_id TEXT REFERENCES runs(id) ON DELETE SET NULL,
            parent_task_id TEXT REFERENCES tasks(id) ON DELETE SET NULL,
            team_name TEXT,
            status TEXT NOT NULL CHECK (
                status IN ('queued', 'leased', 'running', 'succeeded', 'failed', 'cancelled')
            ),
            input_json TEXT NOT NULL DEFAULT '{}',
            result_json TEXT,
            result_path TEXT,
            error TEXT,
            lease_owner TEXT,
            lease_expires_at TEXT,
            heartbeat_at TEXT,
            attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
            max_attempts INTEGER NOT NULL DEFAULT 1 CHECK (max_attempts > 0),
            next_retry_at TEXT,
            input_tokens INTEGER NOT NULL DEFAULT 0 CHECK (input_tokens >= 0),
            output_tokens INTEGER NOT NULL DEFAULT 0 CHECK (output_tokens >= 0),
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            completed_at TEXT,
            version INTEGER NOT NULL DEFAULT 0
        )
        """,
        """
        CREATE TABLE task_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
            attempt INTEGER NOT NULL CHECK (attempt > 0),
            worker_id TEXT,
            status TEXT NOT NULL,
            error TEXT,
            started_at TEXT NOT NULL,
            completed_at TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            UNIQUE (task_id, attempt)
        )
        """,
        """
        CREATE TABLE run_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT REFERENCES sessions(id) ON DELETE CASCADE,
            run_id TEXT REFERENCES runs(id) ON DELETE CASCADE,
            step_id TEXT REFERENCES steps(id) ON DELETE CASCADE,
            tool_call_id TEXT REFERENCES tool_calls(id) ON DELETE CASCADE,
            task_id TEXT REFERENCES tasks(id) ON DELETE CASCADE,
            event_type TEXT NOT NULL,
            payload_json TEXT NOT NULL DEFAULT '{}',
            idempotency_key TEXT,
            created_at TEXT NOT NULL
        )
        """,
        "CREATE INDEX idx_runs_session_created ON runs(session_id, created_at)",
        "CREATE INDEX idx_runs_status_updated ON runs(status, updated_at)",
        "CREATE INDEX idx_steps_run_sequence ON steps(run_id, sequence)",
        "CREATE INDEX idx_tool_calls_run_status ON tool_calls(run_id, status)",
        "CREATE INDEX idx_tool_calls_step ON tool_calls(step_id)",
        "CREATE UNIQUE INDEX idx_tool_calls_idempotency ON tool_calls(idempotency_key) WHERE idempotency_key IS NOT NULL",
        "CREATE INDEX idx_tasks_status_retry ON tasks(status, next_retry_at)",
        "CREATE INDEX idx_tasks_lease_expiry ON tasks(status, lease_expires_at)",
        "CREATE INDEX idx_tasks_run ON tasks(run_id)",
        "CREATE INDEX idx_task_attempts_task ON task_attempts(task_id, attempt)",
        "CREATE INDEX idx_run_events_run_id ON run_events(run_id, id)",
        "CREATE INDEX idx_run_events_session_id ON run_events(session_id, id)",
        "CREATE INDEX idx_run_events_task_id ON run_events(task_id, id)",
        "CREATE UNIQUE INDEX idx_run_events_idempotency ON run_events(idempotency_key) WHERE idempotency_key IS NOT NULL",
    ]
    for statement in statements:
        connection.execute(statement)


def _migration_002_task_dependencies(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE task_dependencies (
            task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
            depends_on_task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
            created_at TEXT NOT NULL,
            PRIMARY KEY (task_id, depends_on_task_id),
            CHECK (task_id != depends_on_task_id)
        )
        """
    )
    connection.execute(
        "CREATE INDEX idx_task_dependencies_dependency ON task_dependencies(depends_on_task_id)"
    )


MIGRATIONS = (
    Migration(1, "initial_control_plane", _migration_001_initial_control_plane),
    Migration(2, "task_dependencies", _migration_002_task_dependencies),
)


def apply_migrations(database: Database) -> int:
    latest = MIGRATIONS[-1].version if MIGRATIONS else 0
    with database.transaction(immediate=True) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at TEXT NOT NULL
            )
            """
        )
        row = connection.execute(
            "SELECT COALESCE(MAX(version), 0) AS version FROM schema_migrations"
        ).fetchone()
        current = int(row["version"])
        if current > latest:
            raise MigrationError(
                f"Database schema version {current} is newer than supported version {latest}"
            )
        for migration in MIGRATIONS:
            if migration.version <= current:
                continue
            migration.apply(connection)
            connection.execute(
                """
                INSERT INTO schema_migrations(version, name, applied_at)
                VALUES (?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                """,
                (migration.version, migration.name),
            )
            connection.execute(f"PRAGMA user_version = {migration.version}")
            current = migration.version
    return current
