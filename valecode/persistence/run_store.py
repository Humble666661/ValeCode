from __future__ import annotations

import sqlite3
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
    RUN_TRANSITIONS,
    STEP_TRANSITIONS,
    TOOL_CALL_TRANSITIONS,
    RunState,
    RunStatus,
    RunTraceState,
    StepState,
    StepStatus,
    ToolCallState,
    ToolCallStatus,
)


_RUN_FINISHED = {
    RunStatus.COMPLETED,
    RunStatus.FAILED,
    RunStatus.INTERRUPTED,
    RunStatus.CANCELLED,
}
_STEP_FINISHED = {
    StepStatus.COMPLETED,
    StepStatus.FAILED,
    StepStatus.INTERRUPTED,
    StepStatus.CANCELLED,
}
_TOOL_FINISHED = {
    ToolCallStatus.COMPLETED,
    ToolCallStatus.FAILED,
    ToolCallStatus.UNCERTAIN,
    ToolCallStatus.CANCELLED,
    ToolCallStatus.DENIED,
}


class RunStore:
    def __init__(self, database: Database, events: EventStore | None = None) -> None:
        self.database = database
        self.events = events or EventStore(database)

    @staticmethod
    def _run_from_row(row: sqlite3.Row) -> RunState:
        return RunState(
            id=row["id"],
            session_id=row["session_id"],
            status=RunStatus(row["status"]),
            input=row["input"],
            agent_id=row["agent_id"],
            parent_run_id=row["parent_run_id"],
            trace_id=row["trace_id"],
            error=row["error"],
            metadata=load_json(row["metadata_json"], {}),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            version=row["version"],
        )

    @staticmethod
    def _step_from_row(row: sqlite3.Row) -> StepState:
        return StepState(
            id=row["id"],
            run_id=row["run_id"],
            sequence=row["sequence"],
            kind=row["kind"],
            status=StepStatus(row["status"]),
            provider=row["provider"],
            model=row["model"],
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            error=row["error"],
            metadata=load_json(row["metadata_json"], {}),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            version=row["version"],
        )

    @staticmethod
    def _tool_from_row(row: sqlite3.Row) -> ToolCallState:
        return ToolCallState(
            id=row["id"],
            run_id=row["run_id"],
            step_id=row["step_id"],
            tool_name=row["tool_name"],
            arguments=load_json(row["arguments_json"], {}),
            status=ToolCallStatus(row["status"]),
            result=load_json(row["result_json"], None),
            is_error=bool(row["is_error"]),
            error=row["error"],
            elapsed_ms=row["elapsed_ms"],
            idempotency_key=row["idempotency_key"],
            side_effect_class=row["side_effect_class"],
            result_path=row["result_path"],
            metadata=load_json(row["metadata_json"], {}),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            version=row["version"],
        )

    def create_run(
        self,
        session_id: str,
        *,
        input: str = "",
        agent_id: str | None = None,
        parent_run_id: str | None = None,
        trace_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        run_id: str | None = None,
    ) -> RunState:
        run_id = run_id or new_id("run")
        now = utc_now()
        with self.database.transaction(immediate=True) as connection:
            connection.execute(
                """
                INSERT INTO runs(
                    id, session_id, status, input, agent_id, parent_run_id,
                    trace_id, metadata_json, created_at, updated_at
                ) VALUES (?, ?, 'pending', ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    session_id,
                    input,
                    agent_id,
                    parent_run_id,
                    trace_id,
                    dump_json(metadata or {}),
                    now,
                    now,
                ),
            )
            self.events._append(
                connection,
                "run.created",
                session_id=session_id,
                run_id=run_id,
                payload={"status": RunStatus.PENDING.value},
            )
            row = connection.execute(
                "SELECT * FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
        return self._run_from_row(row)

    def get_run(self, run_id: str) -> RunState | None:
        with self.database.reader() as connection:
            row = connection.execute(
                "SELECT * FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
        return self._run_from_row(row) if row is not None else None

    def list_runs(
        self,
        *,
        session_id: str | None = None,
        statuses: set[RunStatus] | None = None,
        limit: int = 100,
    ) -> list[RunState]:
        clauses: list[str] = []
        params: list[Any] = []
        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(session_id)
        if statuses:
            placeholders = ", ".join("?" for _ in statuses)
            clauses.append(f"status IN ({placeholders})")
            params.extend(status.value for status in statuses)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        params.append(limit)
        with self.database.reader() as connection:
            rows = connection.execute(
                f"SELECT * FROM runs{where} ORDER BY created_at DESC LIMIT ?", params
            ).fetchall()
        return [self._run_from_row(row) for row in rows]

    def list_unfinished_runs(self) -> list[RunState]:
        return self.list_runs(
            statuses={RunStatus.PENDING, RunStatus.RUNNING, RunStatus.BLOCKED},
            limit=10_000,
        )

    def list_trace_nodes(
        self,
        *,
        session_id: str | None = None,
        trace_id: str | None = None,
        limit: int = 1_000,
    ) -> list[RunTraceState]:
        """Return durable run-tree nodes with aggregated execution usage."""
        if session_id is None and trace_id is None:
            raise ValueError("session_id or trace_id is required")
        limit = max(1, min(int(limit), 10_000))
        clauses: list[str] = []
        params: list[Any] = []
        if session_id is not None:
            clauses.append("r.session_id = ?")
            params.append(session_id)
        if trace_id is not None:
            clauses.append("r.trace_id = ?")
            params.append(trace_id)
        params.append(limit)
        where = " AND ".join(clauses)
        with self.database.reader() as connection:
            rows = connection.execute(
                f"""
                SELECT r.*,
                       COALESCE(s.input_tokens, 0) AS trace_input_tokens,
                       COALESCE(s.output_tokens, 0) AS trace_output_tokens,
                       COALESCE(t.tool_call_count, 0) AS trace_tool_call_count
                FROM runs AS r
                LEFT JOIN (
                    SELECT run_id,
                           SUM(input_tokens) AS input_tokens,
                           SUM(output_tokens) AS output_tokens
                    FROM steps
                    GROUP BY run_id
                ) AS s ON s.run_id = r.id
                LEFT JOIN (
                    SELECT run_id, COUNT(*) AS tool_call_count
                    FROM tool_calls
                    GROUP BY run_id
                ) AS t ON t.run_id = r.id
                WHERE {where}
                ORDER BY r.created_at ASC, r.id ASC
                LIMIT ?
                """,
                params,
            ).fetchall()

        result: list[RunTraceState] = []
        for row in rows:
            run = self._run_from_row(row)
            result.append(
                RunTraceState(
                    run_id=run.id,
                    session_id=run.session_id,
                    agent_id=run.agent_id,
                    parent_run_id=run.parent_run_id,
                    trace_id=run.trace_id,
                    agent_type=str(
                        run.metadata.get("agent_type")
                        or ("lead" if run.parent_run_id is None else "agent")
                    ),
                    status=run.status,
                    input_tokens=int(row["trace_input_tokens"]),
                    output_tokens=int(row["trace_output_tokens"]),
                    tool_call_count=int(row["trace_tool_call_count"]),
                    created_at=run.created_at,
                    started_at=run.started_at,
                    completed_at=run.completed_at,
                )
            )
        return result

    def get_run_tree(self, run_id: str) -> list[RunTraceState]:
        """Return the complete persisted tree containing ``run_id``."""
        target = self.get_run(run_id)
        if target is None:
            return []
        candidates = self.list_trace_nodes(
            session_id=target.session_id,
            trace_id=target.trace_id,
            limit=10_000,
        )
        by_id = {node.run_id: node for node in candidates}
        current = by_id.get(run_id)
        if current is None:
            return []
        seen: set[str] = set()
        while current.parent_run_id in by_id and current.run_id not in seen:
            seen.add(current.run_id)
            current = by_id[current.parent_run_id]
        root_id = current.run_id

        children: dict[str, list[RunTraceState]] = {}
        for node in candidates:
            if node.parent_run_id is not None:
                children.setdefault(node.parent_run_id, []).append(node)

        ordered: list[RunTraceState] = []
        visited: set[str] = set()

        def visit(node_id: str) -> None:
            if node_id in visited or node_id not in by_id:
                return
            visited.add(node_id)
            ordered.append(by_id[node_id])
            for child in children.get(node_id, []):
                visit(child.run_id)

        visit(root_id)
        return ordered

    def transition_run(
        self,
        run_id: str,
        status: RunStatus,
        *,
        error: str | None = None,
        event_payload: dict[str, Any] | None = None,
    ) -> RunState:
        target = RunStatus(status)
        with self.database.transaction(immediate=True) as connection:
            row = connection.execute(
                "SELECT * FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Run not found: {run_id}")
            current = RunStatus(row["status"])
            require_transition("run", current, target, RUN_TRANSITIONS)
            if current == target:
                return self._run_from_row(row)
            now = utc_now()
            started_at = row["started_at"]
            if target == RunStatus.RUNNING and started_at is None:
                started_at = now
            completed_at = now if target in _RUN_FINISHED else None
            connection.execute(
                """
                UPDATE runs SET status = ?, error = ?, updated_at = ?,
                    started_at = ?, completed_at = ?, version = version + 1
                WHERE id = ?
                """,
                (target.value, error, now, started_at, completed_at, run_id),
            )
            payload = {
                "from": current.value,
                "to": target.value,
                **(event_payload or {}),
            }
            self.events._append(
                connection,
                "run.status_changed",
                session_id=row["session_id"],
                run_id=run_id,
                payload=payload,
            )
            updated = connection.execute(
                "SELECT * FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
        return self._run_from_row(updated)

    def create_step(
        self,
        run_id: str,
        *,
        kind: str = "llm",
        provider: str | None = None,
        model: str | None = None,
        metadata: dict[str, Any] | None = None,
        step_id: str | None = None,
        sequence: int | None = None,
    ) -> StepState:
        step_id = step_id or new_id("step")
        now = utc_now()
        with self.database.transaction(immediate=True) as connection:
            run = connection.execute(
                "SELECT session_id FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
            if run is None:
                raise KeyError(f"Run not found: {run_id}")
            if sequence is None:
                seq_row = connection.execute(
                    "SELECT COALESCE(MAX(sequence), 0) + 1 AS sequence FROM steps WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
                sequence = int(seq_row["sequence"])
            connection.execute(
                """
                INSERT INTO steps(
                    id, run_id, sequence, kind, status, provider, model,
                    metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?)
                """,
                (
                    step_id,
                    run_id,
                    sequence,
                    kind,
                    provider,
                    model,
                    dump_json(metadata or {}),
                    now,
                    now,
                ),
            )
            self.events._append(
                connection,
                "step.created",
                session_id=run["session_id"],
                run_id=run_id,
                step_id=step_id,
                payload={"sequence": sequence, "kind": kind, "status": "pending"},
            )
            row = connection.execute(
                "SELECT * FROM steps WHERE id = ?", (step_id,)
            ).fetchone()
        return self._step_from_row(row)

    def get_step(self, step_id: str) -> StepState | None:
        with self.database.reader() as connection:
            row = connection.execute(
                "SELECT * FROM steps WHERE id = ?", (step_id,)
            ).fetchone()
        return self._step_from_row(row) if row is not None else None

    def list_steps(self, run_id: str) -> list[StepState]:
        with self.database.reader() as connection:
            rows = connection.execute(
                "SELECT * FROM steps WHERE run_id = ? ORDER BY sequence ASC",
                (run_id,),
            ).fetchall()
        return [self._step_from_row(row) for row in rows]

    def transition_step(
        self,
        step_id: str,
        status: StepStatus,
        *,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        error: str | None = None,
        event_payload: dict[str, Any] | None = None,
    ) -> StepState:
        target = StepStatus(status)
        with self.database.transaction(immediate=True) as connection:
            row = connection.execute(
                """
                SELECT steps.*, runs.session_id
                FROM steps JOIN runs ON runs.id = steps.run_id
                WHERE steps.id = ?
                """,
                (step_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Step not found: {step_id}")
            current = StepStatus(row["status"])
            require_transition("step", current, target, STEP_TRANSITIONS)
            if current == target:
                return self._step_from_row(row)
            now = utc_now()
            started_at = row["started_at"]
            if target == StepStatus.RUNNING and started_at is None:
                started_at = now
            completed_at = now if target in _STEP_FINISHED else None
            connection.execute(
                """
                UPDATE steps SET status = ?, input_tokens = ?, output_tokens = ?,
                    error = ?, updated_at = ?, started_at = ?, completed_at = ?,
                    version = version + 1
                WHERE id = ?
                """,
                (
                    target.value,
                    row["input_tokens"] if input_tokens is None else input_tokens,
                    row["output_tokens"] if output_tokens is None else output_tokens,
                    error,
                    now,
                    started_at,
                    completed_at,
                    step_id,
                ),
            )
            self.events._append(
                connection,
                "step.status_changed",
                session_id=row["session_id"],
                run_id=row["run_id"],
                step_id=step_id,
                payload={"from": current.value, "to": target.value, **(event_payload or {})},
            )
            updated = connection.execute(
                "SELECT * FROM steps WHERE id = ?", (step_id,)
            ).fetchone()
        return self._step_from_row(updated)

    def create_tool_call(
        self,
        run_id: str,
        step_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        tool_call_id: str | None = None,
        idempotency_key: str | None = None,
        side_effect_class: str = "unknown",
        metadata: dict[str, Any] | None = None,
    ) -> ToolCallState:
        tool_call_id = tool_call_id or new_id("tool")
        now = utc_now()
        with self.database.transaction(immediate=True) as connection:
            if idempotency_key is not None:
                existing = connection.execute(
                    "SELECT * FROM tool_calls WHERE idempotency_key = ?",
                    (idempotency_key,),
                ).fetchone()
                if existing is not None:
                    return self._tool_from_row(existing)
            run = connection.execute(
                "SELECT session_id FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
            if run is None:
                raise KeyError(f"Run not found: {run_id}")
            connection.execute(
                """
                INSERT INTO tool_calls(
                    id, run_id, step_id, tool_name, arguments_json, status,
                    idempotency_key, side_effect_class, metadata_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?)
                """,
                (
                    tool_call_id,
                    run_id,
                    step_id,
                    tool_name,
                    dump_json(arguments),
                    idempotency_key,
                    side_effect_class,
                    dump_json(metadata or {}),
                    now,
                    now,
                ),
            )
            self.events._append(
                connection,
                "tool_call.created",
                session_id=run["session_id"],
                run_id=run_id,
                step_id=step_id,
                tool_call_id=tool_call_id,
                payload={"tool_name": tool_name, "status": "pending"},
            )
            row = connection.execute(
                "SELECT * FROM tool_calls WHERE id = ?", (tool_call_id,)
            ).fetchone()
        return self._tool_from_row(row)

    def get_tool_call(self, tool_call_id: str) -> ToolCallState | None:
        with self.database.reader() as connection:
            row = connection.execute(
                "SELECT * FROM tool_calls WHERE id = ?", (tool_call_id,)
            ).fetchone()
        return self._tool_from_row(row) if row is not None else None

    def get_tool_call_by_idempotency_key(self, key: str) -> ToolCallState | None:
        with self.database.reader() as connection:
            row = connection.execute(
                "SELECT * FROM tool_calls WHERE idempotency_key = ?", (key,)
            ).fetchone()
        return self._tool_from_row(row) if row is not None else None

    def list_tool_calls(self, run_id: str) -> list[ToolCallState]:
        with self.database.reader() as connection:
            rows = connection.execute(
                "SELECT * FROM tool_calls WHERE run_id = ? ORDER BY created_at ASC",
                (run_id,),
            ).fetchall()
        return [self._tool_from_row(row) for row in rows]

    def transition_tool_call(
        self,
        tool_call_id: str,
        status: ToolCallStatus,
        *,
        result: Any = None,
        is_error: bool | None = None,
        error: str | None = None,
        elapsed_ms: int | None = None,
        result_path: str | None = None,
        event_payload: dict[str, Any] | None = None,
    ) -> ToolCallState:
        target = ToolCallStatus(status)
        with self.database.transaction(immediate=True) as connection:
            row = connection.execute(
                """
                SELECT tool_calls.*, runs.session_id
                FROM tool_calls JOIN runs ON runs.id = tool_calls.run_id
                WHERE tool_calls.id = ?
                """,
                (tool_call_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Tool call not found: {tool_call_id}")
            current = ToolCallStatus(row["status"])
            require_transition("tool call", current, target, TOOL_CALL_TRANSITIONS)
            if current == target:
                return self._tool_from_row(row)
            now = utc_now()
            started_at = row["started_at"]
            if target == ToolCallStatus.RUNNING and started_at is None:
                started_at = now
            completed_at = now if target in _TOOL_FINISHED else None
            effective_is_error = (
                target != ToolCallStatus.COMPLETED if is_error is None else is_error
            )
            connection.execute(
                """
                UPDATE tool_calls SET status = ?, result_json = ?, is_error = ?,
                    error = ?, elapsed_ms = ?, result_path = ?, updated_at = ?,
                    started_at = ?, completed_at = ?, version = version + 1
                WHERE id = ?
                """,
                (
                    target.value,
                    dump_json(result),
                    int(effective_is_error),
                    error,
                    elapsed_ms,
                    result_path,
                    now,
                    started_at,
                    completed_at,
                    tool_call_id,
                ),
            )
            self.events._append(
                connection,
                "tool_call.status_changed",
                session_id=row["session_id"],
                run_id=row["run_id"],
                step_id=row["step_id"],
                tool_call_id=tool_call_id,
                payload={"from": current.value, "to": target.value, **(event_payload or {})},
            )
            updated = connection.execute(
                "SELECT * FROM tool_calls WHERE id = ?", (tool_call_id,)
            ).fetchone()
        return self._tool_from_row(updated)
