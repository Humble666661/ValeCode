from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"


class StepStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"


class ToolCallStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    UNCERTAIN = "uncertain"
    CANCELLED = "cancelled"
    DENIED = "denied"


class TaskStatus(StrEnum):
    QUEUED = "queued"
    LEASED = "leased"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


RUN_TRANSITIONS: dict[RunStatus, frozenset[RunStatus]] = {
    RunStatus.PENDING: frozenset(
        {RunStatus.RUNNING, RunStatus.INTERRUPTED, RunStatus.CANCELLED}
    ),
    RunStatus.RUNNING: frozenset(
        {
            RunStatus.COMPLETED,
            RunStatus.FAILED,
            RunStatus.INTERRUPTED,
            RunStatus.CANCELLED,
            RunStatus.BLOCKED,
        }
    ),
    RunStatus.BLOCKED: frozenset(
        {
            RunStatus.RUNNING,
            RunStatus.FAILED,
            RunStatus.INTERRUPTED,
            RunStatus.CANCELLED,
        }
    ),
    RunStatus.INTERRUPTED: frozenset({RunStatus.RUNNING, RunStatus.FAILED}),
    RunStatus.COMPLETED: frozenset(),
    RunStatus.FAILED: frozenset(),
    RunStatus.CANCELLED: frozenset(),
}

STEP_TRANSITIONS: dict[StepStatus, frozenset[StepStatus]] = {
    StepStatus.PENDING: frozenset(
        {StepStatus.RUNNING, StepStatus.INTERRUPTED, StepStatus.CANCELLED}
    ),
    StepStatus.RUNNING: frozenset(
        {
            StepStatus.COMPLETED,
            StepStatus.FAILED,
            StepStatus.INTERRUPTED,
            StepStatus.CANCELLED,
            StepStatus.BLOCKED,
        }
    ),
    StepStatus.BLOCKED: frozenset(
        {
            StepStatus.RUNNING,
            StepStatus.FAILED,
            StepStatus.INTERRUPTED,
            StepStatus.CANCELLED,
        }
    ),
    StepStatus.INTERRUPTED: frozenset({StepStatus.RUNNING, StepStatus.FAILED}),
    StepStatus.COMPLETED: frozenset(),
    StepStatus.FAILED: frozenset(),
    StepStatus.CANCELLED: frozenset(),
}

TOOL_CALL_TRANSITIONS: dict[ToolCallStatus, frozenset[ToolCallStatus]] = {
    ToolCallStatus.PENDING: frozenset(
        {
            ToolCallStatus.RUNNING,
            ToolCallStatus.UNCERTAIN,
            ToolCallStatus.CANCELLED,
            ToolCallStatus.DENIED,
        }
    ),
    ToolCallStatus.RUNNING: frozenset(
        {
            ToolCallStatus.COMPLETED,
            ToolCallStatus.FAILED,
            ToolCallStatus.UNCERTAIN,
            ToolCallStatus.CANCELLED,
            ToolCallStatus.DENIED,
        }
    ),
    ToolCallStatus.UNCERTAIN: frozenset(
        {
            ToolCallStatus.RUNNING,
            ToolCallStatus.COMPLETED,
            ToolCallStatus.FAILED,
            ToolCallStatus.CANCELLED,
            ToolCallStatus.DENIED,
        }
    ),
    ToolCallStatus.COMPLETED: frozenset(),
    ToolCallStatus.FAILED: frozenset(),
    ToolCallStatus.CANCELLED: frozenset(),
    ToolCallStatus.DENIED: frozenset(),
}

TASK_TRANSITIONS: dict[TaskStatus, frozenset[TaskStatus]] = {
    TaskStatus.QUEUED: frozenset({TaskStatus.LEASED, TaskStatus.CANCELLED}),
    TaskStatus.LEASED: frozenset(
        {TaskStatus.RUNNING, TaskStatus.QUEUED, TaskStatus.FAILED, TaskStatus.CANCELLED}
    ),
    TaskStatus.RUNNING: frozenset(
        {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.QUEUED, TaskStatus.CANCELLED}
    ),
    TaskStatus.SUCCEEDED: frozenset(),
    TaskStatus.FAILED: frozenset({TaskStatus.QUEUED}),
    TaskStatus.CANCELLED: frozenset(),
}


class InvalidTransitionError(ValueError):
    def __init__(self, entity: str, current: str, target: str) -> None:
        super().__init__(f"Invalid {entity} transition: {current} -> {target}")


@dataclass(frozen=True)
class SessionState:
    id: str
    title: str
    summary: str
    message_count: int
    total_tokens: int
    metadata: dict[str, Any]
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class RunState:
    id: str
    session_id: str
    status: RunStatus
    input: str
    agent_id: str | None
    parent_run_id: str | None
    trace_id: str | None
    error: str | None
    metadata: dict[str, Any]
    created_at: str
    updated_at: str
    started_at: str | None
    completed_at: str | None
    version: int


@dataclass(frozen=True)
class RunTraceState:
    run_id: str
    session_id: str
    agent_id: str | None
    parent_run_id: str | None
    trace_id: str | None
    agent_type: str
    status: RunStatus
    input_tokens: int
    output_tokens: int
    tool_call_count: int
    created_at: str
    started_at: str | None
    completed_at: str | None


@dataclass(frozen=True)
class StepState:
    id: str
    run_id: str
    sequence: int
    kind: str
    status: StepStatus
    provider: str | None
    model: str | None
    input_tokens: int
    output_tokens: int
    error: str | None
    metadata: dict[str, Any]
    created_at: str
    updated_at: str
    started_at: str | None
    completed_at: str | None
    version: int


@dataclass(frozen=True)
class ToolCallState:
    id: str
    run_id: str
    step_id: str
    tool_name: str
    arguments: dict[str, Any]
    status: ToolCallStatus
    result: Any
    is_error: bool
    error: str | None
    elapsed_ms: int | None
    idempotency_key: str | None
    side_effect_class: str
    result_path: str | None
    metadata: dict[str, Any]
    created_at: str
    updated_at: str
    started_at: str | None
    completed_at: str | None
    version: int


@dataclass(frozen=True)
class TaskState:
    id: str
    session_id: str | None
    run_id: str | None
    parent_task_id: str | None
    team_name: str | None
    status: TaskStatus
    input: dict[str, Any]
    result: Any
    result_path: str | None
    error: str | None
    lease_owner: str | None
    lease_expires_at: str | None
    heartbeat_at: str | None
    attempt_count: int
    max_attempts: int
    next_retry_at: str | None
    input_tokens: int
    output_tokens: int
    metadata: dict[str, Any]
    created_at: str
    updated_at: str
    completed_at: str | None
    version: int


@dataclass(frozen=True)
class TaskAttemptState:
    id: int
    task_id: str
    attempt: int
    worker_id: str | None
    status: str
    error: str | None
    started_at: str
    completed_at: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunEvent:
    id: int
    session_id: str | None
    run_id: str | None
    step_id: str | None
    tool_call_id: str | None
    task_id: str | None
    event_type: str
    payload: dict[str, Any]
    idempotency_key: str | None
    created_at: str
