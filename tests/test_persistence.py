from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from valecode.persistence import (
    CheckpointStore,
    Database,
    EventStore,
    InvalidTransitionError,
    RunStatus,
    RunStore,
    ResultArtifactStore,
    SessionStore,
    StepStatus,
    TaskStatus,
    TaskStore,
    TeamStore,
    ToolCallStatus,
)
from valecode.persistence.migrations import MigrationError


@pytest.fixture
def database(tmp_path: Path) -> Database:
    database = Database(tmp_path / "control.db")
    assert database.initialize() == 5
    return database


def test_initialize_is_versioned_and_idempotent(database: Database) -> None:
    assert database.initialize() == 5
    with database.reader() as connection:
        tables = {
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
        foreign_keys = connection.execute("PRAGMA foreign_keys").fetchone()[0]

    assert {
        "schema_migrations",
        "sessions",
        "runs",
        "steps",
        "tool_calls",
        "tasks",
        "task_attempts",
        "task_dependencies",
        "run_events",
        "checkpoints",
        "result_artifacts",
        "teams",
        "team_members",
    }.issubset(tables)
    assert version == 5
    assert journal_mode == "wal"
    assert foreign_keys == 1


def test_newer_database_version_is_rejected(tmp_path: Path) -> None:
    database = Database(tmp_path / "future.db")
    assert database.initialize() == 5
    with database.transaction(immediate=True) as connection:
        connection.execute(
            "INSERT INTO schema_migrations(version, name, applied_at) VALUES (99, 'future', 'now')"
        )

    with pytest.raises(MigrationError, match="newer than supported"):
        database.initialize()


def test_checkpoint_store_indexes_transcript_boundary(database: Database) -> None:
    SessionStore(database).upsert("session-1")
    store = CheckpointStore(database)
    checkpoint = store.upsert(
        "checkpoint-1",
        "session-1",
        kind="compact",
        tail_id="tail-1",
        payload={"summary": "saved"},
        transcript_offset=42,
        run_id="run-1",
        step_id="step-1",
    )

    assert checkpoint.tail_id == "tail-1"
    assert checkpoint.payload == {"summary": "saved"}
    assert checkpoint.transcript_offset == 42
    assert store.get("checkpoint-1") == checkpoint
    assert store.list_for_session("session-1") == [checkpoint]


def test_result_artifact_store_tracks_hash_references_and_cleanup(
    database: Database, tmp_path: Path
) -> None:
    SessionStore(database).upsert("session-1")
    root = tmp_path / "tool-results"
    root.mkdir()
    keep_path = root / "keep.txt"
    drop_path = root / "drop.txt"
    keep_path.write_text("keep", encoding="utf-8")
    drop_path.write_text("drop", encoding="utf-8")
    store = ResultArtifactStore(database)
    kept = store.register(
        keep_path, tool_use_id="tool-keep", session_id="session-1"
    )
    dropped = store.register(
        drop_path, tool_use_id="tool-drop", session_id="session-1"
    )

    states = store.reconcile_references(
        "session-1",
        {"tool-keep"},
        root_dir=root,
        checkpoint_id="checkpoint-1",
    )
    by_id = {state.id: state for state in states}

    assert len(kept.sha256) == 64 and kept.size_bytes == 4
    assert by_id[kept.id].state == "active"
    assert by_id[kept.id].checkpoint_id == "checkpoint-1"
    assert keep_path.exists()
    assert by_id[dropped.id].state == "deleted"
    assert not drop_path.exists()


def test_result_artifact_cleanup_refuses_paths_outside_root(
    database: Database, tmp_path: Path
) -> None:
    SessionStore(database).upsert("session-1")
    root = tmp_path / "tool-results"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("do not delete", encoding="utf-8")
    store = ResultArtifactStore(database)
    artifact = store.register(
        outside, tool_use_id="tool-outside", session_id="session-1"
    )

    store.reconcile_references("session-1", set(), root_dir=root)

    assert outside.exists()
    assert store.get(artifact.id).state == "released"


def test_session_store_upsert_and_invalid_json_fallback(database: Database) -> None:
    store = SessionStore(database)
    created = store.upsert(
        "session-1",
        title="First",
        message_count=2,
        metadata={"source": "test"},
    )
    updated = store.upsert(
        "session-1",
        title="Updated",
        summary="summary",
        message_count=3,
        total_tokens=42,
    )

    assert updated.created_at == created.created_at
    assert updated.title == "Updated"
    assert updated.total_tokens == 42
    with database.transaction() as connection:
        connection.execute(
            "UPDATE sessions SET metadata_json = 'not-json' WHERE id = 'session-1'"
        )
    assert store.get("session-1").metadata == {}


def test_team_store_persists_member_runtime_state(database: Database) -> None:
    store = TeamStore(database)
    team = store.upsert_team(
        "team-a", "lead", description="build", backend_type="in-process"
    )
    member = type(
        "Member",
        (),
        {
            "agent_id": "agent-1",
            "name": "worker",
            "agent_type": "general",
            "model": "test",
            "worktree_path": "C:/worktree",
            "backend_type": "in-process",
            "is_active": True,
        },
    )()
    running = store.upsert_member(team.name, member)
    store.set_member_active(team.name, member.name, False)

    idle = store.list_members(team.name)[0]
    assert running.status == "running"
    assert idle.status == "idle" and idle.is_active is False
    store.mark_deleted(team.name)
    assert store.get_team(team.name).status == "deleted"
    assert store.list_members(team.name)[0].status == "stopped"


def test_run_step_and_tool_call_lifecycle_is_queryable(database: Database) -> None:
    sessions = SessionStore(database)
    events = EventStore(database)
    runs = RunStore(database, events)
    sessions.upsert("session-1")

    run = runs.create_run(
        "session-1", input="implement feature", agent_id="agent-1", trace_id="trace-1"
    )
    run = runs.transition_run(run.id, RunStatus.RUNNING)
    step = runs.create_step(run.id, provider="anthropic", model="claude-test")
    step = runs.transition_step(step.id, StepStatus.RUNNING)
    tool = runs.create_tool_call(
        run.id,
        step.id,
        "ReadFile",
        {"file_path": "README.md"},
        tool_call_id="tool-1",
        idempotency_key="run/step/tool-1",
        side_effect_class="read",
    )
    duplicate = runs.create_tool_call(
        run.id,
        step.id,
        "ReadFile",
        {"file_path": "different"},
        idempotency_key="run/step/tool-1",
    )
    assert duplicate.id == tool.id

    tool = runs.transition_tool_call(tool.id, ToolCallStatus.RUNNING)
    tool = runs.transition_tool_call(
        tool.id,
        ToolCallStatus.COMPLETED,
        result={"content": "hello"},
        elapsed_ms=17,
    )
    step = runs.transition_step(
        step.id, StepStatus.COMPLETED, input_tokens=20, output_tokens=5
    )
    run = runs.transition_run(run.id, RunStatus.COMPLETED)

    assert run.status == RunStatus.COMPLETED
    assert run.started_at is not None and run.completed_at is not None
    assert step.input_tokens == 20 and step.output_tokens == 5
    assert tool.result == {"content": "hello"}
    assert tool.elapsed_ms == 17 and tool.is_error is False
    assert runs.list_runs(session_id="session-1") == [run]
    assert runs.list_steps(run.id) == [step]
    assert runs.list_tool_calls(run.id) == [tool]
    assert runs.list_unfinished_runs() == []

    event_types = [event.event_type for event in events.list(run_id=run.id)]
    assert event_types == [
        "run.created",
        "run.status_changed",
        "step.created",
        "step.status_changed",
        "tool_call.created",
        "tool_call.status_changed",
        "tool_call.status_changed",
        "step.status_changed",
        "run.status_changed",
    ]


def test_illegal_transition_is_rejected(database: Database) -> None:
    SessionStore(database).upsert("session-1")
    runs = RunStore(database)
    run = runs.create_run("session-1")

    with pytest.raises(InvalidTransitionError, match="pending -> completed"):
        runs.transition_run(run.id, RunStatus.COMPLETED)

    assert runs.get_run(run.id).status == RunStatus.PENDING


def test_state_and_event_are_rolled_back_together(database: Database) -> None:
    SessionStore(database).upsert("session-1")
    events = EventStore(database)
    runs = RunStore(database, events)
    run = runs.create_run("session-1")
    runs.transition_run(run.id, RunStatus.RUNNING)
    before = events.list(run_id=run.id)

    with pytest.raises(TypeError):
        runs.transition_run(
            run.id,
            RunStatus.COMPLETED,
            event_payload={"not_serializable": object()},
        )

    assert runs.get_run(run.id).status == RunStatus.RUNNING
    assert events.list(run_id=run.id) == before


def test_tool_call_must_belong_to_step_run(database: Database) -> None:
    SessionStore(database).upsert("session-1")
    runs = RunStore(database)
    first = runs.create_run("session-1")
    second = runs.create_run("session-1")
    step = runs.create_step(first.id)

    with pytest.raises(sqlite3.IntegrityError):
        runs.create_tool_call(second.id, step.id, "ReadFile", {})


def test_task_and_attempt_lifecycle(database: Database) -> None:
    SessionStore(database).upsert("session-1")
    runs = RunStore(database)
    run = runs.create_run("session-1")
    tasks = TaskStore(database)
    task = tasks.create(
        {"prompt": "inspect"},
        session_id="session-1",
        run_id=run.id,
        team_name="team-a",
        max_attempts=3,
    )

    task = tasks.transition(
        task.id,
        TaskStatus.LEASED,
        lease_owner="worker-1",
        lease_expires_at="2099-01-01T00:00:00+00:00",
        increment_attempt=True,
    )
    attempt = tasks.start_attempt(task.id, task.attempt_count, worker_id="worker-1")
    task = tasks.transition(task.id, TaskStatus.RUNNING, lease_owner="worker-1")
    task = tasks.transition(
        task.id,
        TaskStatus.SUCCEEDED,
        result={"answer": "done"},
        input_tokens=10,
        output_tokens=4,
    )
    attempt = tasks.finish_attempt(task.id, attempt.attempt, status="succeeded")

    assert task.status == TaskStatus.SUCCEEDED
    assert task.result == {"answer": "done"}
    assert task.input_tokens == 10 and task.output_tokens == 4
    assert task.completed_at is not None
    assert attempt.completed_at is not None
    assert tasks.list(run_id=run.id) == [task]
    assert tasks.list_attempts(task.id) == [attempt]


def test_session_delete_cascades_control_state(database: Database) -> None:
    sessions = SessionStore(database)
    runs = RunStore(database)
    events = EventStore(database)
    sessions.upsert("session-1")
    run = runs.create_run("session-1")
    assert events.list(run_id=run.id)

    assert sessions.delete("session-1") is True
    assert runs.get_run(run.id) is None
    assert events.list(run_id=run.id) == []
