from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from valecode.agents.durable_task_manager import RESULT_INLINE_LIMIT, DurableTaskManager
from valecode.memory.session import SessionManager
from valecode.persistence import TaskStatus
from valecode.teams.shared_task import DurableSharedTaskStore


def make_agent(session_id: str, result: str = "done"):
    agent = MagicMock()
    agent.session_id = session_id
    agent._current_run_id = None
    agent.parent_run_id = None
    agent.team_name = ""
    agent._team_manager = None
    agent.total_input_tokens = 12
    agent.total_output_tokens = 7
    agent.run_to_completion = AsyncMock(return_value=result)
    return agent


def test_durable_manager_uses_shared_background_task_config(tmp_path):
    sessions = SessionManager(str(tmp_path))
    config = SimpleNamespace(
        lease_seconds=42.0,
        heartbeat_interval=9.0,
        maintenance_interval=4.0,
        max_concurrency=6,
        per_team_concurrency=2,
        retry_base_seconds=2.0,
        retry_max_seconds=12.0,
        result_retention_days=14.0,
        result_gc_interval=120.0,
    )

    manager = DurableTaskManager.from_config(sessions.task_store, config)

    assert manager.lease_seconds == 42.0
    assert manager.heartbeat_interval == 9.0
    assert manager.maintenance_interval == 4.0
    assert manager._global_capacity._value == 6
    assert manager._team_limit == 2
    assert manager.retry_base_seconds == 2.0
    assert manager.retry_max_seconds == 12.0
    assert manager.result_retention_days == 14.0
    assert manager.result_gc_interval == 120.0


def test_task_store_claim_enforces_dependencies_and_leases(tmp_path):
    sessions = SessionManager(str(tmp_path))
    session = sessions.create()
    store = sessions.task_store
    dependency = store.create({"task": "first"}, session_id=session.session_id)
    child = store.create(
        {"task": "second"},
        session_id=session.session_id,
        dependencies=[dependency.id],
    )

    assert store.dependencies(child.id) == [dependency.id]
    assert store.dependencies_ready(child.id) is False
    assert store.claim(child.id, "worker") is None

    leased = store.claim(dependency.id, "worker", lease_seconds=10)
    assert leased is not None
    before = leased.lease_expires_at
    running = store.mark_running(dependency.id, "worker")
    assert running.status == TaskStatus.RUNNING
    assert store.heartbeat(dependency.id, "worker", lease_seconds=20)
    assert store.get(dependency.id).lease_expires_at > before
    store.finish_attempt(dependency.id, 1, status="succeeded")
    store.transition(dependency.id, TaskStatus.SUCCEEDED)

    assert store.dependencies_ready(child.id) is True
    assert store.claim(child.id, "worker") is not None
    session.close()


def test_task_store_list_can_filter_by_session(tmp_path):
    sessions = SessionManager(str(tmp_path))
    first_session = sessions.create()
    second_session = sessions.create()
    first = sessions.task_store.create(
        {"task": "first"}, session_id=first_session.session_id
    )
    sessions.task_store.create(
        {"task": "second"}, session_id=second_session.session_id
    )

    listed = sessions.task_store.list(session_id=first_session.session_id)

    assert [task.id for task in listed] == [first.id]
    first_session.close()
    second_session.close()


def test_expired_lease_is_requeued_then_exhausted(tmp_path):
    sessions = SessionManager(str(tmp_path))
    session = sessions.create()
    store = sessions.task_store
    task = store.create(
        {"task": "recover"}, session_id=session.session_id, max_attempts=2
    )

    store.claim(task.id, "dead-1", lease_seconds=-1)
    store.mark_running(task.id, "dead-1")
    first = store.recover_expired_leases()
    assert first[0].status == TaskStatus.QUEUED
    assert store.list_attempts(task.id)[0].status == "expired"

    store.claim(task.id, "dead-2", lease_seconds=-1)
    store.mark_running(task.id, "dead-2")
    second = store.recover_expired_leases()
    assert second[0].status == TaskStatus.FAILED
    assert second[0].attempt_count == 2
    session.close()


@pytest.mark.asyncio
async def test_durable_manager_retries_and_persists_result(tmp_path):
    sessions = SessionManager(str(tmp_path))
    session = sessions.create()
    agent = make_agent(session.session_id)
    agent.run_to_completion = AsyncMock(
        side_effect=[RuntimeError("temporary"), "completed on retry"]
    )
    manager = DurableTaskManager(
        sessions.task_store,
        lease_seconds=1,
        heartbeat_interval=0.05,
        retry_base_seconds=0.01,
        retry_max_seconds=0.01,
    )

    task_id = manager.launch(agent, "work", max_attempts=2)
    await manager._async_tasks[task_id]

    state = sessions.task_store.get(task_id)
    assert state.status == TaskStatus.SUCCEEDED
    assert state.attempt_count == 2
    assert state.result == {"output": "completed on retry"}
    assert [a.status for a in sessions.task_store.list_attempts(task_id)] == [
        "failed",
        "succeeded",
    ]
    assert manager.get(task_id).status == "completed"
    session.close()


@pytest.mark.asyncio
async def test_launch_persists_only_explicit_reconstruction_descriptor(tmp_path):
    sessions = SessionManager(str(tmp_path))
    session = sessions.create()
    manager = DurableTaskManager(sessions.task_store)
    descriptor = {
        "version": 1,
        "agent_type": "Explore",
        "model": "inherit",
    }

    resumable_id = manager.launch(
        make_agent(session.session_id),
        "inspect",
        resume_spec=descriptor,
    )
    forked_id = manager.launch(
        make_agent(session.session_id),
        "",
        fork_conversation=SimpleNamespace(),
        resume_spec=descriptor,
    )

    resumable = sessions.task_store.get(resumable_id)
    forked = sessions.task_store.get(forked_id)
    assert resumable.metadata["resumable"] is True
    assert resumable.metadata["resume_spec"] == descriptor
    assert forked.metadata["resumable"] is False
    assert forked.metadata["resume_spec"] is None
    assert "fork conversation" in forked.metadata["resume_reason"]
    await manager.shutdown()
    session.close()


@pytest.mark.asyncio
async def test_graceful_shutdown_requeues_resumable_task_and_preserves_budget(
    tmp_path,
):
    sessions = SessionManager(str(tmp_path))
    session = sessions.create()
    started = asyncio.Event()
    agent = make_agent(session.session_id)

    async def wait_forever(*_args, **_kwargs):
        started.set()
        await asyncio.Event().wait()

    agent.run_to_completion = wait_forever
    first_manager = DurableTaskManager(sessions.task_store, worker_id="worker-one")
    task_id = first_manager.launch(
        agent,
        "resume after restart",
        max_attempts=1,
        resume_spec={"version": 1, "agent_type": "Explore"},
    )
    await started.wait()

    await first_manager.shutdown()

    queued = sessions.task_store.get(task_id)
    assert queued.status == TaskStatus.QUEUED
    assert queued.attempt_count == 1
    assert queued.max_attempts == 2
    assert queued.lease_owner is None
    assert [attempt.status for attempt in sessions.task_store.list_attempts(task_id)] == [
        "interrupted"
    ]

    second_manager = DurableTaskManager(sessions.task_store, worker_id="worker-two")
    resumed_agent = make_agent(session.session_id, "finished after restart")
    second_manager.adopt_persisted(task_id, resumed_agent)
    await second_manager._async_tasks[task_id]

    completed = sessions.task_store.get(task_id)
    assert completed.status == TaskStatus.SUCCEEDED
    assert completed.attempt_count == 2
    assert [attempt.status for attempt in sessions.task_store.list_attempts(task_id)] == [
        "interrupted",
        "succeeded",
    ]
    session.close()


@pytest.mark.asyncio
async def test_user_cancel_does_not_requeue_resumable_task(tmp_path):
    sessions = SessionManager(str(tmp_path))
    session = sessions.create()
    started = asyncio.Event()
    agent = make_agent(session.session_id)

    async def wait_forever(*_args, **_kwargs):
        started.set()
        await asyncio.Event().wait()

    agent.run_to_completion = wait_forever
    manager = DurableTaskManager(sessions.task_store)
    task_id = manager.launch(
        agent,
        "cancel me",
        resume_spec={"version": 1, "agent_type": "Explore"},
    )
    await started.wait()

    assert manager.cancel(task_id) is True
    await manager._async_tasks[task_id]

    assert sessions.task_store.get(task_id).status == TaskStatus.CANCELLED
    assert sessions.task_store.list_attempts(task_id)[0].status == "cancelled"
    session.close()


@pytest.mark.asyncio
async def test_shutdown_before_first_timeslice_keeps_only_resumable_task(tmp_path):
    sessions = SessionManager(str(tmp_path))
    session = sessions.create()
    manager = DurableTaskManager(sessions.task_store)
    resumable_id = manager.launch(
        make_agent(session.session_id),
        "resume later",
        resume_spec={"version": 1},
    )
    transient_id = manager.launch(
        make_agent(session.session_id),
        "cannot reconstruct",
    )

    await manager.shutdown()

    assert sessions.task_store.get(resumable_id).status == TaskStatus.QUEUED
    assert sessions.task_store.get(transient_id).status == TaskStatus.CANCELLED
    assert manager._async_tasks == {}
    session.close()


@pytest.mark.asyncio
async def test_launch_links_task_to_parent_run(tmp_path):
    sessions = SessionManager(str(tmp_path))
    session = sessions.create()
    parent_run = sessions.run_store.create_run(
        session.session_id,
        input="lead request",
        agent_id="lead",
        trace_id="trace-parent",
    )
    agent = make_agent(session.session_id)
    agent.parent_run_id = parent_run.id
    started = asyncio.Event()

    async def wait_forever(*_args, **_kwargs):
        started.set()
        await asyncio.Event().wait()

    agent.run_to_completion = wait_forever
    manager = DurableTaskManager(sessions.task_store)

    task_id = manager.launch(agent, "child work")
    await started.wait()

    assert sessions.task_store.get(task_id).run_id == parent_run.id
    manager.cancel(task_id)
    await manager._async_tasks[task_id]
    session.close()


@pytest.mark.asyncio
async def test_heartbeat_prevents_other_worker_from_recovering_task(tmp_path):
    sessions = SessionManager(str(tmp_path))
    session = sessions.create()
    started = asyncio.Event()
    release = asyncio.Event()
    agent = make_agent(session.session_id)

    async def slow(*args, **kwargs):
        started.set()
        await release.wait()
        return "done"

    agent.run_to_completion = slow
    manager = DurableTaskManager(
        sessions.task_store,
        lease_seconds=0.15,
        heartbeat_interval=0.03,
    )
    task_id = manager.launch(agent, "slow")
    await started.wait()
    await asyncio.sleep(0.22)

    assert sessions.task_store.recover_expired_leases() == []
    assert sessions.task_store.get(task_id).status == TaskStatus.RUNNING
    release.set()
    await manager._async_tasks[task_id]
    session.close()


@pytest.mark.asyncio
async def test_large_task_result_is_offloaded(tmp_path):
    sessions = SessionManager(str(tmp_path))
    session = sessions.create()
    output = "x" * (RESULT_INLINE_LIMIT + 100)
    agent = make_agent(session.session_id, output)
    manager = DurableTaskManager(sessions.task_store)

    task_id = manager.launch(agent, "large")
    await manager._async_tasks[task_id]

    state = sessions.task_store.get(task_id)
    assert state.status == TaskStatus.SUCCEEDED
    assert state.result_path is not None
    assert "full result stored" in state.result["output"]
    assert Path(state.result_path).read_text(encoding="utf-8") == output
    session.close()


@pytest.mark.asyncio
async def test_expired_and_orphaned_task_results_are_garbage_collected(tmp_path):
    sessions = SessionManager(str(tmp_path))
    session = sessions.create()
    output = "x" * (RESULT_INLINE_LIMIT + 100)
    manager = DurableTaskManager(
        sessions.task_store,
        result_retention_days=30,
    )
    task_id = manager.launch(make_agent(session.session_id, output), "large")
    await manager._async_tasks[task_id]
    state = sessions.task_store.get(task_id)
    result_path = Path(state.result_path)
    orphan = result_path.parent / "orphan.txt"
    orphan.write_text("orphan", encoding="utf-8")

    removed = manager.cleanup_result_artifacts(
        now=datetime.now(UTC) + timedelta(days=31)
    )

    refreshed = sessions.task_store.get(task_id)
    assert removed == 2
    assert result_path.exists() is False
    assert orphan.exists() is False
    assert refreshed.result_path is None
    assert "full result stored" in refreshed.result["output"]
    session.close()


def test_result_gc_refuses_paths_outside_managed_directory(tmp_path):
    sessions = SessionManager(str(tmp_path / "project"))
    session = sessions.create()
    outside = tmp_path / "outside.txt"
    outside.write_text("keep", encoding="utf-8")
    task = sessions.task_store.create(
        {"task": "unsafe path"}, session_id=session.session_id
    )
    sessions.task_store.transition(
        task.id,
        TaskStatus.CANCELLED,
        result_path=str(outside),
    )
    manager = DurableTaskManager(
        sessions.task_store,
        result_retention_days=1,
    )

    removed = manager.cleanup_result_artifacts(
        now=datetime.now(UTC) + timedelta(days=2)
    )

    assert removed == 0
    assert outside.read_text(encoding="utf-8") == "keep"
    assert sessions.task_store.get(task.id).result_path == str(outside)
    session.close()


@pytest.mark.asyncio
async def test_team_concurrency_limit_is_enforced(tmp_path):
    sessions = SessionManager(str(tmp_path))
    session = sessions.create()
    active = 0
    peak = 0

    async def measured(*args, **kwargs):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.05)
        active -= 1
        return "done"

    manager = DurableTaskManager(
        sessions.task_store, max_concurrency=4, per_team_concurrency=1
    )
    agents = [make_agent(session.session_id) for _ in range(2)]
    for agent in agents:
        agent.team_name = "same-team"
        agent.run_to_completion = measured
    task_ids = [manager.launch(agent, "work") for agent in agents]
    await asyncio.gather(*(manager._async_tasks[task_id] for task_id in task_ids))

    assert peak == 1
    session.close()


@pytest.mark.asyncio
async def test_durable_manager_waits_for_dependencies(tmp_path):
    sessions = SessionManager(str(tmp_path))
    session = sessions.create()
    release = asyncio.Event()
    parent = make_agent(session.session_id)
    child = make_agent(session.session_id)

    async def parent_work(*args, **kwargs):
        await release.wait()
        return "parent done"

    parent.run_to_completion = parent_work
    manager = DurableTaskManager(sessions.task_store)
    parent_id = manager.launch(parent, "parent")
    child_id = manager.launch(child, "child", dependencies=[parent_id])
    await asyncio.sleep(0.05)

    assert child.run_to_completion.await_count == 0
    assert manager.get(child_id).status == "blocked"
    release.set()
    await asyncio.gather(manager._async_tasks[parent_id], manager._async_tasks[child_id])
    assert child.run_to_completion.await_count == 1
    assert sessions.task_store.get(child_id).status == TaskStatus.SUCCEEDED
    session.close()


@pytest.mark.asyncio
async def test_expired_task_can_be_adopted_by_another_worker(tmp_path):
    sessions = SessionManager(str(tmp_path))
    session = sessions.create()
    task = sessions.task_store.create(
        {"task": "continue me", "name": "recovered"},
        session_id=session.session_id,
        max_attempts=2,
    )
    sessions.task_store.claim(task.id, "dead-worker", lease_seconds=-1)
    sessions.task_store.mark_running(task.id, "dead-worker")
    sessions.task_store.recover_expired_leases()

    new_agent = make_agent(session.session_id, "adopted result")
    manager = DurableTaskManager(sessions.task_store, worker_id="new-worker")
    manager.adopt_persisted(task.id, new_agent)
    await manager._async_tasks[task.id]

    state = sessions.task_store.get(task.id)
    assert state.status == TaskStatus.SUCCEEDED
    assert state.attempt_count == 2
    assert new_agent.run_to_completion.await_args.args[0] == "continue me"
    session.close()


def test_recovered_queued_task_can_be_cancelled_without_live_handle(tmp_path):
    sessions = SessionManager(str(tmp_path))
    session = sessions.create()
    task = sessions.task_store.create(
        {"task": "continue me", "name": "recovered"},
        session_id=session.session_id,
    )
    manager = DurableTaskManager(sessions.task_store)

    assert manager.cancel(task.id) is True
    state = sessions.task_store.get(task.id)
    assert state.status == TaskStatus.CANCELLED
    assert state.error == "Task was cancelled before adoption"
    session.close()


@pytest.mark.asyncio
async def test_periodic_maintenance_recovers_later_expired_lease(tmp_path):
    sessions = SessionManager(str(tmp_path))
    session = sessions.create()
    manager = DurableTaskManager(
        sessions.task_store,
        maintenance_interval=0.01,
    )
    task = sessions.task_store.create(
        {"task": "recover later"},
        session_id=session.session_id,
        max_attempts=2,
    )
    sessions.task_store.claim(task.id, "dead-worker", lease_seconds=-1)
    sessions.task_store.mark_running(task.id, "dead-worker")

    manager.start_maintenance()
    manager.start_maintenance()
    maintenance_task = manager._maintenance_task
    for _ in range(20):
        if sessions.task_store.get(task.id).status == TaskStatus.QUEUED:
            break
        await asyncio.sleep(0.01)

    assert sessions.task_store.get(task.id).status == TaskStatus.QUEUED
    assert manager._maintenance_task is maintenance_task
    await manager.shutdown()
    assert manager._maintenance_task is None
    session.close()


def test_team_task_board_uses_sqlite_and_dependency_relations(tmp_path):
    sessions = SessionManager(str(tmp_path))
    board = DurableSharedTaskStore(sessions.task_store, "alpha")
    first = board.create("Design", created_by="lead")
    second = board.create("Implement", blocked_by=[first.id], assignee="worker")

    assert [task.title for task in board.list_tasks()] == ["Design", "Implement"]
    assert sessions.task_store.dependencies(f"shared:alpha:{second.id}") == [
        f"shared:alpha:{first.id}"
    ]
    assert sessions.task_store.dependencies_ready(f"shared:alpha:{second.id}") is False

    updated = board.update(first.id, status="completed", add_blocks=[second.id])
    assert updated.status == "completed"
    assert board.get(second.id).blocked_by == [first.id]
    assert sessions.task_store.dependencies_ready(f"shared:alpha:{second.id}") is True


def test_durable_team_task_claim_is_atomic_and_dependency_aware(tmp_path):
    sessions = SessionManager(str(tmp_path))
    board = DurableSharedTaskStore(sessions.task_store, "alpha")
    prerequisite = board.create("Design")
    implementation = board.create("Implement", blocked_by=[prerequisite.id])

    with pytest.raises(ValueError, match="blocked by incomplete"):
        board.claim(implementation.id, "alice")

    board.update(prerequisite.id, status="completed")
    claimed = board.claim(implementation.id, "alice")
    assert claimed.status == "in_progress"
    assert claimed.assignee == "alice"
    assert board.claim(implementation.id, "alice").assignee == "alice"

    with pytest.raises(ValueError, match="already claimed by 'alice'"):
        board.claim(implementation.id, "bob")


def test_durable_team_task_rejects_missing_and_cyclic_dependencies(tmp_path):
    sessions = SessionManager(str(tmp_path))
    board = DurableSharedTaskStore(sessions.task_store, "alpha")
    first = board.create("First")
    second = board.create("Second", blocked_by=[first.id])

    with pytest.raises(ValueError, match="cannot contain a cycle"):
        board.update(first.id, add_blocked_by=[second.id])
    with pytest.raises(ValueError, match="Unknown shared task dependencies"):
        board.create("Broken", blocked_by=["999"])
    assert [task.id for task in board.list_tasks()] == [first.id, second.id]


def test_durable_team_task_allocates_unique_ids_concurrently(tmp_path):
    sessions = SessionManager(str(tmp_path))

    def create(index: int) -> str:
        board = DurableSharedTaskStore(sessions.task_store, "alpha")
        return board.create(f"Task {index}").id

    with ThreadPoolExecutor(max_workers=12) as pool:
        ids = list(pool.map(create, range(20)))

    assert len(set(ids)) == 20
    board = DurableSharedTaskStore(sessions.task_store, "alpha")
    assert len(board.list_tasks()) == 20


def test_durable_team_task_persists_priority_and_progress(tmp_path):
    sessions = SessionManager(str(tmp_path))
    board = DurableSharedTaskStore(sessions.task_store, "alpha")
    task = board.create("Important", priority="high", progress=20)

    assert board.get(task.id).priority == "high"
    assert board.get(task.id).progress == 20
    updated = board.update(task.id, progress=75)
    assert updated.progress == 75
    assert [item.id for item in board.list_tasks(priority="high")] == [task.id]

    completed = board.update(task.id, status="completed")
    assert completed.progress == 100
