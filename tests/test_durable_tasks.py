from __future__ import annotations

import asyncio
from pathlib import Path
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
    agent.team_name = ""
    agent._team_manager = None
    agent.total_input_tokens = 12
    agent.total_output_tokens = 7
    agent.run_to_completion = AsyncMock(return_value=result)
    return agent


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
