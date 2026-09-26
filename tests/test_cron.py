from __future__ import annotations

import asyncio
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from valecode.agent import Agent
from valecode.agents.durable_task_manager import DurableTaskManager
from valecode.agents.parser import AgentDef
from valecode.agents.trace import TraceManager
from valecode.persistence import Database, SessionStore, TaskStore, TaskStatus, RunStore
from valecode.persistence.schedule_store import ScheduleStore
from valecode.permissions import PermissionChecker, PermissionMode, PathSandbox, RuleEngine, DangerousCommandDetector
from valecode.runtime.cron import CronRuntime, install_cron
from valecode.runtime.schedules import next_occurrence
from valecode.tools import create_default_registry
from valecode.tools.agent_tool import AgentTool
from valecode.tools.cron import CronCreate, CronDelete, CronList, CronUpdate, CreateParams, IdParams, UpdateParams, EmptyParams
from valecode.tools.base import TextDelta, StreamEnd, ToolCallComplete
from tests.test_agent import MockLLMClient

NOW = datetime(2026, 9, 26, tzinfo=UTC)


def _admit_process(path, work_dir, timestamp):
    return ScheduleStore(Database(path)).admit_due("s", work_dir, now=datetime.fromisoformat(timestamp))


@pytest.fixture
def store(tmp_path):
    db = Database(tmp_path / "control.db")
    db.initialize()
    SessionStore(db).upsert("s")
    SessionStore(db).upsert("other")
    return ScheduleStore(db)


def descriptor():
    return AgentTool._resume_spec(AgentDef("probe", "test", max_turns=3), None)


def plan(store, tmp_path, **kwargs):
    params = dict(session_id="s", name="test", prompt="read a note", work_dir=str(tmp_path),
        kind="interval", spec={"every_seconds": 60}, timezone="UTC", agent_spec=descriptor(), now=NOW)
    params.update(kwargs)
    return store.create(**params)


def test_calendar_timezone_once_cron_and_interval():
    assert next_occurrence("cron", {"cron": "0 9 * * *"}, "Asia/Shanghai", NOW) == NOW + timedelta(hours=1)
    assert next_occurrence("once", {"run_at": "2026-09-26T09:00:00"}, "Asia/Shanghai", NOW) == NOW + timedelta(hours=1)
    assert next_occurrence("once", {"run_at": "2026-09-25T09:00:00Z"}, "UTC", NOW) is None
    assert next_occurrence("interval", {"every_seconds": 60}, "UTC", NOW) == NOW + timedelta(seconds=60)


@pytest.mark.parametrize("kind,spec,zone", [
    ("interval", {"every_seconds": True}, "UTC"), ("interval", {"every_seconds": 1}, "UTC"),
    ("cron", {"cron": "* * * * * *"}, "UTC"), ("cron", {"cron": "bad * * * *"}, "UTC"),
    ("once", {"run_at": "2026-11-01T01:30:00"}, "America/New_York"),
    ("once", {"run_at": "2026-03-08T02:30:00"}, "America/New_York"),
    ("once", {"run_at": "tomorrow"}, "UTC"), ("unknown", {}, "UTC"),
    ("interval", {"every_seconds": 60}, "Unknown/Zone"),
])
def test_bad_calendar_rejected(kind, spec, zone):
    with pytest.raises((ValueError, KeyError)):
        next_occurrence(kind, spec, zone, NOW)


def test_missed_run_coalesces_and_no_overlap_or_cross_session(store, tmp_path):
    created = plan(store, tmp_path)
    late = NOW + timedelta(days=2)
    ids = store.admit_due("s", str(tmp_path), now=late)
    assert len(ids) == 1
    assert store.list("s")[0]["next_run"] == (late + timedelta(seconds=60)).timestamp()
    assert store.admit_due("s", str(tmp_path), now=late + timedelta(minutes=2)) == []
    assert store.admit_due("other", str(tmp_path), now=late) == []
    assert store.admit_due("s", str(tmp_path / "different"), now=late) == []
    tasks = TaskStore(store.database)
    tasks.claim(ids[0], "worker")
    assert store.admit_due("s", str(tmp_path), now=late + timedelta(minutes=3)) == []
    tasks.transition(ids[0], TaskStatus.RUNNING)
    tasks.transition(ids[0], TaskStatus.SUCCEEDED)
    assert len(store.admit_due("s", str(tmp_path), now=late + timedelta(minutes=4))) == 1
    assert store.list("s")[0]["id"] == created["id"]


def test_multiple_processes_only_admit_one_occurrence(store, tmp_path):
    plan(store, tmp_path)
    with ProcessPoolExecutor(max_workers=3) as workers:
        futures = [workers.submit(_admit_process, str(store.database.path), str(tmp_path), (NOW + timedelta(minutes=1)).isoformat()) for _ in range(6)]
        assert sum(len(f.result(timeout=30)) for f in futures) == 1
    assert len(TaskStore(store.database).list(session_id="s")) == 1


def test_admission_failure_rolls_back_plan_occurrence_and_task(store, tmp_path, monkeypatch):
    created = plan(store, tmp_path)
    def fail(*args, **kwargs):
        raise RuntimeError("fault injection")
    monkeypatch.setattr(store.events, "_append", fail)
    with pytest.raises(RuntimeError):
        store.admit_due("s", str(tmp_path), now=NOW + timedelta(minutes=1))
    assert store.list("s")[0]["next_run"] == created["next_run"]
    assert TaskStore(store.database).list(session_id="s") == []
    with store.database.reader() as db:
        assert db.execute("SELECT count(*) FROM schedule_occurrences").fetchone()[0] == 0


def test_pause_cancel_queue_resume_from_now_and_soft_delete(store, tmp_path):
    created = plan(store, tmp_path)
    ids = store.admit_due("s", str(tmp_path), now=NOW + timedelta(minutes=1))
    with pytest.raises(KeyError):
        store.change(created["id"], "other", "delete")
    store.change(created["id"], "s", "pause", now=NOW + timedelta(minutes=1))
    assert TaskStore(store.database).get(ids[0]).status == TaskStatus.CANCELLED
    assert store.admit_due("s", str(tmp_path), now=NOW + timedelta(days=1)) == []
    resumed = store.change(created["id"], "s", "resume", now=NOW + timedelta(days=1))
    assert resumed["next_run"] == (NOW + timedelta(days=1, minutes=1)).timestamp()
    store.change(created["id"], "s", "delete")
    assert store.list("s") == []
    with store.database.reader() as db:
        assert db.execute("SELECT count(*) FROM schedule_occurrences").fetchone()[0] == 1


def test_pause_does_not_cancel_claimed_work_and_once_tracks_real_completion(store, tmp_path):
    created = plan(store, tmp_path, kind="once", spec={"run_at": (NOW + timedelta(minutes=1)).isoformat()})
    ids = store.admit_due("s", str(tmp_path), now=NOW + timedelta(minutes=1))
    assert store.list("s")[0]["status"] == "running"
    tasks = TaskStore(store.database)
    tasks.claim(ids[0], "worker")
    store.change(created["id"], "s", "pause")
    assert tasks.get(ids[0]).status == TaskStatus.LEASED
    with pytest.raises(ValueError):
        store.change(created["id"], "s", "resume", now=NOW + timedelta(minutes=2))
    tasks.transition(ids[0], TaskStatus.RUNNING)
    tasks.transition(ids[0], TaskStatus.SUCCEEDED)
    assert store.list("s")[0]["status"] == "paused"  # explicit user state wins


@pytest.mark.parametrize("status,result", [(TaskStatus.SUCCEEDED, "completed"), (TaskStatus.FAILED, "failed")])
def test_once_terminal_state_is_not_faked_at_admission(store, tmp_path, status, result):
    plan(store, tmp_path, kind="once", spec={"run_at": (NOW + timedelta(minutes=1)).isoformat()})
    ids = store.admit_due("s", str(tmp_path), now=NOW + timedelta(minutes=1))
    tasks = TaskStore(store.database)
    tasks.claim(ids[0], "worker")
    tasks.transition(ids[0], TaskStatus.RUNNING)
    tasks.transition(ids[0], status)
    assert store.admit_due("s", str(tmp_path), now=NOW + timedelta(minutes=2)) == []
    assert store.list("s")[0]["status"] == result


def make_tool(store, tmp_path, responses=None):
    parent = Agent(MockLLMClient(responses or [[TextDelta("scheduled done"), StreamEnd("end_turn", 1, 1)]]), create_default_registry(), "anthropic", work_dir=str(tmp_path), run_store=RunStore(store.database))
    parent.session_id = "s"
    parent.permission_checker = PermissionChecker(DangerousCommandDetector(), PathSandbox(str(tmp_path)), RuleEngine(), PermissionMode.BYPASS)
    manager = DurableTaskManager(TaskStore(store.database))
    definition = AgentDef("probe", "test", permission_mode="bypassPermissions", max_turns=3)
    tool = AgentTool(SimpleNamespace(get=lambda name: definition if name == "probe" else None), manager, TraceManager(), parent)
    return tool


@pytest.mark.asyncio
async def test_tools_create_validate_permissions_and_exclude_recursive_child(store, tmp_path):
    tool = make_tool(store, tmp_path)
    runtime = CronRuntime(tool)
    params = CreateParams(name="test", prompt="read", subagent_type="probe", schedule_type="interval", schedule_spec={"every_seconds": 60})
    result = await CronCreate(runtime).execute(params)
    assert not result.is_error
    sid = json.loads(result.output)["schedule_id"]
    assert runtime.store.list("s")[0]["agent"]["permission_mode"] == "default"
    tool._parent_agent.permission_checker.mode = PermissionMode.DEFAULT
    assert tool._parent_agent.permission_checker.check(CronCreate(runtime), params.model_dump()).effect == "ask"
    tool._parent_agent.registry.register(CronCreate(runtime))
    from valecode.agents.tool_filter import resolve_agent_tools
    assert resolve_agent_tools(tool._parent_agent.registry, AgentDef("probe", "test"), True).get("CronCreate") is None
    listing = await CronList(runtime).execute(EmptyParams())
    assert sid in listing.output
    assert not (await CronUpdate(runtime).execute(UpdateParams(schedule_id=sid, action="pause"))).is_error
    assert not (await CronUpdate(runtime).execute(UpdateParams(schedule_id=sid, action="resume"))).is_error
    assert not (await CronDelete(runtime).execute(IdParams(schedule_id=sid))).is_error
    assert (await CronCreate(runtime).execute(params.model_copy(update={"subagent_type": "unknown"}))).is_error
    await tool._task_manager.shutdown()


@pytest.mark.asyncio
async def test_restart_adopts_due_instance_once_and_closes_runtime(store, tmp_path):
    created = plan(store, tmp_path, kind="once", spec={"run_at": (NOW + timedelta(minutes=1)).isoformat()})
    queued = store.admit_due("s", str(tmp_path), now=NOW + timedelta(minutes=1))
    # Simulate process death after durable admission, before in-memory launch.
    fresh_store = ScheduleStore(Database(store.database.path))
    tool = make_tool(fresh_store, tmp_path)
    runtime = install_cron(tool, tool._parent_agent.registry)
    try:
        runtime.tick(now=NOW + timedelta(minutes=2))
        runtime.tick(now=NOW + timedelta(minutes=2))
        assert len(tool._task_manager._async_tasks) == 1
        bg = tool._task_manager.get(queued[0])
        assert bg.agent.permission_checker.mode == PermissionMode.DEFAULT
        assert bg.agent.cancellation_token is not tool._parent_agent.cancellation_token
        await asyncio.gather(*tool._task_manager._async_tasks.values())
        assert tool._task_manager.task_store.get(queued[0]).status == TaskStatus.SUCCEEDED
        runtime.tick(now=NOW + timedelta(minutes=3))
        assert fresh_store.list("s")[0]["status"] == "completed"
    finally:
        await runtime.close()
        await tool._task_manager.shutdown()
    assert runtime._task is None and runtime.tick() == []
    assert created["id"] in (await CronList(runtime).execute(EmptyParams())).output


@pytest.mark.asyncio
async def test_scheduled_agent_preserves_current_rules_and_denies_unattended_write(store, tmp_path):
    destination = tmp_path / "unsafe.txt"
    tool = make_tool(store, tmp_path, [
        [ToolCallComplete("write", "WriteFile", {"file_path": str(destination), "content": "must not write"}), StreamEnd("tool_use", 1, 1)],
        [TextDelta("permission denied"), StreamEnd("end_turn", 1, 1)],
    ])
    plan(store, tmp_path)
    runtime = CronRuntime(tool)
    try:
        ids = runtime.tick(now=NOW + timedelta(minutes=1))
        await asyncio.gather(*tool._task_manager._async_tasks.values())
        assert ids and not destination.exists()
        assert tool._task_manager.get(ids[0]).agent.permission_checker.mode == PermissionMode.DEFAULT
    finally:
        await runtime.close()
        await tool._task_manager.shutdown()


def test_worker_cancellation_does_not_touch_another_lease(store, tmp_path):
    plan(store, tmp_path)
    ids = store.admit_due("s", str(tmp_path), now=NOW + timedelta(minutes=1))
    tasks = TaskStore(store.database)
    tasks.claim(ids[0], "owner")
    assert tasks.cancel_for_worker(ids[0], "competitor", error="not ours") is None
    assert tasks.get(ids[0]).status == TaskStatus.LEASED
    assert tasks.cancel_for_worker(ids[0], "owner", error="cancel").status == TaskStatus.CANCELLED


@pytest.mark.asyncio
async def test_two_workers_adopting_same_instance_exit_after_terminal_result(store, tmp_path):
    plan(store, tmp_path)
    tool_a, tool_b = make_tool(store, tmp_path), make_tool(store, tmp_path)
    runtime_a, runtime_b = CronRuntime(tool_a), CronRuntime(tool_b)
    try:
        ids = runtime_a.tick(now=NOW + timedelta(minutes=1))
        runtime_b.tick(now=NOW + timedelta(minutes=1))
        await asyncio.wait_for(asyncio.gather(*tool_a._task_manager._async_tasks.values(), *tool_b._task_manager._async_tasks.values()), 3)
        assert TaskStore(store.database).get(ids[0]).attempt_count == 1
        assert TaskStore(store.database).get(ids[0]).status == TaskStatus.SUCCEEDED
        assert not tool_a._task_manager._async_tasks and not tool_b._task_manager._async_tasks
    finally:
        await runtime_a.close()
        await runtime_b.close()
        await tool_a._task_manager.shutdown()
        await tool_b._task_manager.shutdown()


@pytest.mark.asyncio
async def test_started_scheduled_task_is_not_silently_replayed_on_shutdown(store, tmp_path):
    plan(store, tmp_path)
    tool = make_tool(store, tmp_path)
    entered = asyncio.Event()
    async def hang(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()
    tool._parent_agent.client.stream = hang  # build overridden below for a real wait
    original = tool._build_recovered_agent
    def build(*args, **kwargs):
        agent = original(*args, **kwargs)
        agent.run_to_completion = hang
        return agent
    tool._build_recovered_agent = build
    runtime = CronRuntime(tool)
    ids = runtime.tick(now=NOW + timedelta(minutes=1))
    await asyncio.wait_for(entered.wait(), 1)
    await runtime.close()
    await tool._task_manager.shutdown()
    assert TaskStore(store.database).get(ids[0]).status == TaskStatus.CANCELLED


@pytest.mark.asyncio
async def test_remote_initialization_registers_cron_and_shutdown_closes_it(tmp_path, monkeypatch):
    from valecode import remote as module
    from valecode.config import ProviderConfig
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    monkeypatch.setattr(module, "create_client", lambda provider: MockLLMClient([]))
    provider = ProviderConfig("test", "anthropic", "https://example.invalid", "model", "fake")
    server = module.RemoteServer([provider])
    server._init_agent()
    assert server.registry.get("CronCreate") is not None
    assert server.command_registry.find("cron") is not None
    assert server.cron_runtime._task is None  # only starts after MCP init in run()
    server.cron_runtime.start()
    await server._shutdown()
    assert server.cron_runtime._closed and server.cron_runtime._task is None


@pytest.mark.asyncio
async def test_tui_initialization_and_user_exit_close_timer(tmp_path, monkeypatch):
    from valecode import app as module
    from valecode.config import ProviderConfig
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    monkeypatch.setattr(module, "create_client", lambda provider: MockLLMClient([]))
    provider = ProviderConfig("test", "anthropic", "https://example.invalid", "model", "fake")
    app = module.ValeCodeApp([provider])
    async with app.run_test() as pilot:
        assert app.registry.get("CronCreate") is not None
        assert app.command_registry.find("cron") is not None
        runtime = app.cron_runtime
        await app.action_handle_ctrl_c()
        await pilot.pause()
        assert runtime._closed and runtime._task is None


@pytest.mark.asyncio
async def test_cron_slash_command_is_session_scoped(store, tmp_path):
    from tests.test_commands import _make_context
    from valecode.commands.handlers.cron import create_cron_command
    created = plan(store, tmp_path)
    runtime = CronRuntime(make_tool(store, tmp_path))
    command = create_cron_command(runtime)
    ctx = _make_context("pause " + created["id"])
    ctx.session = SimpleNamespace(session_id="other")
    await command.handler(ctx)
    assert store.list("s")[0]["status"] == "enabled"
    ctx.session = SimpleNamespace(session_id="s")
    await command.handler(ctx)
    assert store.list("s")[0]["status"] == "paused"
    ctx.args = "list"
    await command.handler(ctx)
    assert created["id"] in ctx.ui.messages[-1]
    await runtime.close()
    await runtime.agent_tool._task_manager.shutdown()


@pytest.mark.asyncio
async def test_real_timer_waits_for_readiness_and_dispatches(store, tmp_path):
    plan(store, tmp_path)
    tool = make_tool(store, tmp_path)
    ready = False
    runtime = CronRuntime(tool, poll_interval=0.05, ready=lambda: ready)
    original_tick = runtime.tick
    runtime.tick = lambda: original_tick(now=NOW + timedelta(minutes=1))
    runtime.start()
    try:
        await asyncio.sleep(0.12)
        assert TaskStore(store.database).list(session_id="s") == []
        ready = True
        for _ in range(60):
            await asyncio.sleep(0.05)
            tasks = TaskStore(store.database).list(session_id="s")
            if tasks and tasks[0].status == TaskStatus.SUCCEEDED:
                break
        assert len(tasks) == 1 and tasks[0].status == TaskStatus.SUCCEEDED
    finally:
        await runtime.close()
        await tool._task_manager.shutdown()


@pytest.mark.asyncio
async def test_prompt_entry_registers_cron_and_closes_owned_timer(tmp_path, monkeypatch):
    from unittest.mock import patch
    from tests.test_cli import _prompt_config, _PromptClient
    from valecode.__main__ import _run_prompt, _PromptResources
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    config = _prompt_config()
    config.mcp_servers = []
    client = _PromptClient()
    resources = _PromptResources()
    async def no_resolve(provider):
        return None
    with patch("valecode.client.create_client", return_value=client), patch("valecode.client.resolve_context_window", no_resolve):
        try:
            await _run_prompt(config, PermissionMode.DEFAULT, None, "hello", _resources=resources)
            assert "CronCreate" in client.tool_names
            runtime = resources.cron_runtime
        finally:
            await resources.close()
    assert runtime._closed and runtime._task is None


@pytest.mark.asyncio
async def test_real_timer_waits_for_readiness_and_dispatches(store, tmp_path):
    plan(store, tmp_path)
    tool = make_tool(store, tmp_path)
    ready = False
    runtime = CronRuntime(tool, poll_interval=0.05, ready=lambda: ready)
    original_tick = runtime.tick
    runtime.tick = lambda: original_tick(now=NOW + timedelta(minutes=1))
    runtime.start()
    try:
        await asyncio.sleep(0.12)
        assert TaskStore(store.database).list(session_id="s") == []
        ready = True
        for _ in range(60):
            await asyncio.sleep(0.05)
            tasks = TaskStore(store.database).list(session_id="s")
            if tasks and tasks[0].status == TaskStatus.SUCCEEDED:
                break
        assert len(tasks) == 1 and tasks[0].status == TaskStatus.SUCCEEDED
    finally:
        await runtime.close()
        await tool._task_manager.shutdown()


@pytest.mark.asyncio
async def test_prompt_entry_registers_cron_and_closes_owned_timer(tmp_path, monkeypatch):
    from unittest.mock import patch
    from tests.test_cli import _prompt_config, _PromptClient
    from valecode.__main__ import _run_prompt, _PromptResources
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    config = _prompt_config()
    config.mcp_servers = []
    client = _PromptClient()
    resources = _PromptResources()
    async def no_resolve(provider):
        return None
    with patch("valecode.client.create_client", return_value=client), patch("valecode.client.resolve_context_window", no_resolve):
        try:
            await _run_prompt(config, PermissionMode.DEFAULT, None, "hello", _resources=resources)
            assert "CronCreate" in client.tool_names
            runtime = resources.cron_runtime
        finally:
            await resources.close()
    assert runtime._closed and runtime._task is None
