from __future__ import annotations

import asyncio
import json
import shlex
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from test_teammate_worker import launch_data
from valecode.teams.worker_launch import WorkerLaunch, atomic_json
from valecode.teams.manager import TeamManager
from valecode.teams.models import AgentTeam, BackendType, TeammateInfo
from valecode.teams.progress import TeammateProgress
from valecode.persistence import TaskStore, TeamStore
from valecode.agents.trace import TraceManager
from valecode.teams.backend_detect import detect_backend, BackendDetectionError
from valecode.teams import spawn_tmux, spawn_iterm2


@pytest.mark.parametrize("inside", [True, False])
def test_tmux_exact_identity_and_shell_quoting(monkeypatch, inside):
    monkeypatch.setenv("TMUX", "session" if inside else "")
    captured = []
    monkeypatch.setattr(spawn_tmux, "_run_tmux", lambda *args: captured.append(args) or "%42")
    launch = "/tmp/project with spaces/o'neil;$(echo no).json"
    result = spawn_tmux.spawn_tmux_teammate(launch, "/tmp/project with spaces", "worker;bad")
    assert result.pane_id == "%42"
    args = captured[0]
    assert args[0] == ("new-window" if inside else "new-session")
    assert "#{pane_id}" in args and args[args.index("-c") + 1] == "/tmp/project with spaces"
    assert shlex.split(args[-1]) == [sys.executable, "-m", "valecode", "--teammate-launch", launch]
    assert "worker;bad" not in args


def test_tmux_rejects_ambiguous_target_and_never_wakes_shell(monkeypatch):
    called = []
    monkeypatch.setattr(spawn_tmux, "_run_tmux", lambda *args: called.append(args) or "window-name")
    with pytest.raises(spawn_tmux.TmuxSpawnError):
        spawn_tmux.spawn_tmux_teammate("/tmp/id.json", "/tmp", "member")
    with pytest.raises(spawn_tmux.TmuxSpawnError):
        spawn_tmux.kill_pane("any-window")
    before = len(called)
    spawn_tmux.send_keys_to_pane("%42", "anything")
    assert len(called) == before


@pytest.mark.parametrize("mode", ["tmux", "iterm2"])
def test_native_windows_and_noninteractive_fail_before_launch(monkeypatch, mode):
    monkeypatch.setattr("valecode.teams.backend_detect.sys.platform", "win32")
    with pytest.raises(BackendDetectionError):
        detect_backend(mode)
    with pytest.raises(BackendDetectionError, match="interactive"):
        detect_backend(mode, False)


@pytest.mark.asyncio
async def test_iterm_official_window_and_exact_close_api():
    session = SimpleNamespace(session_id="fixture-session", async_close=AsyncMock())
    window = SimpleNamespace(current_tab=SimpleNamespace(current_session=session))
    api = SimpleNamespace(Window=SimpleNamespace(async_create=AsyncMock(return_value=window)),
        async_get_app=AsyncMock(return_value=SimpleNamespace(get_session_by_id=lambda identity: session)))
    connection = object()
    value = await spawn_iterm2._request(connection, "spawn", "/tmp/launch with quote'.json", "/tmp/root space", api)
    assert value == session.session_id
    outer = shlex.split(api.Window.async_create.call_args.kwargs["command"])
    assert outer[:2] == ["/bin/sh", "-c"]
    assert shlex.split(outer[2])[-1] == "/tmp/launch with quote'.json"
    await spawn_iterm2._request(connection, "close", value, "", api)
    session.async_close.assert_awaited_once_with(force=True)


def make_manager(launch_data, tmp_path, monkeypatch):
    root, database, data = launch_data
    manager = TeamManager(task_store=TaskStore(database), trace_manager=TraceManager())
    member = TeammateInfo(data["member_name"], data["agent_id"], "probe", "fake", data["work_dir"], "tmux", True,
        TeammateProgress(data["member_name"], data["team_name"]))
    team = AgentTeam(data["team_name"], data["lead_id"], [member], str(tmp_path / "cache.json"), backend_type="tmux")
    manager._teams[team.name] = team
    kills = []
    monkeypatch.setattr(manager, "_kill_pane", lambda pane_id, backend: kills.append((pane_id, backend)))
    return manager, team, member, kills


@pytest.mark.asyncio
async def test_supervisor_tracks_idle_followup_terminal_without_replay(launch_data, tmp_path, monkeypatch):
    root, database, data = launch_data
    manager, team, member, kills = make_manager(launch_data, tmp_path, monkeypatch)
    launch = WorkerLaunch.prepare(root, data)
    manager.register_pane_worker(team.name, member, launch, "%42")
    launch.update_state("idle", tool_count=3, token_count=12, input_tokens=7, output_tokens=5, last_activity="ReadFile", last_message="done")
    await manager.pane_supervisor.poll()
    assert member.is_active is False and member.progress.tool_use_count == 3
    assert member.progress.status == "idle" and member.progress.last_message == "done"
    assert launch.parent_alive() and not kills
    launch.update_state("running", tool_count=4)
    await manager.pane_supervisor.poll()
    assert member.is_active is True
    launch.update_state("failed", error="fixture failure")
    await manager.pane_supervisor.poll()
    assert member.is_active is False and member.progress.status == "failed"
    assert kills == [("%42", "tmux")]
    assert not manager.pane_supervisor.workers
    assert TeamStore(database).list_members(team.name)[0].status == "stopped"
    await manager.close()


@pytest.mark.asyncio
async def test_expired_worker_stopped_not_restarted_and_worktree_preserved(launch_data, tmp_path, monkeypatch):
    root, database, data = launch_data
    manager, team, member, kills = make_manager(launch_data, tmp_path, monkeypatch)
    launch = WorkerLaunch.prepare(root, data)
    manager.register_pane_worker(team.name, member, launch, "%42")
    state = launch.state()
    state["timestamp"] -= 31
    atomic_json(launch.state_path, state)
    await manager.pane_supervisor.poll()
    assert not launch.parent_alive() and kills == [("%42", "tmux")]
    assert member.progress.status == "failed" and Path(member.worktree_path).is_dir()
    await manager.close()


@pytest.mark.parametrize("key,value", [("tool_count", True), ("token_count", -1), ("timestamp", float("inf")), ("last_activity", 4)])
def test_bad_progress_cannot_poison_supervisor(launch_data, key, value):
    root, database, data = launch_data
    launch = WorkerLaunch.prepare(root, data)
    state = launch.state() | {key: value}
    launch.state_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ValueError):
        launch.state()


def test_explicit_state_dir_does_not_inherit_other_scope_fallback(monkeypatch, tmp_path):
    from valecode.teams.models import resolve_team_dir, use_fallback_team_root
    monkeypatch.setenv("VALECODE_STATE_DIR", str(tmp_path / "first"))
    use_fallback_team_root()
    monkeypatch.setenv("VALECODE_STATE_DIR", str(tmp_path / "second"))
    assert resolve_team_dir("team") == tmp_path / "second" / "teams" / "team"


@pytest.mark.parametrize("parent,child,expected", [
    ("default", "bypassPermissions", "default"), ("acceptEdits", "bypassPermissions", "acceptEdits"),
    ("bypassPermissions", "default", "default"), ("plan", "bypassPermissions", "plan"),
])
def test_teammate_cannot_elevate_parent_permissions(parent, child, expected):
    from valecode.tools.agent_tool import AgentTool
    from valecode.permissions import PermissionMode
    tool = object.__new__(AgentTool)
    tool._parent_agent = SimpleNamespace(permission_checker=SimpleNamespace(mode=PermissionMode(parent)))
    assert tool._teammate_permission_mode(SimpleNamespace(permission_mode=child)).value == expected


@pytest.mark.asyncio
async def test_agent_pane_launch_serializes_actual_model_and_failure_is_terminal(launch_data, tmp_path, monkeypatch):
    from valecode.tools.agent_tool import AgentTool, AgentToolParams
    from valecode.tools import create_default_registry
    from valecode.permissions import PermissionMode
    from valecode.teams.mailbox import Mailbox
    root, database, data = launch_data
    manager, team, member, kills = make_manager(launch_data, tmp_path, monkeypatch)
    manager._mailboxes[team.name] = Mailbox(data["mailbox_dir"])
    parent = SimpleNamespace(registry=create_default_registry(), permission_checker=SimpleNamespace(sandbox_enabled=False))
    child = SimpleNamespace(session_id=data["session_id"], parent_run_id=data["parent_run_id"], trace_id=data["trace_id"],
        model="actual-override", registry=parent.registry, permission_checker=SimpleNamespace(mode=PermissionMode.DEFAULT))
    tool = object.__new__(AgentTool)
    tool._team_manager = manager
    tool._worktree_manager = SimpleNamespace(repo_root=str(root))
    tool._provider_config = SimpleNamespace(name="fixture", model="fake", api_key="never serialize me")
    tool._parent_agent = parent
    tool._trace_manager = TraceManager()
    definition = AgentTool._definition_from_resume_spec(data["definition"])
    captured = []
    def failure(path, project, label):
        captured.append(WorkerLaunch(path))
        raise OSError("fixture launch failure")
    monkeypatch.setattr(spawn_tmux, "spawn_tmux_teammate", failure)
    result = await tool._spawn_pane_teammate(AgentToolParams(prompt="new task", description="test", team_name=team.name),
        team, member, BackendType.TMUX, SimpleNamespace(path=data["work_dir"]), member.agent_id, member.name, definition, child)
    assert result.is_error and not member.is_active and member.progress.status == "failed"
    assert captured[0].read()["model"] == "actual-override"
    assert "never serialize me" not in captured[0].path.read_text(encoding="utf-8")
    assert not captured[0].parent_alive() and not manager.pane_supervisor.workers
    assert Path(member.worktree_path).is_dir()
    await manager.close()
