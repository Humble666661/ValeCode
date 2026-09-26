from __future__ import annotations

import asyncio
import json
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from valecode.config import ProviderConfig
from valecode.permissions import PermissionMode
from valecode.remote import RemoteServer
from valecode.runtime.harness import close_resources
from valecode.tools.ask_user import AskUserTool, AskUserParams
from valecode.tools.agent_tool import AgentToolParams
from valecode.tools.team_create import TeamCreateParams
from valecode.tools.enter_worktree import EnterWorktreeParams
from valecode.tools.exit_worktree import ExitWorktreeParams


@pytest_asyncio.fixture
async def remote(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    monkeypatch.setenv("VALECODE_STATE_DIR", str(tmp_path / "state"))
    provider = ProviderConfig("offline", "openai-compat", "https://example.invalid", "fixture", "unused")
    server = RemoteServer([provider])
    server._init_agent()
    server._broadcast = AsyncMock()
    try:
        yield server
    finally:
        await server._shutdown()


@pytest.mark.asyncio
async def test_shared_factory_remote_team_worktree_and_permission_retarget(remote, tmp_path):
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, check=True)
    (tmp_path / "seed.txt").write_text("fixture", encoding="utf-8")
    subprocess.run(["git", "add", "seed.txt"], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-m", "fixture"],
        cwd=tmp_path, capture_output=True, check=True)
    assert remote.harness.agent_tool is remote.agent_tool
    for name in ["Agent", "TeamCreate", "TeamDelete", "TaskDispatch", "CronCreate", "EnterWorktree", "ExitWorktree", "AskUserQuestion", "ExitPlanMode"]:
        assert remote.registry.get(name) is not None
    result = await remote.registry.get("TeamCreate").execute(TeamCreateParams(team_name="fixture-team"))
    assert not result.is_error
    team = remote.team_manager.get_team("fixture-team")
    assert team.backend_type == "in-process"
    enter = await remote.registry.get("EnterWorktree").execute(EnterWorktreeParams(name="fixture-worktree"))
    assert not enter.is_error
    assert remote.agent.work_dir == remote.worktree_manager.get_current_session().worktree_path
    assert Path(remote.agent.permission_checker.sandbox.project_root).resolve() == Path(remote.agent.work_dir).resolve()
    exit_result = await remote.registry.get("ExitWorktree").execute(ExitWorktreeParams(action="keep"))
    assert not exit_result.is_error and Path(remote.agent.work_dir).resolve() == tmp_path.resolve()
    assert remote.command_registry.find("worktree") is not None
    await remote._shutdown()
    assert remote.cron_runtime._closed and remote.team_manager.pane_supervisor._closed


@pytest.mark.asyncio
async def test_question_adapter_is_called_before_tool_result_and_cleans_pending(remote):
    async def respond(message):
        if message["type"] == "ask_user":
            remote._handle_ask_response({"id": message["data"]["id"], "answers": {"choice": "正确回答"}})
    remote._broadcast.side_effect = respond
    tool = remote.registry.get("AskUserQuestion")
    result = await asyncio.wait_for(tool.execute(AskUserParams(questions=[
        {"type": "radio", "name": "choice", "message": "选择什么？", "options": ["A", "B"]},
    ])), timeout=1)
    assert "choice: 正确回答" in result.output and tool._pending_event is None
    assert not remote._pending_asks
    await remote._shutdown()


@pytest.mark.asyncio
async def test_noninteractive_question_fails_immediately():
    tool = AskUserTool()
    result = await asyncio.wait_for(tool.execute(AskUserParams(questions=[
        {"type": "text", "name": "answer", "message": "answer?"},
    ])), timeout=1)
    assert result.is_error and tool._pending_event is None


@pytest.mark.asyncio
async def test_question_response_identity_and_schema_are_checked(remote):
    tool = remote.registry.get("AskUserQuestion")
    task = asyncio.create_task(tool.execute(AskUserParams(questions=[{"type": "text", "name": "answer", "message": "answer?"}])))
    await asyncio.sleep(0)
    identity = next(iter(remote._pending_asks))
    remote._handle_ask_response({"id": "foreign", "answers": {"answer": "wrong"}})
    remote._handle_ask_response({"id": identity, "answers": {"foreign": "wrong"}})
    remote._handle_ask_response({"id": identity, "answers": {"answer": []}})
    assert not task.done()
    remote._handle_ask_response({"id": identity, "answers": {"answer": "valid"}})
    assert "valid" in (await task).output
    await remote._shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("choice", ["approve", "feedback", "reject"])
async def test_plan_approval_is_explicit_once_and_restores_prior_mode(remote, choice):
    remote.agent.set_permission_mode(PermissionMode.ACCEPT_EDITS)
    remote.set_plan_mode(True)
    remote.agent._get_plan_path().write_text("# 实施计划\nRead then edit", encoding="utf-8")
    await remote._request_plan_approval()
    identity = remote._pending_plan["id"]
    remote._handle_user_message = AsyncMock()
    await remote._handle_plan_response({"id": "stale", "choice": "approve"})
    remote._handle_user_message.assert_not_awaited()
    await remote._handle_plan_response({"id": identity, "choice": choice, "feedback": "调整计划"})
    assert remote._pending_plan is None
    assert remote.agent.permission_mode == (PermissionMode.ACCEPT_EDITS if choice == "approve" else PermissionMode.PLAN)
    count = remote._handle_user_message.await_count
    await remote._handle_plan_response({"id": identity, "choice": "approve"})
    assert remote._handle_user_message.await_count == count
    if choice == "approve":
        assert "Read then edit" in remote._handle_user_message.call_args.args[0]
    await remote._shutdown()


@pytest.mark.asyncio
async def test_edited_plan_cannot_reuse_old_approval(remote):
    remote.set_plan_mode(True)
    remote.agent._get_plan_path().write_text("original", encoding="utf-8")
    await remote._request_plan_approval()
    identity = remote._pending_plan["id"]
    remote.agent._get_plan_path().write_text("different plan", encoding="utf-8")
    remote._handle_user_message = AsyncMock()
    await remote._handle_plan_response({"id": identity, "choice": "approve"})
    remote._handle_user_message.assert_not_awaited()
    assert remote.agent.plan_mode and remote._pending_plan is None
    await remote._shutdown()


@pytest.mark.asyncio
async def test_team_mailbox_reaches_lead_without_background_completion(remote):
    from valecode.teams.mailbox import create_message
    await remote.registry.get("TeamCreate").execute(TeamCreateParams(team_name="fixture-team"))
    remote._connections.add(MagicMock())
    remote.team_manager.get_mailbox("fixture-team").write(remote.agent.agent_id,
        create_message("fixture-member", remote.agent.agent_id, "team result"))
    remote._handle_user_message = AsyncMock()
    await remote._process_task_notifications()
    assert "team result" in remote._handle_user_message.call_args.args[0]
    await remote._shutdown()


@pytest.mark.asyncio
async def test_plan_gate_does_not_consume_waiting_notifications(remote):
    remote._connections.add(MagicMock())
    remote._pending_plan = {"id": "pending"}
    remote.task_manager.poll_completed = MagicMock()
    await remote._process_task_notifications()
    remote.task_manager.poll_completed.assert_not_called()
    await remote._shutdown()


@pytest.mark.asyncio
async def test_resource_failure_does_not_abandon_next_owner():
    first = AsyncMock(side_effect=RuntimeError("fixture"))
    second = AsyncMock()
    await close_resources([("first", first), ("second", second)])
    second.assert_awaited_once()


@pytest.mark.asyncio
async def test_real_model_stream_ask_and_response_completes_remote_loop(remote):
    from test_teammate_worker import FakeServer
    from valecode.client import create_client
    call = {"tool_calls": [{"index": 0, "id": "question-1", "type": "function", "function": {
        "name": "AskUserQuestion", "arguments": json.dumps({"questions": [{"type": "text", "name": "answer", "message": "choose?"}]})}}]}
    service = FakeServer([[(call, None), ({}, "tool_calls")], [({"content": "finished"}, None), ({}, "stop")]])
    thread = threading.Thread(target=service.serve_forever, daemon=True)
    thread.start()
    remote.agent.client = create_client(ProviderConfig("fixture", "openai-compat", f"http://127.0.0.1:{service.server_port}/v1", "fake", "fixture"))
    observed = []
    async def reply(message):
        observed.append(message["type"])
        if message["type"] == "ask_user":
            remote._handle_ask_response({"id": message["data"]["id"], "answers": {"answer": "fixture answer"}})
    remote._broadcast.side_effect = reply
    try:
        await asyncio.wait_for(remote._handle_user_message("ask first"), timeout=10)
        assert observed.index("ask_user") < observed.index("tool_result")
        assert len(service.requests) == 2
        assert "fixture answer" in json.dumps(service.requests[-1]["messages"])
        assert "loop_complete" in observed and not remote._streaming
    finally:
        await remote._shutdown()
        service.shutdown()
        service.server_close()


@pytest.mark.asyncio
async def test_remote_real_teammate_runs_isolated_and_delivers_result(remote, tmp_path):
    from test_teammate_worker import FakeServer
    from valecode.client import create_client
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "seed.txt").write_text("fixture", encoding="utf-8")
    subprocess.run(["git", "add", "seed.txt"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-m", "fixture"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / ".valecode" / "permissions.yaml").write_text('- rule: SendMessage(*)\n  effect: allow\n', encoding="utf-8")
    call = {"tool_calls": [{"index": 0, "id": "message-1", "type": "function", "function": {
        "name": "SendMessage", "arguments": json.dumps({"to": "lead", "message": "isolated teammate result", "summary": "fixture result"})}}]}
    service = FakeServer([[(call, None), ({}, "tool_calls")], [({"content": "done"}, None), ({}, "stop")]])
    threading.Thread(target=service.serve_forever, daemon=True).start()
    remote.agent.client = create_client(ProviderConfig("fixture", "openai-compat", f"http://127.0.0.1:{service.server_port}/v1", "fake", "fixture"))
    remote.agent_tool._provider_config = ProviderConfig("fixture", "openai-compat", f"http://127.0.0.1:{service.server_port}/v1", "fake", "fixture")
    try:
        await remote.registry.get("TeamCreate").execute(TeamCreateParams(team_name="fixture-team"))
        result = await remote.agent_tool.execute(AgentToolParams(prompt="return result", description="test", subagent_type="Explore", team_name="fixture-team"))
        assert not result.is_error
        member = remote.team_manager.get_team("fixture-team").members[0]
        for _ in range(100):
            if member.progress.status == "idle":
                break
            await asyncio.sleep(0.05)
        assert Path(member.worktree_path).is_relative_to(tmp_path / ".valecode" / "worktrees")
        assert member.progress.status == "idle" and len(service.requests) == 2
        remote._connections.add(MagicMock())
        remote._handle_user_message = AsyncMock()
        await remote._process_task_notifications()
        assert "isolated teammate result" in remote._handle_user_message.call_args.args[0]
        assert member.is_active is False
    finally:
        await remote._shutdown()
        service.shutdown()
        service.server_close()
