"""Session task progress: validation, persistence, prompt visibility, and UI."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncIterator
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from valecode.agent import Agent
from valecode.client import LLMClient
from valecode.commands.handlers.todos import handle_todos
from valecode.commands.registry import CommandContext
from valecode.conversation import ConversationManager
from valecode.permissions import (
    DangerousCommandDetector, PathSandbox, PermissionChecker, PermissionMode, RuleEngine,
)
from valecode.tools import ToolRegistry
from valecode.tools.base import StreamEnd, StreamEvent, TextDelta, ToolCallComplete
from valecode.tools.todo_write import TodoStore, TodoWrite, TodoWriteParams


def _params(*items: tuple[str, str]) -> TodoWriteParams:
    return TodoWriteParams.model_validate({
        "todos": [{"content": content, "status": status} for content, status in items]
    })


@pytest.mark.asyncio
async def test_todo_progress_is_session_scoped_and_survives_restart(tmp_path: Path) -> None:
    tool = TodoWrite(lambda: (tmp_path, "session-one"))
    result = await tool.execute(_params(
        ("实现接口", "completed"), ("运行测试", "in_progress"),
    ))
    assert not result.is_error
    assert "1/2 已完成" in result.output
    assert "运行测试" in TodoWrite(lambda: (tmp_path, "session-one")).current_summary()
    assert TodoWrite(lambda: (tmp_path, "session-two")).current_summary() == ""
    assert TodoStore(tmp_path, "session-one").path.is_file()

    await tool.execute(_params(("实现接口", "completed"), ("运行测试", "completed")))
    assert "2/2 已完成" in tool.current_summary()
    await tool.execute(_params())
    assert tool.current_summary() == ""


def test_todo_validation_and_path_guards(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="Only one task"):
        _params(("one", "in_progress"), ("two", "in_progress"))
    with pytest.raises(ValidationError, match="unique"):
        _params(("one", "pending"), (" ONE ", "completed"))
    with pytest.raises(ValueError, match="session ID"):
        TodoStore(tmp_path, "../outside")
    with pytest.raises(ValidationError):
        TodoWriteParams.model_validate({"todos": [
            {"content": str(i), "status": "pending"} for i in range(31)
        ]})


def test_builtin_todo_updates_do_not_prompt_for_permission(tmp_path: Path) -> None:
    checker = PermissionChecker(
        detector=DangerousCommandDetector(), sandbox=PathSandbox(tmp_path),
        rule_engine=RuleEngine(), mode=PermissionMode.DEFAULT,
    )
    tool = TodoWrite(lambda: (tmp_path, "session-one"))
    assert checker.check(tool, {"todos": []}).effect == "allow"
    checker.mode = PermissionMode.PLAN
    assert checker.check(tool, {"todos": []}).effect == "allow"
    rules = tmp_path / "permissions.yaml"
    rules.write_text(
        "- rule: TodoWrite(*)\n  effect: deny\n", encoding="utf-8",
    )
    checker.rule_engine = RuleEngine(user_rules_path=rules)
    assert checker.check(tool, {"todos": []}).effect == "deny"


class TodoClient(LLMClient):
    def __init__(self) -> None:
        self.systems: list[str] = []

    async def stream(
        self, conversation: ConversationManager, system: str = "",
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        self.systems.append(system)
        if len(self.systems) == 1:
            yield ToolCallComplete("todo-1", "TodoWrite", {"todos": [
                {"content": "补齐测试", "status": "in_progress"},
                {"content": "提交修改", "status": "pending"},
            ]})
            yield StreamEnd("tool_use", 5, 2)
        else:
            yield TextDelta("继续实现")
            yield StreamEnd("end_turn", 5, 2)


@pytest.mark.asyncio
async def test_agent_sees_updated_todos_on_next_model_turn(tmp_path: Path) -> None:
    registry = ToolRegistry()
    client = TodoClient()
    agent = Agent(client, registry, "anthropic", work_dir=str(tmp_path))
    agent.session_id = "session-one"
    tool = TodoWrite(lambda: (agent.work_dir, agent.session_id))
    registry.register(tool)
    agent.set_todo_state_provider(tool.current_summary)
    conversation = ConversationManager()
    conversation.add_user_message("完成两步任务")

    async for _ in agent.run(conversation):
        pass

    assert len(client.systems) == 2
    assert "补齐测试" not in client.systems[0]
    assert "补齐测试" in client.systems[1]
    assert "0/2 已完成" in client.systems[1]


@pytest.mark.asyncio
async def test_todos_slash_command_shows_persisted_progress(tmp_path: Path) -> None:
    registry = ToolRegistry()
    tool = TodoWrite(lambda: (tmp_path, "session-one"))
    registry.register(tool)
    await tool.execute(_params(("补测试", "in_progress")))
    ui = MagicMock()
    context = CommandContext(
        args="", agent=SimpleNamespace(registry=registry), conversation=None,
        session=None, session_manager=None, memory_manager=None, ui=ui, config={},
    )

    await handle_todos(context)

    assert "补测试" in ui.add_system_message.call_args.args[0]


@pytest.mark.asyncio
async def test_tui_todo_tool_follows_session_switch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from valecode.app import ValeCodeApp
    from valecode.config import ProviderConfig

    monkeypatch.chdir(tmp_path)
    provider = ProviderConfig(
        name="offline", protocol="anthropic", base_url="https://example.invalid",
        model="offline", api_key="not-used",
    )
    with patch("valecode.app.create_client", return_value=TodoClient()):
        app = ValeCodeApp([provider])
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            tool = app.registry.get("TodoWrite")
            assert isinstance(tool, TodoWrite)
            await tool.execute(_params(("第一会话的任务", "in_progress")))
            assert "第一会话的任务" in app.agent._system_with_todo_progress("base")
            app._set_session(app.session_manager.create())
            assert tool.current_summary() == ""


@pytest.mark.asyncio
async def test_prompt_mode_registers_todo_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from valecode.__main__ import _run_prompt
    from valecode.config import ProviderConfig

    monkeypatch.chdir(tmp_path)
    provider = ProviderConfig(
        name="offline", protocol="anthropic", base_url="https://example.invalid",
        model="offline", api_key="not-used",
    )
    config = SimpleNamespace(
        providers=[provider], sandbox=SimpleNamespace(enabled=False),
        worktree=None, enable_verification_agent=False,
        enable_fork=False, teammate_mode="", enable_coordinator_mode=False,
    )
    client = TodoClient()

    async def no_resolve(_provider: ProviderConfig) -> None:
        return None

    with (
        patch("valecode.client.create_client", return_value=client),
        patch("valecode.client.resolve_context_window", no_resolve),
    ):
        await _run_prompt(config, PermissionMode.DEFAULT, None, "完成两步任务")

    assert len(client.systems) == 2
    assert "补齐测试" in client.systems[1]


@pytest.mark.asyncio
async def test_remote_mode_registers_todo_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from valecode.config import ProviderConfig
    from valecode.remote import RemoteServer

    monkeypatch.chdir(tmp_path)
    provider = ProviderConfig(
        name="offline", protocol="anthropic", base_url="https://example.invalid",
        model="offline", api_key="not-used",
    )
    with patch("valecode.remote.create_client", return_value=TodoClient()):
        server = RemoteServer([provider])
        server._init_agent()
        tool = server.registry.get("TodoWrite")
        assert isinstance(tool, TodoWrite)
        await tool.execute(_params(("远程会话任务", "in_progress")))
        assert "远程会话任务" in server.agent._system_with_todo_progress("base")
        await server._shutdown()
