"""Integration coverage for lifecycle hook events emitted by Agent."""

from __future__ import annotations

from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from valecode.agent import Agent, PermissionRequest, PermissionResponse
from valecode.client import LLMClient
from valecode.context import CompactEvent
from valecode.conversation import ConversationManager
from valecode.hooks import Action, Hook, HookEngine
from valecode.permissions import (
    DangerousCommandDetector,
    PathSandbox,
    PermissionChecker,
    PermissionMode,
    RuleEngine,
)
from valecode.tools import ToolRegistry
from valecode.tools.base import (
    StreamEvent,
    Tool,
    ToolCallComplete,
    ToolResult,
)


class _WriteParams(BaseModel):
    file_path: str
    content: str = ""


class _WriteTool(Tool):
    name = "WriteFile"
    description = "write test file"
    params_model = _WriteParams
    category = "write"

    async def execute(self, params: _WriteParams) -> ToolResult:
        return ToolResult(output=f"wrote {params.file_path}")


class _BashParams(BaseModel):
    command: str


class _BashTool(Tool):
    name = "Bash"
    description = "run test command"
    params_model = _BashParams
    category = "command"

    async def execute(self, params: _BashParams) -> ToolResult:
        return ToolResult(output=f"ran {params.command}")


class _FailingClient(LLMClient):
    async def stream(
        self,
        conversation: ConversationManager,
        system: str = "",
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        raise RuntimeError("provider exploded")
        yield  # pragma: no cover - keeps this an async generator


def _engine(*events: str) -> HookEngine:
    return HookEngine([
        Hook(
            id=f"observe-{event}",
            event=event,
            action=Action(
                type="prompt",
                message=(
                    "$EVENT|$TOOL_NAME|$FILE_PATH|$MESSAGE|$ERROR"
                ),
            ),
        )
        for event in events
    ])


@pytest.mark.asyncio
async def test_permission_request_hook_runs_before_user_decision(tmp_path) -> None:
    engine = _engine("permission_request", "file_change")
    registry = ToolRegistry()
    registry.register(_WriteTool())
    checker = PermissionChecker(
        detector=DangerousCommandDetector(),
        sandbox=PathSandbox(str(tmp_path)),
        rule_engine=RuleEngine(),
        mode=PermissionMode.DEFAULT,
    )
    agent = Agent(
        _FailingClient(), registry, "anthropic",
        work_dir=str(tmp_path), permission_checker=checker, hook_engine=engine,
    )
    call = ToolCallComplete(
        "tool-1", "WriteFile", {"file_path": "notes.txt", "content": "x"}
    )

    async for item in agent._execute_tool_untraced(call):
        if isinstance(item, PermissionRequest):
            item.future.set_result(PermissionResponse.DENY)

    notifications = engine.drain_notifications()
    assert [note.event for note in notifications] == ["permission_request"]
    assert "WriteFile|notes.txt" in notifications[0].output


@pytest.mark.asyncio
async def test_file_and_command_hooks_only_follow_actual_execution(tmp_path) -> None:
    engine = _engine("file_change", "command_execute")
    registry = ToolRegistry()
    registry.register(_WriteTool())
    registry.register(_BashTool())
    agent = Agent(
        _FailingClient(), registry, "anthropic",
        work_dir=str(tmp_path), hook_engine=engine,
    )

    write_result = await agent._execute_tool_noninteractive(
        ToolCallComplete(
            "write-1", "WriteFile", {"file_path": "notes.txt", "content": "x"}
        )
    )
    command_result = await agent._execute_tool_noninteractive(
        ToolCallComplete("bash-1", "Bash", {"command": "echo ok"})
    )

    assert not write_result.is_error
    assert not command_result.is_error
    notifications = engine.drain_notifications()
    assert [note.event for note in notifications] == [
        "file_change", "command_execute",
    ]
    assert "WriteFile|notes.txt|wrote notes.txt" in notifications[0].output
    assert "Bash||ran echo ok" in notifications[1].output


@pytest.mark.asyncio
async def test_compact_hook_receives_compaction_metadata(tmp_path) -> None:
    engine = _engine("compact")
    agent = Agent(
        _FailingClient(), ToolRegistry(), "anthropic",
        work_dir=str(tmp_path), hook_engine=engine,
    )
    conversation = ConversationManager()
    conversation.add_user_message("compact me")

    with patch("valecode.agent.auto_compact", new=AsyncMock(
        return_value=CompactEvent(before_tokens=1234)
    )):
        result = await agent._auto_compact_with_trace(conversation)

    assert isinstance(result, CompactEvent)
    notifications = engine.drain_notifications()
    assert [note.event for note in notifications] == ["compact"]
    assert "Compacted context from 1234 tokens" in notifications[0].output


@pytest.mark.asyncio
async def test_error_hook_preserves_original_agent_exception(tmp_path) -> None:
    engine = _engine("error")
    agent = Agent(
        _FailingClient(), ToolRegistry(), "anthropic",
        work_dir=str(tmp_path), hook_engine=engine,
    )
    conversation = ConversationManager()
    conversation.add_user_message("fail")

    with pytest.raises(RuntimeError, match="provider exploded"):
        async for _event in agent.run(conversation):
            pass

    notifications = engine.drain_notifications()
    assert [note.event for note in notifications] == ["error"]
    assert "RuntimeError: provider exploded" in notifications[0].output
