"""Concurrent-safe tools must not bypass permission prompts or tool hooks."""

from __future__ import annotations

from typing import Any, AsyncIterator

import pytest
from pydantic import BaseModel

from valecode.agent import Agent, PermissionRequest, PermissionResponse, ToolResultEvent
from valecode.client import LLMClient
from valecode.conversation import ConversationManager
from valecode.hooks import HookEngine
from valecode.hooks.conditions import parse_condition
from valecode.hooks.models import Action, Hook
from valecode.permissions import (
    DangerousCommandDetector, PathSandbox, PermissionChecker, PermissionMode, RuleEngine,
)
from valecode.tools import ToolRegistry
from valecode.tools.base import StreamEnd, StreamEvent, TextDelta, Tool, ToolCallComplete, ToolResult


class Params(BaseModel):
    value: str


class ConcurrentWrite(Tool):
    name = "ConcurrentWrite"
    description = "Test guarded batch execution"
    params_model = Params
    category = "write"
    is_concurrency_safe = True

    def __init__(self) -> None:
        self.executed: list[str] = []

    async def execute(self, params: Params) -> ToolResult:
        self.executed.append(params.value)
        return ToolResult(output=params.value)


class BatchClient(LLMClient):
    def __init__(self) -> None:
        self.calls = 0

    async def stream(
        self, conversation: ConversationManager, system: str = "",
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        self.calls += 1
        if self.calls == 1:
            yield ToolCallComplete("first", "ConcurrentWrite", {"value": "blocked"})
            yield ToolCallComplete("second", "ConcurrentWrite", {"value": "allowed"})
            yield StreamEnd("tool_use", 5, 2)
        else:
            yield TextDelta("done")
            yield StreamEnd("end_turn", 5, 2)


@pytest.mark.asyncio
async def test_parallel_safe_tools_still_require_permission() -> None:
    tool = ConcurrentWrite()
    registry = ToolRegistry()
    registry.register(tool)
    checker = PermissionChecker(
        detector=DangerousCommandDetector(), sandbox=PathSandbox("."),
        rule_engine=RuleEngine(), mode=PermissionMode.DEFAULT,
    )
    agent = Agent(BatchClient(), registry, "anthropic", permission_checker=checker)
    conversation = ConversationManager()
    conversation.add_user_message("run two writes")
    requests: list[PermissionRequest] = []
    results: list[ToolResultEvent] = []

    async for event in agent.run(conversation):
        if isinstance(event, PermissionRequest):
            requests.append(event)
            event.future.set_result(PermissionResponse.DENY)
        elif isinstance(event, ToolResultEvent):
            results.append(event)

    assert len(requests) == 2
    assert tool.executed == []
    assert len(results) == 2
    assert all(result.is_error and "Permission denied" in result.output for result in results)


@pytest.mark.asyncio
async def test_parallel_safe_tools_still_run_pre_and_post_hooks() -> None:
    tool = ConcurrentWrite()
    registry = ToolRegistry()
    registry.register(tool)
    pre = Hook(
        id="reject-blocked", event="pre_tool_use",
        action=Action(type="prompt", message="blocked by hook"),
        condition=parse_condition('args.value == "blocked"'), reject=True,
    )
    post = Hook(
        id="record-post", event="post_tool_use",
        action=Action(type="prompt", message="post hook ran"),
    )
    engine = HookEngine([pre, post])
    agent = Agent(BatchClient(), registry, "anthropic", hook_engine=engine)
    conversation = ConversationManager()
    conversation.add_user_message("run two calls")
    results: list[ToolResultEvent] = []

    async for event in agent.run(conversation):
        if isinstance(event, ToolResultEvent):
            results.append(event)

    assert tool.executed == ["allowed"]
    assert len(results) == 2
    assert results[0].is_error and "Hook rejected" in results[0].output
    assert not results[1].is_error
    assert pre.executed and post.executed
