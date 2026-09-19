from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncIterator

import pytest
from pydantic import BaseModel

from valecode.agent import Agent, LoopComplete, PermissionRequest, PermissionResponse
from valecode.client import LLMClient
from valecode.conversation import ConversationManager
from valecode.memory.session import SessionManager
from valecode.persistence import RunStatus, StepStatus, ToolCallStatus
from valecode.permissions import PermissionMode
from valecode.tools import ToolRegistry
from valecode.tools.base import (
    StreamEnd,
    StreamEvent,
    TextDelta,
    Tool,
    ToolCallComplete,
    ToolResult,
)


class ScriptedClient(LLMClient):
    def __init__(self, responses: list[list[StreamEvent]]) -> None:
        self.responses = responses
        self.index = 0

    async def stream(
        self,
        conversation: ConversationManager,
        system: str = "",
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        response = self.responses[self.index]
        self.index += 1
        for event in response:
            yield event


class CancelledClient(LLMClient):
    async def stream(
        self,
        conversation: ConversationManager,
        system: str = "",
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        if False:
            yield TextDelta("unused")
        raise asyncio.CancelledError


class AskPermissionChecker:
    mode = PermissionMode.DEFAULT

    def check(self, tool: Tool, arguments: dict[str, Any]):
        return SimpleNamespace(effect="ask", reason="confirm")


class EchoParams(BaseModel):
    text: str


class EchoTool(Tool):
    name = "Echo"
    description = "Echo text"
    params_model = EchoParams
    category = "read"
    is_concurrency_safe = True

    async def execute(self, params: EchoParams) -> ToolResult:
        return ToolResult(output=params.text)


def make_agent(tmp_path, responses, *, max_iterations: int = 0):
    sessions = SessionManager(str(tmp_path))
    session = sessions.create()
    registry = ToolRegistry()
    registry.register(EchoTool())
    agent = Agent(
        ScriptedClient(responses),
        registry,
        "anthropic",
        work_dir=str(tmp_path),
        max_iterations=max_iterations,
        run_store=sessions.run_store,
        provider_name="test-provider",
        model="test-model",
    )
    agent.session_id = session.session_id
    return agent, sessions, session


@pytest.mark.asyncio
async def test_agent_persists_completed_run_and_step(tmp_path):
    agent, sessions, session = make_agent(
        tmp_path,
        [[TextDelta("done"), StreamEnd("end_turn", 12, 7)]],
    )
    conversation = ConversationManager()
    conversation.add_user_message("hello")

    events = [event async for event in agent.run(conversation)]

    assert any(isinstance(event, LoopComplete) for event in events)
    run = sessions.run_store.list_runs(session_id=session.session_id)[0]
    assert run.status == RunStatus.COMPLETED
    assert run.input == "hello"
    assert run.metadata == {"protocol": "anthropic", "agent_type": "lead"}
    step = sessions.run_store.list_steps(run.id)[0]
    assert step.status == StepStatus.COMPLETED
    assert (step.provider, step.model) == ("test-provider", "test-model")
    assert (step.input_tokens, step.output_tokens) == (12, 7)
    session.close()


@pytest.mark.asyncio
async def test_agent_persists_tool_lifecycle(tmp_path):
    agent, sessions, session = make_agent(
        tmp_path,
        [
            [
                ToolCallComplete("provider-1", "Echo", {"text": "hello"}),
                StreamEnd("tool_use", 10, 3),
            ],
            [TextDelta("finished"), StreamEnd("end_turn", 15, 4)],
        ],
    )
    conversation = ConversationManager()
    conversation.add_user_message("echo hello")

    _ = [event async for event in agent.run(conversation)]

    run = sessions.run_store.list_runs(session_id=session.session_id)[0]
    steps = sessions.run_store.list_steps(run.id)
    calls = sessions.run_store.list_tool_calls(run.id)
    assert run.status == RunStatus.COMPLETED
    assert [step.status for step in steps] == [
        StepStatus.COMPLETED,
        StepStatus.COMPLETED,
    ]
    assert len(calls) == 1
    call = calls[0]
    assert call.status == ToolCallStatus.COMPLETED
    assert call.arguments == {"text": "hello"}
    assert call.result == {"output": "hello", "is_error": False}
    assert call.idempotency_key.startswith("tool:")
    assert call.metadata["provider_tool_call_id"] == "provider-1"
    assert call.side_effect_class == "read"
    event_types = [event.event_type for event in sessions.run_store.events.list(run_id=run.id)]
    assert event_types.count("tool_call.status_changed") == 2
    session.close()


@pytest.mark.asyncio
async def test_large_tool_result_is_scoped_indexed_and_recoverable(tmp_path):
    large = "x" * 50_100
    agent, sessions, session = make_agent(
        tmp_path,
        [
            [
                ToolCallComplete("provider-large", "Echo", {"text": large}),
                StreamEnd("tool_use", 10, 3),
            ],
            [TextDelta("finished"), StreamEnd("end_turn", 15, 4)],
        ],
    )
    conversation = ConversationManager()
    conversation.add_user_message("produce a large result")

    _ = [event async for event in agent.run(conversation)]

    run = sessions.run_store.list_runs(session_id=session.session_id)[0]
    call = sessions.run_store.list_tool_calls(run.id)[0]
    assert call.result_path is not None
    result_path = Path(call.result_path)
    assert result_path.read_text(encoding="utf-8") == large
    assert result_path.parent.parent.parent == agent.session_dir
    artifact = sessions.result_artifact_store.get_by_path(result_path)
    assert artifact is not None
    assert artifact.session_id == session.session_id
    assert artifact.run_id == run.id
    assert artifact.tool_call_id == call.id
    assert artifact.size_bytes == len(large)
    session.close()


@pytest.mark.asyncio
async def test_agent_persists_interrupted_run(tmp_path):
    agent, sessions, session = make_agent(
        tmp_path,
        [
            [
                ToolCallComplete("provider-1", "Echo", {"text": "hello"}),
                StreamEnd("tool_use", 1, 1),
            ],
            [TextDelta("unused"), StreamEnd("end_turn", 1, 1)],
        ],
        max_iterations=1,
    )
    conversation = ConversationManager()
    conversation.add_user_message("stop early")

    _ = [event async for event in agent.run(conversation)]

    run = sessions.run_store.list_runs(session_id=session.session_id)[0]
    assert run.status == RunStatus.INTERRUPTED
    assert [step.status for step in sessions.run_store.list_steps(run.id)] == [
        StepStatus.COMPLETED
    ]
    session.close()


@pytest.mark.asyncio
async def test_permission_wait_is_persisted_as_blocked(tmp_path):
    agent, sessions, session = make_agent(
        tmp_path,
        [
            [
                ToolCallComplete("provider-1", "Echo", {"text": "allowed"}),
                StreamEnd("tool_use", 2, 1),
            ],
            [TextDelta("done"), StreamEnd("end_turn", 3, 1)],
        ],
    )
    agent.permission_checker = AskPermissionChecker()  # type: ignore[assignment]
    conversation = ConversationManager()
    conversation.add_user_message("ask first")

    events = agent.run(conversation)
    while True:
        event = await anext(events)
        if isinstance(event, PermissionRequest):
            run = sessions.run_store.list_runs(session_id=session.session_id)[0]
            step = sessions.run_store.list_steps(run.id)[0]
            assert run.status == RunStatus.BLOCKED
            assert step.status == StepStatus.BLOCKED
            event.future.set_result(PermissionResponse.ALLOW)
            break
    _ = [event async for event in events]

    run = sessions.run_store.list_runs(session_id=session.session_id)[0]
    assert run.status == RunStatus.COMPLETED
    assert sessions.run_store.list_tool_calls(run.id)[0].status == ToolCallStatus.COMPLETED
    session.close()


@pytest.mark.asyncio
async def test_cancelled_llm_call_persists_cancelled_state(tmp_path):
    sessions = SessionManager(str(tmp_path))
    session = sessions.create()
    agent = Agent(
        CancelledClient(),
        ToolRegistry(),
        "anthropic",
        work_dir=str(tmp_path),
        run_store=sessions.run_store,
    )
    agent.session_id = session.session_id
    conversation = ConversationManager()
    conversation.add_user_message("cancel")

    with pytest.raises(asyncio.CancelledError):
        _ = [event async for event in agent.run(conversation)]

    run = sessions.run_store.list_runs(session_id=session.session_id)[0]
    assert run.status == RunStatus.CANCELLED
    assert sessions.run_store.list_steps(run.id)[0].status == StepStatus.CANCELLED
    session.close()
