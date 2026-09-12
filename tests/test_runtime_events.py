from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

import pytest

from types import SimpleNamespace

from pydantic import BaseModel

from valecode.agent import (
    Agent,
    MailboxEvent,
    PermissionDecisionEvent,
    PermissionRequest,
    PermissionResponse,
    StreamText,
)
from valecode.client import LLMClient
from valecode.conversation import ConversationManager
from valecode.memory.session import SessionManager
from valecode.runtime import RuntimeEvent
from valecode.teams.mailbox import create_message
from valecode.tools import ToolRegistry
from valecode.tools.base import (
    StreamEnd,
    StreamEvent,
    TextDelta,
    Tool,
    ToolCallComplete,
    ToolResult,
)
from valecode.permissions import PermissionMode


class OneShotClient(LLMClient):
    async def stream(
        self,
        conversation: ConversationManager,
        system: str = "",
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        yield TextDelta("hello")
        yield StreamEnd("end_turn", input_tokens=4, output_tokens=2)


class ScriptedClient(LLMClient):
    def __init__(self, responses: list[list[StreamEvent]]) -> None:
        self.responses = responses

    async def stream(
        self,
        conversation: ConversationManager,
        system: str = "",
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        for event in self.responses.pop(0):
            yield event


class EmptyParams(BaseModel):
    pass


class ConfirmTool(Tool):
    name = "Confirm"
    description = "requires approval"
    params_model = EmptyParams

    async def execute(self, params: EmptyParams) -> ToolResult:
        return ToolResult("approved")


class AskChecker:
    mode = PermissionMode.DEFAULT

    def check(self, tool: Tool, arguments: dict[str, Any]):
        return SimpleNamespace(effect="ask", reason="confirm")


@pytest.mark.asyncio
async def test_agent_events_share_ordered_envelope_and_persistence(tmp_path) -> None:
    sessions = SessionManager(str(tmp_path))
    session = sessions.create()
    agent = Agent(
        OneShotClient(),
        ToolRegistry(),
        "anthropic",
        work_dir=str(tmp_path),
        run_store=sessions.run_store,
        provider_name="test",
        model="model",
    )
    agent.session_id = session.session_id
    conversation = ConversationManager()
    conversation.add_user_message("say hello")

    events = [event async for event in agent.run(conversation)]

    assert all(isinstance(event, RuntimeEvent) for event in events)
    assert [event.sequence for event in events] == list(range(1, len(events) + 1))
    assert len({event.event_id for event in events}) == len(events)
    assert all(event.session_id == session.session_id for event in events)
    assert all(event.run_id for event in events)
    assert all(event.trace_id for event in events)
    assert all(event.emitted_at for event in events)

    run_id = events[0].run_id
    persisted = [
        event
        for event in sessions.run_store.events.list(run_id=run_id)
        if event.event_type.startswith("runtime.")
    ]
    assert [event.payload["event_id"] for event in persisted] == [
        event.event_id for event in events
    ]
    assert [event.payload["sequence"] for event in persisted] == list(
        range(1, len(events) + 1)
    )
    assert persisted[0].event_type == "runtime.stream.text"
    assert persisted[0].payload["payload"] == {"text": "hello"}
    session.close()


def test_permission_envelope_omits_process_local_future() -> None:
    loop = asyncio.new_event_loop()
    try:
        future = loop.create_future()
        event = PermissionRequest(
            tool_name="WriteFile",
            description="confirm",
            future=future,
        )
        event.stamp(
            sequence=3,
            session_id="session-1",
            run_id="run-1",
            step_id="step-1",
            tool_call_id="tool-1",
            trace_id="trace-1",
        )

        envelope = event.to_envelope()
        assert envelope.event_type == "permission.requested"
        assert envelope.sequence == 3
        assert envelope.payload == {
            "tool_name": "WriteFile",
            "description": "confirm",
        }
    finally:
        loop.close()


def test_legacy_event_type_remains_directly_consumable() -> None:
    event = StreamText(text="delta")
    assert event.text == "delta"
    assert isinstance(event, RuntimeEvent)


@pytest.mark.asyncio
async def test_permission_request_and_response_share_event_stream(tmp_path) -> None:
    sessions = SessionManager(str(tmp_path))
    session = sessions.create()
    registry = ToolRegistry()
    registry.register(ConfirmTool())
    agent = Agent(
        ScriptedClient(
            [
                [
                    ToolCallComplete("call-1", "Confirm", {}),
                    StreamEnd("tool_use", 2, 1),
                ],
                [TextDelta("done"), StreamEnd("end_turn", 3, 1)],
            ]
        ),
        registry,
        "anthropic",
        work_dir=str(tmp_path),
        permission_checker=AskChecker(),
        run_store=sessions.run_store,
    )
    agent.session_id = session.session_id
    conversation = ConversationManager()
    conversation.add_user_message("confirm")

    events: list[RuntimeEvent] = []
    async for event in agent.run(conversation):
        events.append(event)
        if isinstance(event, PermissionRequest):
            event.future.set_result(PermissionResponse.ALLOW)

    request = next(event for event in events if isinstance(event, PermissionRequest))
    response = next(
        event for event in events if isinstance(event, PermissionDecisionEvent)
    )
    assert response.sequence == request.sequence + 1
    assert response.response == "allow"
    persisted_types = {
        event.event_type
        for event in sessions.run_store.events.list(run_id=request.run_id)
    }
    assert "runtime.permission.requested" in persisted_types
    assert "runtime.permission.responded" in persisted_types
    session.close()


@pytest.mark.asyncio
async def test_mailbox_message_is_emitted_and_injected(tmp_path) -> None:
    message = create_message("worker", "lead", "result", summary="done")

    class Inbox:
        def consume(self, agent_id: str):
            assert agent_id == "lead"
            return [message]

    manager = SimpleNamespace(get_mailbox=lambda team_name: Inbox())
    agent = Agent(
        OneShotClient(),
        ToolRegistry(),
        "anthropic",
        work_dir=str(tmp_path),
    )
    agent.agent_id = "lead"
    agent.team_name = "team"
    agent._team_manager = manager
    conversation = ConversationManager()
    conversation.add_user_message("start")

    events = [event async for event in agent.run(conversation)]

    mailbox_event = next(event for event in events if isinstance(event, MailboxEvent))
    assert mailbox_event.message_id == message.id
    assert mailbox_event.content == "result"
    assert any(
        item.content == "[Message from worker] result"
        for item in conversation.history
    )
