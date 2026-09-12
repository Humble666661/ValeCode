from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

import pytest

from valecode.agent import Agent, PermissionRequest, PermissionResponse, StreamText
from valecode.client import LLMClient
from valecode.conversation import ConversationManager
from valecode.memory.session import SessionManager
from valecode.runtime import RuntimeEvent
from valecode.tools import ToolRegistry
from valecode.tools.base import StreamEnd, StreamEvent, TextDelta


class OneShotClient(LLMClient):
    async def stream(
        self,
        conversation: ConversationManager,
        system: str = "",
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        yield TextDelta("hello")
        yield StreamEnd("end_turn", input_tokens=4, output_tokens=2)


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
