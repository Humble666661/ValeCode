from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from valecode.agent import Agent, ToolResultEvent, ErrorEvent, PermissionRequest, PermissionResponse
from valecode.client import LLMClient, NetworkError
from valecode.conversation import ConversationManager
from valecode.hooks import HookEngine
from valecode.memory.session import SessionManager
from valecode.permissions import PermissionMode
from valecode.runtime import RetryPolicy
from valecode.tools import ToolRegistry, ToolSource
from valecode.tools.base import Tool, ToolResult, ToolCallComplete, StreamEnd, TextDelta


class Params(BaseModel):
    value: str


class ProbeRead(Tool):
    name = "ProbeRead"
    description = "Test read ahead"
    params_model = Params
    category = "read"
    is_concurrency_safe = True

    def __init__(self, *, hang=False):
        self.values = []
        self.started = asyncio.Event()
        self.stopped = asyncio.Event()
        self.hang = hang

    async def execute(self, params):
        self.values.append(params.value)
        self.started.set()
        try:
            if self.hang:
                await asyncio.Event().wait()
            return ToolResult(params.value)
        finally:
            self.stopped.set()


class Client(LLMClient):
    def __init__(self, first):
        self.first = first
        self.calls = 0

    async def stream(self, conversation, system="", tools=None):
        self.calls += 1
        if self.calls == 1:
            async for event in self.first():
                yield event
        else:
            yield TextDelta("done")
            yield StreamEnd("end_turn", 1, 1)


def conversation():
    conv = ConversationManager()
    conv.add_user_message("test")
    return conv


def registry_for(tool, source=ToolSource.BUILTIN):
    registry = ToolRegistry()
    registry.register(tool, source=source)
    return registry


@pytest.mark.asyncio
@pytest.mark.parametrize("noninteractive", [False, True])
async def test_read_executes_before_stream_end_once_and_persists(tmp_path, noninteractive):
    tool = ProbeRead()

    async def first():
        yield ToolCallComplete("read", tool.name, {"value": "accepted"})
        await asyncio.wait_for(tool.stopped.wait(), 1)
        assert tool.values == ["accepted"]
        yield TextDelta("continuing output")
        yield StreamEnd("tool_use", 2, 1)

    sessions = SessionManager(str(tmp_path))
    session = sessions.create()
    agent = Agent(Client(first), registry_for(tool), "anthropic", work_dir=str(tmp_path), run_store=sessions.run_store)
    agent.session_id = session.session_id
    try:
        if noninteractive:
            await agent.run_to_completion("read ahead")
        else:
            events = [event async for event in agent.run(conversation())]
            assert [event.output for event in events if isinstance(event, ToolResultEvent)] == ["accepted"]
        assert tool.values == ["accepted"]
        run = sessions.run_store.list_runs(session_id=session.session_id)[0]
        calls = sessions.run_store.list_tool_calls(run.id)
        assert len(calls) == 1 and calls[0].status.value == "completed"
        assert calls[0].result["output"] == "accepted"
    finally:
        session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("condition", ["write", "plugin", "hook", "deny", "ask"])
async def test_guarded_calls_do_not_execute_during_stream(tmp_path, condition):
    tool = ProbeRead()
    if condition == "write":
        tool.category = "write"
    source = ToolSource.PLUGIN if condition == "plugin" else ToolSource.BUILTIN

    async def first():
        yield ToolCallComplete("read", tool.name, {"value": "guarded"})
        await asyncio.sleep(0)
        assert tool.values == []
        yield StreamEnd("tool_use", 1, 1)

    checker = None
    if condition in ("deny", "ask"):
        checker = SimpleNamespace(mode=PermissionMode.DEFAULT, check=lambda *_: SimpleNamespace(effect=condition, reason="test"))
    agent = Agent(Client(first), registry_for(tool, source), "anthropic", work_dir=str(tmp_path),
                  hook_engine=HookEngine() if condition == "hook" else None, permission_checker=checker)
    async for event in agent.run(conversation()):
        if isinstance(event, PermissionRequest):
            event.future.set_result(PermissionResponse.DENY)
    assert tool.values == ([] if condition in ("deny", "ask") else ["guarded"])


@pytest.mark.asyncio
async def test_read_after_write_is_not_reordered(tmp_path):
    read = ProbeRead()
    write = ProbeRead()
    write.name = "WriteProbe"
    write.category = "write"
    registry = registry_for(read)
    registry.register(write, source=ToolSource.BUILTIN)

    async def first():
        yield ToolCallComplete("write", write.name, {"value": "write"})
        yield ToolCallComplete("read", read.name, {"value": "after-write"})
        await asyncio.sleep(0)
        assert read.values == write.values == []
        yield StreamEnd("tool_use", 1, 1)

    await Agent(Client(first), registry, "anthropic", work_dir=str(tmp_path)).run_to_completion("test")
    assert write.values == ["write"] and read.values == ["after-write"]


@pytest.mark.asyncio
async def test_failed_stream_discards_read_result_before_retry(tmp_path):
    tool = ProbeRead()

    class RetryClient(LLMClient):
        calls = 0

        async def stream(self, conv, system="", tools=None):
            self.calls += 1
            if self.calls == 1:
                yield ToolCallComplete("same-id", tool.name, {"value": "discarded"})
                await asyncio.wait_for(tool.stopped.wait(), 1)
                raise NetworkError("disconnected")
            if self.calls == 2:
                yield ToolCallComplete("same-id", tool.name, {"value": "accepted"})
                yield StreamEnd("tool_use", 1, 1)
            else:
                yield TextDelta("done")
                yield StreamEnd("end_turn", 1, 1)

    agent = Agent(RetryClient(), registry_for(tool), "anthropic", work_dir=str(tmp_path),
                  retry_policy=RetryPolicy(base_delay=0, max_delay=0, jitter_ratio=0))
    events = [event async for event in agent.run(conversation())]
    assert [event.output for event in events if isinstance(event, ToolResultEvent)] == ["accepted"]
    assert tool.values == ["discarded", "accepted"]


@pytest.mark.asyncio
async def test_cancelled_stream_awaits_inflight_read_cleanup(tmp_path):
    tool = ProbeRead(hang=True)

    async def first():
        yield ToolCallComplete("read", tool.name, {"value": "blocked"})
        await asyncio.Event().wait()

    agent = Agent(Client(first), registry_for(tool), "anthropic", work_dir=str(tmp_path))

    async def consume():
        return [event async for event in agent.run(conversation())]

    task = asyncio.create_task(consume())
    await asyncio.wait_for(tool.started.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert tool.stopped.is_set()


@pytest.mark.asyncio
async def test_loop_guard_blocks_third_read_before_it_starts(tmp_path):
    tool = ProbeRead()

    class RepeatedClient(LLMClient):
        calls = 0

        async def stream(self, conv, system="", tools=None):
            self.calls += 1
            yield ToolCallComplete(str(self.calls), tool.name, {"value": "repeat"})
            yield StreamEnd("tool_use", 1, 1)

    events = [event async for event in Agent(RepeatedClient(), registry_for(tool), "anthropic", work_dir=str(tmp_path)).run(conversation())]
    assert tool.values == ["repeat", "repeat"]
    assert any(isinstance(event, ErrorEvent) and "Repeated" in event.message for event in events)


@pytest.mark.asyncio
async def test_token_limit_discards_read_ahead_result(tmp_path):
    tool = ProbeRead()

    class LimitedClient(LLMClient):
        calls = 0

        async def stream(self, conv, system="", tools=None):
            self.calls += 1
            if self.calls == 1:
                yield ToolCallComplete("discard", tool.name, {"value": "incomplete"})
                await asyncio.wait_for(tool.stopped.wait(), 1)
                yield StreamEnd("max_tokens", 1, 1)
            elif self.calls == 2:
                yield ToolCallComplete("accept", tool.name, {"value": "complete"})
                yield StreamEnd("tool_use", 1, 1)
            else:
                yield TextDelta("done")
                yield StreamEnd("end_turn", 1, 1)

    events = [event async for event in Agent(LimitedClient(), registry_for(tool), "anthropic", work_dir=str(tmp_path)).run(conversation())]
    assert [event.output for event in events if isinstance(event, ToolResultEvent)] == ["complete"]


@pytest.mark.asyncio
async def test_permission_revocation_before_result_delivery_is_honored(tmp_path):
    tool = ProbeRead()
    state = {"effect": "allow"}
    checker = SimpleNamespace(mode=PermissionMode.DEFAULT,
        check=lambda *_: SimpleNamespace(effect=state["effect"], reason="revoked"))

    async def first():
        yield ToolCallComplete("read", tool.name, {"value": "private result"})
        await asyncio.wait_for(tool.stopped.wait(), 1)
        state["effect"] = "deny"
        yield StreamEnd("tool_use", 1, 1)

    agent = Agent(Client(first), registry_for(tool), "anthropic", work_dir=str(tmp_path), permission_checker=checker)
    events = [event async for event in agent.run(conversation())]
    results = [event for event in events if isinstance(event, ToolResultEvent)]
    assert len(results) == 1 and results[0].is_error
    assert "Permission denied" in results[0].output
