from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from typing import Any, AsyncIterator

import pytest
from pydantic import BaseModel

from valecode.agent import Agent, ErrorEvent, LoopComplete, RetryEvent, ToolResultEvent
from valecode.client import (
    AuthenticationError,
    LLMClient,
    NetworkError,
    OverloadedError,
    RateLimitError,
    ServerError,
)
from valecode.conversation import ConversationManager
from valecode.memory.session import SessionManager
from valecode.observability import TracingConfig, configure_tracing
from valecode.runtime import LoopGuard, RetryPolicy, parse_retry_after
from valecode.tools import ToolRegistry
from valecode.tools.base import StreamEnd, StreamEvent, TextDelta, Tool, ToolCallComplete, ToolResult


class FlakyClient(LLMClient):
    def __init__(self, attempts: list[BaseException | list[StreamEvent]]) -> None:
        self.attempts = list(attempts)
        self.calls = 0

    async def stream(
        self,
        conversation: ConversationManager,
        system: str = "",
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        self.calls += 1
        attempt = self.attempts.pop(0)
        if isinstance(attempt, BaseException):
            raise attempt
        for event in attempt:
            yield event


class EchoParams(BaseModel):
    value: str


class CountingEchoTool(Tool):
    name = "Echo"
    description = "Echo a value"
    params_model = EchoParams
    category = "read"

    def __init__(self) -> None:
        self.calls = 0

    async def execute(self, params: EchoParams) -> ToolResult:
        self.calls += 1
        return ToolResult(output=params.value)


def test_parse_retry_after_seconds_and_http_date() -> None:
    now = datetime(2026, 9, 11, tzinfo=UTC)
    assert parse_retry_after("2.5", now=now) == 2.5
    assert parse_retry_after(format_datetime(now + timedelta(seconds=7)), now=now) == 7
    assert parse_retry_after("invalid", now=now) is None


def test_retry_policy_classifies_and_caps_retries() -> None:
    policy = RetryPolicy(
        max_retries=2,
        base_delay=2,
        max_delay=10,
        max_total_wait=6,
        jitter_ratio=0.25,
    )
    first = policy.decide(
        NetworkError("offline"), retries_used=0, total_wait=0, random_value=1.0
    )
    assert first.retry and first.delay == 2.5
    assert policy.decide(
        RateLimitError("slow", retry_after=4), retries_used=1, total_wait=2
    ).delay == 4
    assert not policy.decide(
        ServerError("down", status_code=503), retries_used=2, total_wait=0
    ).retry
    assert not policy.decide(
        AuthenticationError("bad key"), retries_used=0, total_wait=0
    ).retry
    assert policy.decide(
        OverloadedError("busy", status_code=529), retries_used=0, total_wait=0
    ).retry


def test_loop_guard_uses_canonical_arguments() -> None:
    guard = LoopGuard(repeat_limit=3)
    calls = [
        ToolCallComplete("1", "Echo", {"value": "x", "other": 1}),
        ToolCallComplete("2", "Echo", {"other": 1, "value": "x"}),
        ToolCallComplete("3", "Echo", {"value": "x", "other": 1}),
    ]
    assert not guard.observe(calls[0]).blocked
    assert not guard.observe(calls[1]).blocked
    decision = guard.observe(calls[2])
    assert decision.blocked
    assert decision.repeat_count == 3


def test_retry_and_loop_settings_load_from_project_env(tmp_path, monkeypatch) -> None:
    keys = [
        "VALECODE_LLM_MAX_RETRIES",
        "VALECODE_LLM_RETRY_BASE_DELAY",
        "VALECODE_LLM_RETRY_MAX_DELAY",
        "VALECODE_LLM_RETRY_MAX_WAIT",
        "VALECODE_LLM_RETRY_JITTER",
        "VALECODE_LOOP_REPEAT_LIMIT",
    ]
    for key in keys:
        monkeypatch.delenv(key, raising=False)
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "VALECODE_LLM_MAX_RETRIES=5",
                "VALECODE_LLM_RETRY_BASE_DELAY=1.5",
                "VALECODE_LLM_RETRY_MAX_DELAY=12",
                "VALECODE_LLM_RETRY_MAX_WAIT=25",
                "VALECODE_LLM_RETRY_JITTER=0.1",
                "VALECODE_LOOP_REPEAT_LIMIT=4",
            ]
        ),
        encoding="utf-8",
    )

    policy = RetryPolicy.from_environment(tmp_path)
    guard = LoopGuard.from_environment(tmp_path)
    assert policy.max_retries == 5
    assert policy.base_delay == 1.5
    assert policy.max_delay == 12
    assert policy.max_total_wait == 25
    assert policy.jitter_ratio == 0.1
    assert guard.repeat_limit == 4


@pytest.mark.asyncio
async def test_agent_retries_transient_provider_errors_and_records_events(tmp_path) -> None:
    client = FlakyClient(
        [
            RateLimitError("limited", retry_after=0),
            NetworkError("disconnected"),
            [TextDelta("ok"), StreamEnd("end_turn", input_tokens=4, output_tokens=2)],
        ]
    )
    trace_file = tmp_path / "retry-traces.jsonl"
    tracing = configure_tracing(
        TracingConfig(enabled=True, exporter="file", file_path=str(trace_file))
    )
    sessions = SessionManager(str(tmp_path))
    session = sessions.create()
    agent = Agent(
        client,
        ToolRegistry(),
        "anthropic",
        work_dir=str(tmp_path),
        run_store=sessions.run_store,
        retry_policy=RetryPolicy(
            max_retries=3,
            base_delay=0,
            max_delay=0,
            max_total_wait=0,
            jitter_ratio=0,
        ),
        tracing=tracing,
    )
    agent.session_id = session.session_id
    conversation = ConversationManager()
    conversation.add_user_message("hello")

    try:
        events = [event async for event in agent.run(conversation)]
        assert tracing.force_flush()
    finally:
        tracing.shutdown()
        configure_tracing(TracingConfig())
    assert client.calls == 3
    assert len([event for event in events if isinstance(event, RetryEvent)]) == 2
    assert any(isinstance(event, LoopComplete) for event in events)
    persisted = sessions.run_store.events.list(run_id=agent._current_run_id)
    scheduled = [event for event in persisted if event.event_type == "llm.retry_scheduled"]
    assert [event.payload["category"] for event in scheduled] == [
        "rate_limit",
        "network",
    ]
    spans = [
        json.loads(line)
        for line in trace_file.read_text(encoding="utf-8").splitlines()
    ]
    assert len([span for span in spans if span["name"] == "llm.retry"]) == 2
    failed_streams = [
        span
        for span in spans
        if span["name"] == "llm.stream" and span["status"] == "ERROR"
    ]
    assert len(failed_streams) == 2
    assert failed_streams[0]["attributes"]["retry.category"] == "rate_limit"


@pytest.mark.asyncio
async def test_agent_stops_repeated_identical_tool_calls(tmp_path) -> None:
    repeated = [
        [
            ToolCallComplete(str(index), "Echo", {"value": "same"}),
            StreamEnd("end_turn", input_tokens=1, output_tokens=1),
        ]
        for index in range(1, 4)
    ]
    trace_file = tmp_path / "loop-traces.jsonl"
    tracing = configure_tracing(
        TracingConfig(enabled=True, exporter="file", file_path=str(trace_file))
    )
    tool = CountingEchoTool()
    registry = ToolRegistry()
    registry.register(tool)
    sessions = SessionManager(str(tmp_path))
    session = sessions.create()
    agent = Agent(
        FlakyClient(repeated),
        registry,
        "anthropic",
        work_dir=str(tmp_path),
        run_store=sessions.run_store,
        loop_guard=LoopGuard(repeat_limit=3),
        tracing=tracing,
    )
    agent.session_id = session.session_id
    conversation = ConversationManager()
    conversation.add_user_message("repeat")

    try:
        events = [event async for event in agent.run(conversation)]
        assert tracing.force_flush()
    finally:
        tracing.shutdown()
        configure_tracing(TracingConfig())
    assert tool.calls == 2
    assert len([event for event in events if isinstance(event, ToolResultEvent)]) == 2
    assert any(isinstance(event, ErrorEvent) for event in events)
    persisted = sessions.run_store.events.list(run_id=agent._current_run_id)
    triggered = [event for event in persisted if event.event_type == "loop_guard.triggered"]
    assert len(triggered) == 1
    assert triggered[0].payload["repeat_count"] == 3
    spans = [
        json.loads(line)
        for line in trace_file.read_text(encoding="utf-8").splitlines()
    ]
    blocked = [
        span
        for span in spans
        if span["name"] == "loop_guard.evaluate"
        and span["attributes"]["loop.blocked"]
    ]
    assert len(blocked) == 1
