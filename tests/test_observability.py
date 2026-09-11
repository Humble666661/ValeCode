from __future__ import annotations

import json
from typing import Any, AsyncIterator

import asyncio

import pytest

from valecode.agent import Agent, LoopComplete
from valecode.client import LLMClient
from valecode.conversation import ConversationManager
from valecode.observability import (
    Tracing,
    TracingConfig,
    configure_tracing,
    sanitize_attributes,
)
from valecode.permissions import Decision, PermissionMode
from valecode.tools import create_default_registry
from valecode.tools.base import StreamEnd, StreamEvent, TextDelta, ToolCallComplete


class ScriptedClient(LLMClient):
    def __init__(self, responses: list[list[StreamEvent]]) -> None:
        self.responses = list(responses)

    async def stream(
        self,
        conversation: ConversationManager,
        system: str = "",
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        for event in self.responses.pop(0):
            yield event


class AllowPermissions:
    mode = PermissionMode.BYPASS

    def check(self, tool: Any, arguments: dict[str, Any]) -> Decision:
        return Decision(effect="allow", reason="test allow")


def _read_spans(path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_attribute_redaction_defaults_and_opt_in() -> None:
    hidden = sanitize_attributes(
        {"input": "private prompt", "api_key": "secret", "safe": "visible"}
    )
    assert hidden == {
        "input": "[REDACTED]",
        "api_key": "[REDACTED]",
        "safe": "visible",
    }

    captured = sanitize_attributes(
        {"input": "private prompt", "api_key": "secret"}, capture_content=True
    )
    assert captured["input"] == "private prompt"
    assert captured["api_key"] == "[REDACTED]"


def test_file_exporter_preserves_parent_child_trace_and_redacts(tmp_path) -> None:
    trace_file = tmp_path / "traces.jsonl"
    tracing = configure_tracing(
        TracingConfig(enabled=True, exporter="file", file_path=str(trace_file))
    )
    try:
        with tracing.span("parent", {"input": "secret", "safe": "ok"}):
            with tracing.span("child", {"tool.name": "ReadFile"}):
                pass
        assert tracing.force_flush()
    finally:
        tracing.shutdown()
        configure_tracing(TracingConfig())

    spans = _read_spans(trace_file)
    by_name = {span["name"]: span for span in spans}
    assert by_name["parent"]["trace_id"] == by_name["child"]["trace_id"]
    assert by_name["child"]["parent_span_id"] == by_name["parent"]["span_id"]
    assert by_name["parent"]["attributes"]["input"] == "[REDACTED]"
    assert by_name["parent"]["attributes"]["safe"] == "ok"


def test_span_records_error_status_without_leaking_message(tmp_path) -> None:
    trace_file = tmp_path / "error-trace.jsonl"
    tracing = configure_tracing(
        TracingConfig(enabled=True, exporter="file", file_path=str(trace_file))
    )
    try:
        with pytest.raises(ValueError, match="private failure"):
            with tracing.span("failed.operation"):
                raise ValueError("private failure")
        assert tracing.force_flush()
    finally:
        tracing.shutdown()
        configure_tracing(TracingConfig())

    span = _read_spans(trace_file)[0]
    assert span["status"] == "ERROR"
    assert span["events"][0]["attributes"]["exception.type"] == "ValueError"
    assert span["events"][0]["attributes"]["exception.message"] == "[REDACTED]"


@pytest.mark.asyncio
async def test_trace_context_propagates_into_async_tasks(tmp_path) -> None:
    trace_file = tmp_path / "async-traces.jsonl"
    tracing = configure_tracing(
        TracingConfig(enabled=True, exporter="file", file_path=str(trace_file))
    )

    async def child() -> None:
        with tracing.span("child.agent", {"agent.parent_id": "parent"}):
            await asyncio.sleep(0)

    try:
        with tracing.span("parent.agent"):
            await asyncio.create_task(child())
        assert tracing.force_flush()
    finally:
        tracing.shutdown()
        configure_tracing(TracingConfig())

    spans = _read_spans(trace_file)
    by_name = {span["name"]: span for span in spans}
    assert by_name["parent.agent"]["trace_id"] == by_name["child.agent"]["trace_id"]
    assert by_name["child.agent"]["parent_span_id"] == by_name["parent.agent"]["span_id"]


@pytest.mark.asyncio
async def test_agent_emits_core_spans(tmp_path) -> None:
    source = tmp_path / "sample.txt"
    source.write_text("hello", encoding="utf-8")
    trace_file = tmp_path / "agent-traces.jsonl"
    tracing = configure_tracing(
        TracingConfig(enabled=True, exporter="file", file_path=str(trace_file))
    )
    client = ScriptedClient(
        [
            [
                ToolCallComplete(
                    "tool-1", "ReadFile", {"file_path": str(source)}
                ),
                StreamEnd("end_turn", input_tokens=10, output_tokens=4),
            ],
            [
                TextDelta("done"),
                StreamEnd("end_turn", input_tokens=12, output_tokens=2),
            ],
        ]
    )
    agent = Agent(
        client,
        create_default_registry(),
        "anthropic",
        work_dir=str(tmp_path),
        permission_checker=AllowPermissions(),  # type: ignore[arg-type]
        provider_name="test-provider",
        model="test-model",
        tracing=tracing,
    )
    conversation = ConversationManager()
    conversation.add_user_message("read the sample")
    try:
        events = [event async for event in agent.run(conversation)]
        assert any(isinstance(event, LoopComplete) for event in events)
        assert tracing.force_flush()
    finally:
        tracing.shutdown()
        configure_tracing(TracingConfig())

    spans = _read_spans(trace_file)
    names = [span["name"] for span in spans]
    assert names.count("llm.stream") == 2
    assert "agent.run" in names
    assert "context.compact" in names
    assert "tool.execute" in names
    assert "permission.evaluate" in names
    assert len({span["trace_id"] for span in spans}) == 1

    tool_span = next(span for span in spans if span["name"] == "tool.execute")
    assert tool_span["attributes"]["tool.name"] == "ReadFile"
    assert tool_span["attributes"]["tool.arguments"] == "[REDACTED]"
    assert tool_span["attributes"]["tool.output"] == "[REDACTED]"


@pytest.mark.asyncio
async def test_parent_and_child_agents_share_request_trace(tmp_path) -> None:
    trace_file = tmp_path / "agent-tree.jsonl"
    tracing = configure_tracing(
        TracingConfig(enabled=True, exporter="file", file_path=str(trace_file))
    )
    parent = Agent(
        ScriptedClient(
            [[TextDelta("parent"), StreamEnd("end_turn", input_tokens=1, output_tokens=1)]]
        ),
        create_default_registry(),
        "anthropic",
        work_dir=str(tmp_path),
        tracing=tracing,
    )
    parent_conversation = ConversationManager()
    parent_conversation.add_user_message("parent task")
    try:
        _ = [event async for event in parent.run(parent_conversation)]
        child = Agent(
            ScriptedClient(
                [[TextDelta("child"), StreamEnd("end_turn", input_tokens=1, output_tokens=1)]]
            ),
            create_default_registry(),
            "anthropic",
            work_dir=str(tmp_path),
            tracing=tracing,
        )
        child.parent_id = parent.agent_id
        child.trace_id = parent._current_trace_id
        child_conversation = ConversationManager()
        child_conversation.add_user_message("child task")
        _ = [event async for event in child.run(child_conversation)]
        assert tracing.force_flush()
    finally:
        tracing.shutdown()
        configure_tracing(TracingConfig())

    agent_spans = [
        span for span in _read_spans(trace_file) if span["name"] == "agent.run"
    ]
    assert len(agent_spans) == 2
    assert agent_spans[0]["trace_id"] == agent_spans[1]["trace_id"]
    assert agent_spans[1]["attributes"]["agent.parent_id"] == parent.agent_id


@pytest.mark.asyncio
async def test_tracing_failure_does_not_interrupt_agent(tmp_path) -> None:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult

    class BrokenExporter:
        def export(self, spans: Any) -> SpanExportResult:
            raise RuntimeError("exporter unavailable")

        def shutdown(self) -> None:
            return

        def force_flush(self, timeout_millis: int = 30_000) -> bool:
            return True

    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(BrokenExporter()))
    tracing = Tracing(provider.get_tracer("test"), enabled=True, provider=provider)

    client = ScriptedClient(
        [[TextDelta("still works"), StreamEnd("end_turn", input_tokens=1, output_tokens=2)]]
    )
    agent = Agent(
        client,
        create_default_registry(),
        "anthropic",
        work_dir=str(tmp_path),
        tracing=tracing,
    )
    conversation = ConversationManager()
    conversation.add_user_message("hello")

    try:
        events = [event async for event in agent.run(conversation)]
        assert any(isinstance(event, LoopComplete) for event in events)
    finally:
        tracing.shutdown()
