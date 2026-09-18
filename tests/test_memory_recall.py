"""Memory recall wiring and cross-turn de-duplication."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from valecode.agent import Agent
from valecode.app import ValeCodeApp
from valecode.conversation import ConversationManager, ToolUseBlock
from valecode.memory.recall import (
    MemoryRecallResult,
    RelevantMemory,
    SurfacedMemoryStore,
    render_reminder_with_paths,
)
from valecode.tools import create_default_registry
from valecode.tools.base import StreamEnd, TextDelta


def test_render_reports_only_readable_paths(tmp_path: Path) -> None:
    available = tmp_path / "available.md"
    available.write_text("remember this", encoding="utf-8")
    result = render_reminder_with_paths([
        RelevantMemory(str(available), 0),
        RelevantMemory(str(tmp_path / "missing.md"), 0),
    ])
    assert result.paths == [str(available)]
    assert "remember this" in result.text
    assert render_reminder_with_paths([
        RelevantMemory(str(tmp_path / "missing.md"), 0)
    ]).text == ""


def test_surfaced_state_survives_restart_and_rejects_bad_session(tmp_path: Path) -> None:
    store = SurfacedMemoryStore(tmp_path, "session-a")
    store.save({"memory-a", "memory-b"})
    assert SurfacedMemoryStore(tmp_path, "session-a").load() == {
        "memory-a", "memory-b"
    }
    assert SurfacedMemoryStore(tmp_path, "session-b").load() == set()
    with pytest.raises(ValueError, match="session ID"):
        SurfacedMemoryStore(tmp_path, "../outside")
    store.path.write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError):
        store.load()


@pytest.mark.asyncio
async def test_prefetch_passes_session_surfaced_and_recent_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from valecode import app as app_module

    app = ValeCodeApp([])
    app.agent = SimpleNamespace(work_dir=str(tmp_path))
    app.memory_manager = SimpleNamespace(user_mem_dir=tmp_path, project_mem_dir=tmp_path)
    app._selected_provider = object()
    app.session = SimpleNamespace(session_id="session-a")
    app._surfaced_memories["session-a"] = {"old-memory"}
    app.conversation.add_assistant_message("", [
        ToolUseBlock("1", "ReadFile", {}),
        ToolUseBlock("2", "Grep", {}),
        ToolUseBlock("3", "ReadFile", {}),
    ])
    memory = tmp_path / "memory.md"
    memory.write_text("useful fact", encoding="utf-8")
    captured = {}

    async def fake_find(**kwargs):
        captured.update(kwargs)
        return [RelevantMemory(str(memory), 0)]

    monkeypatch.setattr(app_module, "find_relevant_memories", fake_find)
    result = await app._prefetch_relevant_memories("question")
    assert result.paths == [str(memory)]
    assert captured["already_surfaced"] == {"old-memory"}
    assert captured["recent_tools"] == ["Grep", "ReadFile"]
    assert app._surfaced_memories["session-a"] == {"old-memory"}

    app.session = SimpleNamespace(session_id="session-b")
    await app._prefetch_relevant_memories("question")
    assert captured["already_surfaced"] == set()

    app._mark_surfaced_memories("session-a", [str(memory)])
    restarted = ValeCodeApp([])
    restarted.agent = SimpleNamespace(work_dir=str(tmp_path))
    restarted.memory_manager = app.memory_manager
    restarted._selected_provider = object()
    restarted.session = SimpleNamespace(session_id="session-a")
    assert restarted._get_surfaced_memories("session-a") == {"old-memory", str(memory)}


@pytest.mark.asyncio
async def test_surfaced_only_after_recall_is_injected() -> None:
    from tests.test_agent import MockLLMClient

    agent = Agent(MockLLMClient([]), create_default_registry(), "anthropic")
    conversation = ConversationManager()
    surfaced: set[str] = set()
    agent.memory_recall_on_surfaced = lambda paths: surfaced.update(paths)
    pending = asyncio.get_running_loop().create_future()
    agent.memory_recall_task = pending
    agent._consume_memory_recall(conversation)
    assert not surfaced
    pending.set_result(MemoryRecallResult("memory text", ["memory-a"]))
    agent._consume_memory_recall(conversation)
    assert surfaced == {"memory-a"}
    assert "memory text" in conversation.history[-1].content
    agent._consume_memory_recall(conversation)
    assert len(conversation.history) == 1

    agent._memory_recall_consumed = False
    empty = asyncio.get_running_loop().create_future()
    empty.set_result(MemoryRecallResult("", ["memory-b"]))
    agent.memory_recall_task = empty
    agent._consume_memory_recall(conversation)
    assert surfaced == {"memory-a"}


@pytest.mark.asyncio
async def test_no_tool_answer_sees_recall_before_model_request() -> None:
    from tests.test_agent import MockLLMClient

    class InspectingClient(MockLLMClient):
        async def stream(self, conversation, system="", tools=None):
            assert any("needed fact" in msg.content for msg in conversation.history)
            async for event in super().stream(conversation, system, tools):
                yield event

    client = InspectingClient([[
        TextDelta("answer"),
        StreamEnd("end_turn", input_tokens=1, output_tokens=1),
    ]])
    agent = Agent(client, create_default_registry(), "anthropic")
    conversation = ConversationManager()
    conversation.add_user_message("question")

    async def recall():
        await asyncio.sleep(0)
        return MemoryRecallResult("needed fact", ["memory-a"])

    agent.memory_recall_task = asyncio.create_task(recall())
    surfaced: set[str] = set()
    agent.memory_recall_on_surfaced = lambda paths: surfaced.update(paths)
    async for _ in agent.run(conversation):
        pass
    assert surfaced == {"memory-a"}
    assert client._call_index == 1
