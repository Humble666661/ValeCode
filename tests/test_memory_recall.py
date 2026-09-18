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
    render_reminder_with_paths,
)
from valecode.tools import create_default_registry


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


@pytest.mark.asyncio
async def test_prefetch_passes_session_surfaced_and_recent_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from valecode import app as app_module

    app = ValeCodeApp([])
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
