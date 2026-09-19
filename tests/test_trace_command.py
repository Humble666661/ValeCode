from __future__ import annotations

from types import SimpleNamespace

import pytest

from valecode.agents.trace import TraceManager
from valecode.commands.handlers.trace import create_trace_command
from valecode.commands.registry import CommandContext
from valecode.memory.session import SessionManager
from valecode.persistence import RunStatus, StepStatus


class _UI:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def add_system_message(self, text: str) -> None:
        self.messages.append(text)


@pytest.mark.asyncio
async def test_trace_command_restores_persisted_tree_after_restart(tmp_path) -> None:
    sessions = SessionManager(str(tmp_path))
    session = sessions.create()
    store = sessions.run_store
    root = store.create_run(
        session.session_id,
        agent_id="lead-agent",
        trace_id="trace-1",
        metadata={"agent_type": "lead"},
    )
    store.transition_run(root.id, RunStatus.RUNNING)
    child = store.create_run(
        session.session_id,
        agent_id="child-agent",
        parent_run_id=root.id,
        trace_id="trace-1",
        metadata={"agent_type": "Explore"},
    )
    store.transition_run(child.id, RunStatus.RUNNING)
    step = store.create_step(child.id)
    store.transition_step(step.id, StepStatus.RUNNING)
    store.create_tool_call(child.id, step.id, "ReadFile", {})
    store.transition_step(
        step.id,
        StepStatus.COMPLETED,
        input_tokens=12,
        output_tokens=4,
    )
    store.transition_run(child.id, RunStatus.COMPLETED)

    ui = _UI()
    context = CommandContext(
        args="",
        agent=None,
        conversation=None,
        session=session,
        session_manager=sessions,
        memory_manager=None,
        ui=ui,
        config={},
    )
    command = create_trace_command(TraceManager(), lead_agent_id="lead-agent")
    await command.handler(context)
    session.close()

    output = ui.messages[-1]
    assert "Agent 追踪树" in output
    assert "lead" in output
    assert "Explore" in output
    assert "↑12 ↓4" in output
    assert "工具×1" in output
    assert "2 runs" in output


@pytest.mark.asyncio
async def test_trace_command_keeps_unpersisted_live_nodes() -> None:
    manager = TraceManager()
    manager.create("worker", parent_id="lead-agent", trace_id="trace-live")
    ui = _UI()
    context = CommandContext(
        args="",
        agent=None,
        conversation=None,
        session=SimpleNamespace(session_id="session-1"),
        session_manager=None,
        memory_manager=None,
        ui=ui,
        config={},
    )

    await create_trace_command(manager).handler(context)

    assert "worker" in ui.messages[-1]
    assert "running" in ui.messages[-1]
