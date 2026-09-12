from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from valecode.runtime import (
    CancellationToken,
    ExecutionController,
    ExecutionLimits,
    ExecutionTimeoutError,
)
from valecode.tools.agent_tool import AgentTool


@pytest.mark.asyncio
async def test_shared_tool_capacity_limits_parallel_agents() -> None:
    controller = ExecutionController(
        ExecutionLimits(max_concurrent_tools=2, tool_timeout=1)
    )
    token = CancellationToken()
    active = 0
    peak = 0

    async def operation(index: int) -> int:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return index

    results = await asyncio.gather(
        *(
            controller.execute_tool(
                lambda index=index: operation(index),
                token=token,
                tool_name=f"tool-{index}",
            )
            for index in range(6)
        )
    )

    assert results == list(range(6))
    assert peak == 2


@pytest.mark.asyncio
async def test_cancellation_token_stops_inflight_tool() -> None:
    controller = ExecutionController(
        ExecutionLimits(max_concurrent_tools=1, tool_timeout=10)
    )
    token = CancellationToken()
    started = asyncio.Event()
    stopped = asyncio.Event()

    async def operation() -> None:
        started.set()
        try:
            await asyncio.sleep(10)
        finally:
            stopped.set()

    task = asyncio.create_task(
        controller.execute_tool(
            operation,
            token=token,
            tool_name="slow",
        )
    )
    await started.wait()
    token.cancel("stop the run")

    with pytest.raises(asyncio.CancelledError, match="stop the run"):
        await task
    assert stopped.is_set()


@pytest.mark.asyncio
async def test_tool_timeout_cancels_operation() -> None:
    controller = ExecutionController(
        ExecutionLimits(max_concurrent_tools=1, tool_timeout=0.01)
    )
    token = CancellationToken()

    with pytest.raises(ExecutionTimeoutError, match="tool slow timed out"):
        await controller.execute_tool(
            lambda: asyncio.sleep(10),
            token=token,
            tool_name="slow",
        )


@pytest.mark.asyncio
async def test_llm_stream_uses_one_total_deadline() -> None:
    controller = ExecutionController(ExecutionLimits(llm_timeout=0.03))
    token = CancellationToken()

    async def slow_stream():
        yield "first"
        await asyncio.sleep(0.05)
        yield "second"

    received: list[str] = []
    with pytest.raises(ExecutionTimeoutError, match="LLM stream timed out"):
        async for item in controller.stream(slow_stream(), token=token):
            received.append(item)
    assert received == ["first"]


def test_execution_limits_load_from_layered_env(tmp_path, monkeypatch) -> None:
    keys = (
        "VALECODE_MAX_CONCURRENT_TOOLS",
        "VALECODE_LLM_TIMEOUT",
        "VALECODE_TOOL_TIMEOUT",
        "VALECODE_PERMISSION_TIMEOUT",
    )
    for key in keys:
        monkeypatch.delenv(key, raising=False)
    (tmp_path / ".env").write_text(
        "\n".join(
            (
                "VALECODE_MAX_CONCURRENT_TOOLS=3",
                "VALECODE_LLM_TIMEOUT=12.5",
                "VALECODE_TOOL_TIMEOUT=7",
                "VALECODE_PERMISSION_TIMEOUT=2",
            )
        ),
        encoding="utf-8",
    )

    limits = ExecutionLimits.from_environment(tmp_path)
    assert limits == ExecutionLimits(
        max_concurrent_tools=3,
        llm_timeout=12.5,
        tool_timeout=7,
        permission_timeout=2,
    )


def test_subagent_inherits_controller_and_cancellation_token() -> None:
    controller = ExecutionController()
    token = CancellationToken()
    parent = SimpleNamespace(
        agent_id="parent",
        _current_run_id="run-parent",
        _current_trace_id="trace-parent",
        trace_id=None,
        session_id="session",
        run_store=object(),
        provider_name="provider",
        model="model",
        tracing=object(),
        execution_controller=controller,
        cancellation_token=token,
    )
    child = SimpleNamespace()
    tool = AgentTool.__new__(AgentTool)
    tool._parent_agent = parent

    tool._inherit_runtime_state(child)

    assert child.execution_controller is controller
    assert child.cancellation_token is token
    assert child._owns_cancellation_token is False
