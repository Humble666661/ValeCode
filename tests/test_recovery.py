from __future__ import annotations

from typing import Any, AsyncIterator

import pytest
from pydantic import BaseModel

from valecode.agent import Agent
from valecode.client import LLMClient
from valecode.conversation import ConversationManager, Message, ToolUseBlock
from valecode.memory.session import SessionManager
from valecode.persistence import RunStatus, StepStatus, ToolCallStatus
from valecode.runtime import (
    FileEffectState,
    RecoveryAction,
    RecoveryService,
    inspect_file_effect,
    make_tool_idempotency_key,
)
from valecode.tools import ToolRegistry
from valecode.tools.base import StreamEnd, StreamEvent, Tool, ToolCallComplete, ToolResult


def test_idempotency_key_uses_canonical_arguments():
    first = make_tool_idempotency_key("run", "call", "Tool", {"b": 2, "a": 1})
    second = make_tool_idempotency_key("run", "call", "Tool", {"a": 1, "b": 2})
    different = make_tool_idempotency_key("run", "call", "Tool", {"a": 2, "b": 1})
    assert first == second
    assert first != different


def test_file_effect_inspection(tmp_path):
    target = tmp_path / "sample.txt"
    target.write_text("new", encoding="utf-8")
    write = inspect_file_effect(
        "WriteFile", {"file_path": "sample.txt", "content": "new"}, tmp_path
    )
    edit = inspect_file_effect(
        "EditFile",
        {"file_path": "sample.txt", "old_string": "old", "new_string": "new"},
        tmp_path,
    )
    assert write.state == FileEffectState.APPLIED
    assert edit.state == FileEffectState.APPLIED


def test_startup_reconciles_stale_run_by_side_effect(tmp_path):
    first = SessionManager(str(tmp_path))
    session = first.create()
    run = first.run_store.create_run(session.session_id)
    first.run_store.transition_run(run.id, RunStatus.RUNNING)

    safe = first.run_store.create_step(run.id)
    first.run_store.transition_step(safe.id, StepStatus.RUNNING)
    first.run_store.transition_step(safe.id, StepStatus.COMPLETED)
    active = first.run_store.create_step(run.id)
    first.run_store.transition_step(active.id, StepStatus.RUNNING)

    read = first.run_store.create_tool_call(
        run.id, active.id, "ReadFile", {"file_path": "a.txt"}, side_effect_class="read"
    )
    first.run_store.transition_tool_call(read.id, ToolCallStatus.RUNNING)
    target = tmp_path / "written.txt"
    target.write_text("expected", encoding="utf-8")
    write = first.run_store.create_tool_call(
        run.id,
        active.id,
        "WriteFile",
        {"file_path": "written.txt", "content": "expected"},
        side_effect_class="write",
    )
    first.run_store.transition_tool_call(write.id, ToolCallStatus.RUNNING)
    external = first.run_store.create_tool_call(
        run.id, active.id, "Bash", {"command": "deploy"}, side_effect_class="external"
    )
    first.run_store.transition_tool_call(external.id, ToolCallStatus.RUNNING)
    pending = first.run_store.create_tool_call(
        run.id, active.id, "Bash", {"command": "never-started"}, side_effect_class="external"
    )
    session.close()

    restarted = SessionManager(str(tmp_path))
    report = restarted.recovery_report

    assert len(report.runs) == 1
    recovered = report.runs[0]
    assert recovered.run_id == run.id
    assert recovered.safe_step_id == safe.id
    assert recovered.transcript_valid is False
    assert recovered.missing_tool_results == (write.id,)
    decisions = {item.tool_call_id: item.action for item in recovered.tools}
    assert decisions == {
        read.id: RecoveryAction.RETRY,
        write.id: RecoveryAction.REUSE_RESULT,
        external.id: RecoveryAction.CONFIRM,
        pending.id: RecoveryAction.RETRY,
    }
    assert report.requires_confirmation is True
    assert restarted.run_store.get_run(run.id).status == RunStatus.INTERRUPTED
    assert restarted.run_store.get_step(active.id).status == StepStatus.INTERRUPTED
    assert restarted.run_store.get_tool_call(read.id).status == ToolCallStatus.UNCERTAIN
    assert restarted.run_store.get_tool_call(write.id).status == ToolCallStatus.COMPLETED
    assert restarted.run_store.get_tool_call(external.id).status == ToolCallStatus.UNCERTAIN
    assert restarted.run_store.get_tool_call(pending.id).status == ToolCallStatus.PENDING

    # Repeated startup is idempotent and does not recover the same run twice.
    assert SessionManager(str(tmp_path)).recovery_report.runs == ()
    events = restarted.run_store.events.list(run_id=run.id)
    assert len([e for e in events if e.event_type == "recovery.run_reconciled"]) == 1

    resolved = RecoveryService(restarted.run_store, tmp_path).resolve_uncertain_tool(
        external.id, allow_retry=False
    )
    assert resolved.status == ToolCallStatus.CANCELLED


def test_startup_repairs_committed_tool_result_in_transcript(tmp_path):
    first = SessionManager(str(tmp_path))
    session = first.create()
    session.append(
        Message(
            role="assistant",
            content="",
            tool_uses=[
                ToolUseBlock("provider-write", "WriteFile", {"file_path": "done.txt"})
            ],
        )
    )
    run = first.run_store.create_run(session.session_id)
    first.run_store.transition_run(run.id, RunStatus.RUNNING)
    step = first.run_store.create_step(run.id)
    first.run_store.transition_step(step.id, StepStatus.RUNNING)
    target = tmp_path / "done.txt"
    target.write_text("done", encoding="utf-8")
    call = first.run_store.create_tool_call(
        run.id,
        step.id,
        "WriteFile",
        {"file_path": "done.txt", "content": "done"},
        side_effect_class="write",
        metadata={"provider_tool_call_id": "provider-write"},
    )
    first.run_store.transition_tool_call(call.id, ToolCallStatus.RUNNING)
    session.close()

    restarted = SessionManager(str(tmp_path))
    recovered = restarted.recovery_report.runs[0]
    assert recovered.repaired_tool_results == ("provider-write",)
    assert recovered.missing_tool_results == ()
    assert recovered.transcript_valid is True
    resumed = restarted.resume(session.session_id)
    assert resumed is not None
    assert resumed.messages[-1].tool_results[0].tool_use_id == "provider-write"
    assert resumed.messages[-1].tool_results[0].content.startswith("Recovered:")
    resumed.session.close()


class CountParams(BaseModel):
    value: str


class CountTool(Tool):
    name = "Count"
    description = "Count executions"
    params_model = CountParams
    category = "read"

    def __init__(self) -> None:
        self.calls = 0

    async def execute(self, params: CountParams) -> ToolResult:
        self.calls += 1
        return ToolResult(params.value)


class ResumeClient(LLMClient):
    def __init__(self) -> None:
        self.index = 0

    async def stream(
        self,
        conversation: ConversationManager,
        system: str = "",
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        self.index += 1
        if self.index == 1:
            yield ToolCallComplete("same-call", "Count", {"value": "persisted"})
            yield StreamEnd("tool_use", 2, 1)
        else:
            yield StreamEnd("end_turn", 3, 1)


@pytest.mark.asyncio
async def test_resume_reuses_completed_tool_result(tmp_path):
    manager = SessionManager(str(tmp_path))
    session = manager.create()
    run = manager.run_store.create_run(session.session_id, trace_id="trace-one")
    manager.run_store.transition_run(run.id, RunStatus.RUNNING)
    old_step = manager.run_store.create_step(run.id)
    manager.run_store.transition_step(old_step.id, StepStatus.RUNNING)
    call = manager.run_store.create_tool_call(
        run.id,
        old_step.id,
        "Count",
        {"value": "persisted"},
        idempotency_key=make_tool_idempotency_key(
            run.id, "same-call", "Count", {"value": "persisted"}
        ),
        side_effect_class="read",
        metadata={"provider_tool_call_id": "same-call"},
    )
    manager.run_store.transition_tool_call(call.id, ToolCallStatus.RUNNING)
    manager.run_store.transition_tool_call(
        call.id,
        ToolCallStatus.COMPLETED,
        result={"output": "persisted", "is_error": False},
    )
    manager.run_store.transition_step(old_step.id, StepStatus.COMPLETED)
    manager.run_store.transition_run(run.id, RunStatus.INTERRUPTED)

    tool = CountTool()
    registry = ToolRegistry()
    registry.register(tool)
    agent = Agent(
        ResumeClient(),
        registry,
        "anthropic",
        work_dir=str(tmp_path),
        run_store=manager.run_store,
    )
    agent.session_id = session.session_id
    agent.resume_run(run.id)
    conversation = ConversationManager()
    conversation.add_user_message("continue")

    _ = [event async for event in agent.run(conversation)]

    assert tool.calls == 0
    assert manager.run_store.get_run(run.id).status == RunStatus.COMPLETED
    assert len(manager.run_store.list_tool_calls(run.id)) == 1
    session.close()
