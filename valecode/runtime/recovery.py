from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from valecode.persistence import (
    RunStatus,
    RunStore,
    StepStatus,
    ToolCallState,
    ToolCallStatus,
)
from valecode.runtime.idempotency import FileEffectState, inspect_file_effect


class RecoveryAction(StrEnum):
    REUSE_RESULT = "reuse_result"
    RETRY = "retry"
    CONFIRM = "confirm"


@dataclass(frozen=True)
class ToolRecovery:
    tool_call_id: str
    action: RecoveryAction
    reason: str


@dataclass(frozen=True)
class RunRecovery:
    run_id: str
    previous_status: RunStatus
    safe_step_id: str | None
    transcript_valid: bool
    missing_tool_results: tuple[str, ...]
    repaired_tool_results: tuple[str, ...]
    tools: tuple[ToolRecovery, ...]


@dataclass(frozen=True)
class RecoveryReport:
    runs: tuple[RunRecovery, ...] = ()

    @property
    def requires_confirmation(self) -> bool:
        return any(
            tool.action == RecoveryAction.CONFIRM
            for run in self.runs
            for tool in run.tools
        )


class RecoveryService:
    """Reconcile stale control-plane state left by an unclean process exit."""

    def __init__(self, run_store: RunStore, work_dir: str | Path) -> None:
        self.run_store = run_store
        self.work_dir = Path(work_dir)
        self.sessions_dir = self.work_dir / ".valecode" / "sessions"

    def scan_and_reconcile(self) -> RecoveryReport:
        recovered: list[RunRecovery] = []
        for run in self.run_store.list_unfinished_runs():
            steps = self.run_store.list_steps(run.id)
            calls = self.run_store.list_tool_calls(run.id)
            decisions = tuple(self._reconcile_tool(call) for call in calls if call.status in {
                ToolCallStatus.PENDING,
                ToolCallStatus.RUNNING,
                ToolCallStatus.UNCERTAIN,
            })
            calls = self.run_store.list_tool_calls(run.id)
            safe_step_id = self._last_safe_step_id(steps, calls)
            transcript_valid, missing = self._validate_transcript(run.session_id, calls)
            repaired = self._repair_transcript(run.session_id, calls, missing)
            if repaired:
                transcript_valid, missing = self._validate_transcript(run.session_id, calls)
                self.run_store.events.append(
                    "recovery.transcript_repaired",
                    session_id=run.session_id,
                    run_id=run.id,
                    payload={"tool_call_ids": list(repaired)},
                    idempotency_key=f"recovery:transcript:{run.id}",
                )

            for step in steps:
                if step.status in {StepStatus.PENDING, StepStatus.RUNNING, StepStatus.BLOCKED}:
                    self.run_store.transition_step(
                        step.id,
                        StepStatus.INTERRUPTED,
                        error="Recovered after unclean shutdown",
                        event_payload={"recovery": True},
                    )
            self.run_store.transition_run(
                run.id,
                RunStatus.INTERRUPTED,
                error="Recovered after unclean shutdown",
                event_payload={
                    "safe_step_id": safe_step_id,
                    "transcript_valid": transcript_valid,
                    "missing_tool_results": list(missing),
                },
            )
            self.run_store.events.append(
                "recovery.run_reconciled",
                session_id=run.session_id,
                run_id=run.id,
                payload={
                    "previous_status": run.status.value,
                    "safe_step_id": safe_step_id,
                    "transcript_valid": transcript_valid,
                    "requires_confirmation": any(
                        decision.action == RecoveryAction.CONFIRM for decision in decisions
                    ),
                },
                idempotency_key=f"recovery:{run.id}",
            )
            recovered.append(
                RunRecovery(
                    run_id=run.id,
                    previous_status=run.status,
                    safe_step_id=safe_step_id,
                    transcript_valid=transcript_valid,
                    missing_tool_results=missing,
                    repaired_tool_results=repaired,
                    tools=decisions,
                )
            )
        return RecoveryReport(tuple(recovered))

    def resolve_uncertain_tool(
        self, tool_call_id: str, *, allow_retry: bool
    ) -> ToolCallState:
        call = self.run_store.get_tool_call(tool_call_id)
        if call is None:
            raise KeyError(f"Tool call not found: {tool_call_id}")
        if call.status != ToolCallStatus.UNCERTAIN:
            raise ValueError(f"Tool call is not uncertain: {tool_call_id}")
        target = ToolCallStatus.RUNNING if allow_retry else ToolCallStatus.CANCELLED
        resolved = self.run_store.transition_tool_call(
            tool_call_id,
            target,
            error=None if allow_retry else "Recovery retry rejected by user",
            event_payload={"recovery": True, "approved": allow_retry},
        )
        self.run_store.events.append(
            "recovery.tool_resolved",
            run_id=call.run_id,
            step_id=call.step_id,
            tool_call_id=call.id,
            payload={"allow_retry": allow_retry, "status": target.value},
        )
        return resolved

    def _reconcile_tool(self, call: ToolCallState) -> ToolRecovery:
        if call.status == ToolCallStatus.UNCERTAIN:
            return ToolRecovery(
                call.id, RecoveryAction.CONFIRM, "Previous recovery marked this call uncertain"
            )

        if call.status == ToolCallStatus.PENDING:
            action = RecoveryAction.RETRY
            reason = "Call was committed but never started"
            self.run_store.events.append(
                "recovery.tool_classified",
                run_id=call.run_id,
                step_id=call.step_id,
                tool_call_id=call.id,
                payload={"action": action.value, "reason": reason},
                idempotency_key=f"recovery:tool:{call.id}",
            )
            return ToolRecovery(call.id, action, reason)
        elif call.side_effect_class == "read":
            action = RecoveryAction.RETRY
            reason = "Read-only call is safe to execute again"
        elif call.side_effect_class == "write":
            inspection = inspect_file_effect(call.tool_name, call.arguments, self.work_dir)
            if inspection.state == FileEffectState.APPLIED:
                self.run_store.transition_tool_call(
                    call.id,
                    ToolCallStatus.COMPLETED,
                    result={"output": f"Recovered: {inspection.reason}", "is_error": False},
                    is_error=False,
                    event_payload={"recovery": True, "file_state": inspection.state.value},
                )
                self.run_store.events.append(
                    "recovery.tool_classified",
                    run_id=call.run_id,
                    step_id=call.step_id,
                    tool_call_id=call.id,
                    payload={"action": RecoveryAction.REUSE_RESULT.value, "reason": inspection.reason},
                    idempotency_key=f"recovery:tool:{call.id}",
                )
                return ToolRecovery(call.id, RecoveryAction.REUSE_RESULT, inspection.reason)
            action = RecoveryAction.CONFIRM
            reason = inspection.reason
        else:
            action = RecoveryAction.CONFIRM
            reason = "External or unknown side effect may already have occurred"

        self.run_store.transition_tool_call(
            call.id,
            ToolCallStatus.UNCERTAIN,
            error=reason,
            event_payload={"recovery": True, "action": action.value},
        )
        self.run_store.events.append(
            "recovery.tool_classified",
            run_id=call.run_id,
            step_id=call.step_id,
            tool_call_id=call.id,
            payload={"action": action.value, "reason": reason},
            idempotency_key=f"recovery:tool:{call.id}",
        )
        return ToolRecovery(call.id, action, reason)

    @staticmethod
    def _last_safe_step_id(steps: list[Any], calls: list[ToolCallState]) -> str | None:
        calls_by_step: dict[str, list[ToolCallState]] = {}
        for call in calls:
            calls_by_step.setdefault(call.step_id, []).append(call)
        safe: str | None = None
        for step in steps:
            step_calls = calls_by_step.get(step.id, [])
            if step.status == StepStatus.COMPLETED and all(
                call.status in {
                    ToolCallStatus.COMPLETED,
                    ToolCallStatus.FAILED,
                    ToolCallStatus.DENIED,
                    ToolCallStatus.CANCELLED,
                }
                for call in step_calls
            ):
                safe = step.id
        return safe

    def _validate_transcript(
        self, session_id: str, calls: list[ToolCallState]
    ) -> tuple[bool, tuple[str, ...]]:
        path = self.sessions_dir / f"{session_id}.jsonl"
        if not path.exists():
            return False, tuple(
                call.metadata.get("provider_tool_call_id", call.id)
                for call in calls
                if call.status == ToolCallStatus.COMPLETED
            )
        pending: set[str] = set()
        results: set[str] = set()
        malformed = False
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    malformed = True
                    continue
                if record.get("type") == "assistant" and isinstance(record.get("content"), list):
                    for block in record["content"]:
                        if isinstance(block, dict) and block.get("type") == "tool_use":
                            tool_id = block.get("id")
                            if tool_id:
                                pending.add(tool_id)
                elif record.get("type") == "tool_result":
                    tool_id = record.get("tool_use_id")
                    if tool_id:
                        results.add(tool_id)
                        pending.discard(tool_id)
        except OSError:
            return False, ()
        completed_ids = {
            call.metadata.get("provider_tool_call_id", call.id)
            for call in calls
            if call.status == ToolCallStatus.COMPLETED
        }
        missing = tuple(sorted(completed_ids - results))
        return not malformed and not pending and not missing, missing

    def _repair_transcript(
        self,
        session_id: str,
        calls: list[ToolCallState],
        missing: tuple[str, ...],
    ) -> tuple[str, ...]:
        if not missing:
            return ()
        path = self.sessions_dir / f"{session_id}.jsonl"
        if not path.exists():
            return ()
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return ()
        # Only append results for tool-use ids already present in the transcript.
        # This repairs the common crash window between DB commit and JSONL append
        # without fabricating an assistant message that was never persisted.
        present_uses: set[str] = set()
        for line in text.splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("type") != "assistant" or not isinstance(record.get("content"), list):
                continue
            for block in record["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("id"):
                    present_uses.add(block["id"])

        by_provider_id = {
            call.metadata.get("provider_tool_call_id", call.id): call
            for call in calls
            if call.status == ToolCallStatus.COMPLETED
        }
        repaired: list[str] = []
        try:
            with path.open("a", encoding="utf-8") as handle:
                for provider_id in missing:
                    if provider_id not in present_uses or provider_id not in by_provider_id:
                        continue
                    call = by_provider_id[provider_id]
                    output = ""
                    if call.result_path:
                        try:
                            output = Path(call.result_path).read_text(encoding="utf-8")
                        except OSError:
                            pass
                    if not output and isinstance(call.result, dict):
                        output = str(call.result.get("output", ""))
                    record = {
                        "type": "tool_result",
                        "content": output,
                        "timestamp": datetime.now(UTC).isoformat(),
                        "tool_use_id": provider_id,
                        "is_error": False,
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    repaired.append(provider_id)
                handle.flush()
        except OSError:
            return tuple(repaired)
        return tuple(repaired)
