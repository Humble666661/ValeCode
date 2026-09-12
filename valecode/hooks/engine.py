from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

from valecode.hooks.executors import execute_action
from valecode.hooks.models import ActionResult, Hook, HookContext, ToolRejectedError
from valecode.observability import Tracing, get_tracing

log = logging.getLogger(__name__)


@dataclass
class HookNotification:
    hook_id: str
    event: str
    output: str
    success: bool
    duration_ms: float = 0.0
    error_type: str = ""


class HookEngine:
    def __init__(
        self,
        hooks: list[Hook] | None = None,
        tracing: Tracing | None = None,
    ) -> None:
        self.hooks: list[Hook] = hooks or []
        self.tracing = tracing or get_tracing()
        self._prompt_messages: list[str] = []
        self._notifications: list[HookNotification] = []
        self._async_tasks: set[asyncio.Task[None]] = set()


    def find_matching_hooks(self, event: str, ctx: HookContext) -> list[Hook]:
        matched: list[Hook] = []
        for hook in self.hooks:
            if hook.event != event:
                continue
            if not hook.should_run():
                continue
            if hook.condition is not None and not hook.condition.evaluate(ctx):
                continue
            matched.append(hook)
        return matched


    async def run_hooks(self, event: str, ctx: HookContext) -> None:
        matched = self.find_matching_hooks(event, ctx)
        for hook in matched:
            hook.mark_executed()
            if hook.async_exec:
                task = asyncio.create_task(self._run_single(hook, ctx))
                self._async_tasks.add(task)
                task.add_done_callback(self._async_tasks.discard)
            else:
                await self._run_single(hook, ctx)


    async def _run_single(self, hook: Hook, ctx: HookContext) -> None:
        result, duration_ms, error_type = await self._execute(hook, ctx)
        if hook.action.type == "prompt" and result.success:
            self._prompt_messages.append(result.output)
        self._notifications.append(
            HookNotification(
                hook_id=hook.id,
                event=hook.event,
                output=result.output,
                success=result.success,
                duration_ms=duration_ms,
                error_type=error_type,
            )
        )

    async def _execute(
        self, hook: Hook, ctx: HookContext
    ) -> tuple[ActionResult, float, str]:
        """Run every hook through one timeout, tracing, and error policy."""

        started = time.monotonic()
        error_type = ""
        with self.tracing.span(
            "hook.execute",
            {
                "trace.id": ctx.trace_id,
                "session.id": ctx.session_id,
                "run.id": ctx.run_id,
                "step.id": ctx.step_id,
                "tool.call_id": ctx.tool_call_id,
                "hook.id": hook.id,
                "hook.event": hook.event,
                "hook.action_type": hook.action.type,
                "hook.async": hook.async_exec,
                "hook.reject": hook.reject,
            },
            trace_id=ctx.trace_id or None,
        ) as span:
            try:
                result = await asyncio.wait_for(
                    execute_action(hook.action, ctx),
                    timeout=hook.action.timeout,
                )
            except asyncio.TimeoutError:
                error_type = "TimeoutError"
                result = ActionResult(
                    output=f"Hook timed out after {hook.action.timeout}s",
                    success=False,
                )
            except Exception as exc:
                error_type = type(exc).__name__
                result = ActionResult(
                    output=f"Hook execution error: {exc}",
                    success=False,
                )
            duration_ms = round((time.monotonic() - started) * 1000, 3)
            span.set_attributes(
                {
                    "hook.success": result.success,
                    "hook.duration_ms": duration_ms,
                    "hook.error_type": error_type,
                    "hook.output": result.output,
                }
            )
            if not result.success:
                span.set_error(result.output)
                log.warning("Hook '%s' action failed: %s", hook.id, result.output)
            return result, duration_ms, error_type


    async def run_pre_tool_hooks(
        self, ctx: HookContext
    ) -> ToolRejectedError | None:
        matched = self.find_matching_hooks("pre_tool_use", ctx)
        for hook in matched:
            hook.mark_executed()
            result, duration_ms, error_type = await self._execute(hook, ctx)
            self._notifications.append(
                HookNotification(
                    hook_id=hook.id,
                    event="pre_tool_use",
                    output=result.output,
                    success=result.success,
                    duration_ms=duration_ms,
                    error_type=error_type,
                )
            )
            # reject hooks are deliberately fail-closed: an action failure or
            # timeout must not accidentally permit the guarded tool.
            if hook.reject:
                return ToolRejectedError(
                    tool=ctx.tool_name,
                    reason=result.output,
                    hook_id=hook.id,
                )
        return None

    async def shutdown(self) -> None:
        """Wait for best-effort asynchronous hooks without leaking tasks."""

        if not self._async_tasks:
            return
        await asyncio.gather(*tuple(self._async_tasks), return_exceptions=True)

    def get_prompt_messages(self) -> list[str]:
        messages = list(self._prompt_messages)
        self._prompt_messages.clear()
        return messages


    def drain_notifications(self) -> list[HookNotification]:
        notifications = list(self._notifications)
        self._notifications.clear()
        return notifications
