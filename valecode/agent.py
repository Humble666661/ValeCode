from __future__ import annotations

import asyncio
import logging
import sys
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Callable

from pydantic import ValidationError

from valecode.client import LLMClient
from valecode.context import (
    CompactBoundary,
    CompactCircuitBreaker,
    CompactEvent,
    ContentReplacementRecord,
    ContentReplacementState,
    RecoveryState,
    append_replacement_records,
    apply_tool_result_budget,
    auto_compact,
    create_replacement_state,
    ensure_session_dir,
    load_replacement_records,
    reconstruct_replacement_state,
)
from valecode.conversation import ConversationManager, ToolResultBlock, ToolUseBlock
from valecode.conversation import ThinkingBlock as ConvThinkingBlock
from valecode.memory.auto_memory import MemoryManager
from valecode.permissions import (
    Decision,
    PermissionChecker,
    PermissionMode,
)
from valecode.persistence import (
    ResultArtifactStore,
    RunStatus,
    RunStore,
    StepStatus,
    ToolCallStatus,
)
from valecode.runtime.idempotency import make_tool_idempotency_key
from valecode.runtime import (
    CancellationToken,
    ExecutionController,
    LoopGuard,
    RetryPolicy,
    RuntimeEvent,
)
from valecode.observability import Tracing, get_tracing
from valecode.hooks import HookContext, HookEngine, ToolRejectedError
from valecode.hooks.engine import HookNotification
from valecode.prompts import build_environment_context, build_plan_mode_reminder, build_system_prompt
from valecode.tools import ToolRegistry
from valecode.tools.base import (
    MAX_OUTPUT_CHARS,
    StreamEnd,
    StreamEvent,
    TextDelta,
    ThinkingComplete,
    ThinkingDelta,
    ToolCallComplete,
    ToolCallDelta,
    ToolCallStart,
    ToolResult,
)

log = logging.getLogger(__name__)

MEMORY_EXTRACTION_INTERVAL = 5
MAX_TOKENS_CEILING = 64000
MAX_OUTPUT_TOKENS_RECOVERIES = 3


# ---------------------------------------------------------------------------
# AgentEvent 事件类型
# ---------------------------------------------------------------------------

@dataclass
class StreamText(RuntimeEvent):
    EVENT_TYPE = "stream.text"
    text: str


@dataclass
class ThinkingText(RuntimeEvent):
    EVENT_TYPE = "stream.thinking"
    text: str


@dataclass
class RetryEvent(RuntimeEvent):
    EVENT_TYPE = "llm.retry"
    reason: str
    wait: float = 0.0


@dataclass
class ToolUseEvent(RuntimeEvent):
    EVENT_TYPE = "tool.use"
    tool_name: str
    tool_id: str
    arguments: dict[str, Any]


@dataclass
class ToolResultEvent(RuntimeEvent):
    EVENT_TYPE = "tool.result"
    tool_id: str
    tool_name: str
    output: str
    is_error: bool
    elapsed: float


@dataclass
class TurnComplete(RuntimeEvent):
    EVENT_TYPE = "turn.completed"
    turn: int


@dataclass
class LoopComplete(RuntimeEvent):
    EVENT_TYPE = "run.completed"
    total_turns: int


@dataclass
class UsageEvent(RuntimeEvent):
    EVENT_TYPE = "usage.updated"
    input_tokens: int
    output_tokens: int


@dataclass
class ErrorEvent(RuntimeEvent):
    EVENT_TYPE = "runtime.error"
    message: str


@dataclass
class CompactNotification(RuntimeEvent):
    EVENT_TYPE = "context.compacted"
    before_tokens: int
    message: str
    # 结构化 boundary（摘要 + 原文保留尾部），UI/session 层用它持久化 compact_boundary 记录。
    # 失败路径下为 None。
    boundary: "CompactBoundary | None" = None


@dataclass
class HookEvent(RuntimeEvent):
    EVENT_TYPE = "hook.completed"
    hook_id: str
    event: str
    output: str
    success: bool
    duration_ms: float = 0.0
    error_type: str = ""


class PermissionResponse(Enum):
    ALLOW = "allow"
    DENY = "deny"
    ALLOW_SESSION = "allow_session"
    ALLOW_ALWAYS = "allow_always"  # Legacy response; now session-scoped too.


@dataclass
class PermissionRequest(RuntimeEvent):
    EVENT_TYPE = "permission.requested"
    tool_name: str
    description: str
    future: asyncio.Future[PermissionResponse]


@dataclass
class PermissionDecisionEvent(RuntimeEvent):
    EVENT_TYPE = "permission.responded"
    tool_name: str
    response: str


@dataclass
class MailboxEvent(RuntimeEvent):
    EVENT_TYPE = "mailbox.received"
    message_id: str
    from_agent: str
    to_agent: str
    message_type: str
    content: str
    summary: str = ""


AgentEvent = (
    StreamText
    | ThinkingText
    | RetryEvent
    | ToolUseEvent
    | ToolResultEvent
    | TurnComplete
    | LoopComplete
    | UsageEvent
    | ErrorEvent
    | PermissionRequest
    | PermissionDecisionEvent
    | MailboxEvent
    | CompactNotification
    | HookEvent
)


# ---------------------------------------------------------------------------
# LLM 响应收集器
# ---------------------------------------------------------------------------

@dataclass
class ThinkingBlock:
    thinking: str
    signature: str


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCallComplete] = field(default_factory=list)
    thinking_blocks: list[ThinkingBlock] = field(default_factory=list)
    stop_reason: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    cache_creation: int = 0


class StreamCollector:
    def __init__(self) -> None:
        self.response = LLMResponse()

    async def consume(
        self, stream: AsyncIterator[StreamEvent]
    ) -> AsyncIterator[AgentEvent]:
        async for event in stream:
            if isinstance(event, TextDelta):
                self.response.text += event.text
                yield StreamText(text=event.text)
            elif isinstance(event, ThinkingDelta):
                yield ThinkingText(text=event.text)
            elif isinstance(event, ThinkingComplete):
                self.response.thinking_blocks.append(
                    ThinkingBlock(thinking=event.thinking, signature=event.signature)
                )
            elif isinstance(event, ToolCallStart):
                pass
            elif isinstance(event, ToolCallDelta):
                pass
            elif isinstance(event, ToolCallComplete):
                self.response.tool_calls.append(event)
                yield ToolUseEvent(
                    tool_name=event.tool_name,
                    tool_id=event.tool_id,
                    arguments=event.arguments,
                )
            elif isinstance(event, StreamEnd):
                self.response.stop_reason = event.stop_reason
                self.response.input_tokens = event.input_tokens
                self.response.output_tokens = event.output_tokens
                self.response.cache_read = event.cache_read
                self.response.cache_creation = event.cache_creation


# ---------------------------------------------------------------------------
# tool 批量执行
# ---------------------------------------------------------------------------

@dataclass
class ToolBatch:
    concurrent: bool
    calls: list[ToolCallComplete]


def partition_tool_calls(
    tool_calls: list[ToolCallComplete],
    registry: ToolRegistry,
) -> list[ToolBatch]:
    batches: list[ToolBatch] = []
    for tc in tool_calls:
        tool = registry.get(tc.tool_name)
        safe = tool is not None and tool.is_concurrency_safe and registry.is_enabled(tc.tool_name)

        if safe and batches and batches[-1].concurrent:
            batches[-1].calls.append(tc)
        else:
            batches.append(ToolBatch(concurrent=safe, calls=[tc]))
    return batches


# ---------------------------------------------------------------------------
# streaming 执行器 — 在 LLM streaming 期间启动 tool 执行
# ---------------------------------------------------------------------------

@dataclass
class _ToolExecResult:
    tool_id: str
    tool_name: str
    result: ToolResult
    elapsed: float
    is_unknown: bool


class StreamingExecutor:
    def __init__(self) -> None:
        self._tasks: list[tuple[int, asyncio.Task[_ToolExecResult]]] = []
        self._order = 0

    def submit(
        self,
        coro: Any,
    ) -> None:
        task = asyncio.create_task(coro)
        self._tasks.append((self._order, task))
        self._order += 1

    async def collect_results(self) -> list[_ToolExecResult]:
        if not self._tasks:
            return []
        tasks = [t for _, t in sorted(self._tasks, key=lambda x: x[0])]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        out: list[_ToolExecResult] = []
        for r in results:
            if isinstance(r, Exception):
                out.append(_ToolExecResult(
                    tool_id="",
                    tool_name="",
                    result=ToolResult(output=f"Tool execution error: {r}", is_error=True),
                    elapsed=0.0,
                    is_unknown=False,
                ))
            else:
                out.append(r)
        return out


# ---------------------------------------------------------------------------
# Agent 主循环
# ---------------------------------------------------------------------------

class Agent:
    def __init__(
        self,
        client: LLMClient,
        registry: ToolRegistry,
        protocol: str,
        work_dir: str = ".",
        max_iterations: int = 0,
        permission_checker: PermissionChecker | None = None,
        context_window: int = 200_000,
        instructions_content: str = "",
        memory_manager: MemoryManager | None = None,
        hook_engine: HookEngine | None = None,
        run_store: RunStore | None = None,
        provider_name: str | None = None,
        model: str | None = None,
        tracing: Tracing | None = None,
        retry_policy: RetryPolicy | None = None,
        loop_guard: LoopGuard | None = None,
        execution_controller: ExecutionController | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> None:
        self.client = client
        self.registry = registry
        self.protocol = protocol
        self.work_dir = work_dir
        self.max_iterations = max_iterations
        self.permission_checker = permission_checker
        self.permission_mode: PermissionMode = (
            permission_checker.mode if permission_checker else PermissionMode.DEFAULT
        )
        self.context_window = context_window
        self.session_dir = ensure_session_dir(work_dir)
        self.compact_breaker = CompactCircuitBreaker()
        self.replacement_state: ContentReplacementState = create_replacement_state()
        # 保存重建工作上下文所需的快照，在 Layer 2 压缩对话后使用：
        # 最近的文件读取和 skill 调用。每次 ReadFile / skill 调用时记录，
        # auto_compact 触发阈值时消费。
        self.recovery_state: RecoveryState = RecoveryState()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.instructions_content = instructions_content
        self.memory_manager = memory_manager
        self.hook_engine = hook_engine
        self._hook_agent_lock = asyncio.Lock()
        self.run_store = run_store
        run_database = getattr(run_store, "database", None)
        self.result_artifact_store = (
            ResultArtifactStore(run_database) if run_database is not None else None
        )
        self._persisted_result_paths: dict[str, str] = {}
        self._artifact_sweep_session_id: str | None = None
        self.provider_name = provider_name
        self.model = model
        self.tracing = tracing or get_tracing()
        self.retry_policy = retry_policy or RetryPolicy.from_environment(work_dir)
        self.loop_guard = loop_guard or LoopGuard.from_environment(work_dir)
        self.execution_controller = (
            execution_controller or ExecutionController.from_environment(work_dir)
        )
        self.cancellation_token = cancellation_token or CancellationToken()
        self._owns_cancellation_token = cancellation_token is None
        self._current_run_id: str | None = None
        self._current_trace_id: str | None = None
        self._resume_run_id: str | None = None
        self._current_step_id: str | None = None
        self._control_tool_ids: dict[str, str] = {}
        self._loop_count = 0
        self._event_sequence = 0
        # 记忆提取合并策略（对齐 Go 版 inProgress + pendingContext）：
        # _extracting: 标记是否有提取正在进行
        # _pending_extraction: 提取期间又触发了新请求，标记需要尾随提取
        self._extracting = False
        self._pending_extraction = False
        self.session_id: str = ""
        self.active_skills: dict[str, str] = {}
        self._skill_catalog: str = ""
        self._agent_catalog: str = ""
        self._agent_catalog_list: list[tuple[str, str]] = []
        self.agent_id: str = uuid.uuid4().hex[:12]
        self.parent_id: str | None = None
        # ``parent_id`` is the tracing agent id. This separate field is a
        # database run id and is therefore safe to use as a foreign key.
        self.parent_run_id: str | None = None
        self.trace_id: str | None = None
        self.coordinator_mode: bool = False
        self.team_name: str = ""
        self._team_manager: Any = None
        self.notification_fn: Callable[[], list[str]] | None = None
        self.todo_state_provider: Callable[[], str] | None = None
        self.file_history: Any = None

        # Memory recall starts during UI preparation and is available to the
        # first model call, including requests that never invoke a tool.
        self.memory_recall_task: Any | None = None
        self._memory_recall_consumed: bool = False
        self.memory_recall_on_surfaced: Callable[[list[str]], None] | None = None

    @property
    def _transcript_path(self) -> str:
        if self.session_id:
            return str(Path(self.work_dir) / ".valecode" / "sessions" / f"{self.session_id}.jsonl")
        return ""

    def _consume_memory_recall(self, conversation: ConversationManager) -> None:
        """Inject a completed recall and mark only actually injected files."""
        task = self.memory_recall_task
        if task is None or self._memory_recall_consumed or not task.done():
            return
        if task.cancelled():
            self._memory_recall_consumed = True
            return
        try:
            recall = task.result()
            if recall:
                from valecode.memory.recall import MemoryRecallResult
                if isinstance(recall, MemoryRecallResult):
                    if recall.text:
                        conversation.add_system_reminder(recall.text)
                        if self.memory_recall_on_surfaced:
                            self.memory_recall_on_surfaced(recall.paths)
                else:
                    conversation.add_system_reminder(recall)
        except Exception:
            pass  # Recall must not interrupt the main agent loop.
        self._memory_recall_consumed = True

    @property
    def plan_mode(self) -> bool:
        return self.permission_mode == PermissionMode.PLAN

    _plan_path_cache: Path | None = None

    def _get_plan_path(self) -> Path:
        if self._plan_path_cache is not None:
            return self._plan_path_cache
        import random
        import datetime
        _ADJECTIVES = ["bold", "bright", "calm", "cool", "deep", "fair", "fast", "fine",
                       "glad", "keen", "kind", "lean", "mild", "neat", "pure", "safe",
                       "slim", "soft", "tall", "warm", "wise", "grand", "swift", "vivid"]
        _NOUNS = ["sketch", "draft", "spark", "bloom", "trail", "ridge", "creek", "grove",
                  "cliff", "cloud", "field", "forge", "frost", "haven", "pearl", "stone",
                  "storm", "river", "tower", "delta", "flame", "orbit", "pulse", "shore"]
        plans_dir = Path(self.work_dir) / ".valecode" / "plans"
        plans_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%m%d-%H%M")
        slug = f"{random.choice(_ADJECTIVES)}-{random.choice(_NOUNS)}-{ts}"
        self._plan_path_cache = plans_dir / f"{slug}.md"
        return self._plan_path_cache

    def set_permission_mode(self, mode: PermissionMode) -> None:
        self.permission_mode = mode
        if self.permission_checker:
            self.permission_checker.mode = mode

    def activate_skill(
        self,
        name: str,
        prompt_body: str,
        permission_rules: dict[str, list[str]] | None = None,
    ) -> None:
        self.active_skills[name] = prompt_body
        checker = getattr(self, "permission_checker", None)
        if checker:
            checker.bind_skill_scope(
                name, permission_rules or {}
            )

    def clear_active_skills(self) -> None:
        checker = getattr(self, "permission_checker", None)
        if checker:
            for name in self.active_skills:
                checker.release_skill_scope(name)
        self.active_skills.clear()

    def set_skill_catalog(self, catalog: str) -> None:
        self._skill_catalog = catalog

    def set_todo_state_provider(self, provider: Callable[[], str]) -> None:
        self.todo_state_provider = provider

    def _system_with_todo_progress(self, system: str) -> str:
        if self.todo_state_provider is None:
            return system
        try:
            progress = self.todo_state_provider()
        except Exception:
            log.exception("Unable to load current task progress")
            return system
        return system + "\n\n# Current task progress\n" + progress if progress else system


    def set_agent_catalog(self, catalog: str, catalog_list: list[tuple[str, str]] | None = None) -> None:
        self._agent_catalog = catalog
        if catalog_list is not None:
            self._agent_catalog_list = catalog_list

    def _build_hook_context(self, event: str, **kwargs: str | dict) -> HookContext:
        return HookContext(
            event_name=event,
            tool_name=str(kwargs.get("tool_name", "")),
            tool_args=kwargs.get("tool_args", {}),
            file_path=str(kwargs.get("file_path", "")),
            message=str(kwargs.get("message", "")),
            error=str(kwargs.get("error", "")),
            trace_id=self._current_trace_id or self.trace_id or "",
            session_id=self.session_id,
            run_id=self._current_run_id or "",
            step_id=self._current_step_id or "",
            tool_call_id=str(kwargs.get("tool_call_id", "")),
            agent_runner=self._run_hook_agent,
        )

    async def _run_hook_agent(self, prompt: str) -> str:
        """Run an isolated, tool-free model call for an ``agent`` hook action."""
        async with self._hook_agent_lock:
            conversation = ConversationManager()
            conversation.add_user_message(prompt)
            collector = StreamCollector()
            stream = self.client.stream(
                conversation,
                system=(
                    "You are an isolated hook evaluator. Follow the hook prompt and "
                    "return concise plain text. You cannot call tools and must not "
                    "claim to have changed files or external state."
                ),
                tools=[],
            )
            async for _event in collector.consume(stream):
                pass
            response = collector.response
            if response.tool_calls:
                raise RuntimeError("Agent hook attempted to call a tool")
            output = response.text.strip()
            if not output:
                raise RuntimeError("Agent hook returned no text")
            self.total_input_tokens += response.input_tokens
            self.total_output_tokens += response.output_tokens
            return output

    def _infer_file_path(self, args: dict) -> str:
        return str(args.get("file_path", args.get("path", "")))

    async def _run_error_hook(self, error: BaseException) -> None:
        """Dispatch an error hook without ever replacing the original failure."""
        if not self.hook_engine:
            return
        try:
            message = f"{type(error).__name__}: {error}"
            await self.hook_engine.run_hooks(
                "error",
                self._build_hook_context("error", message=message, error=message),
            )
        except Exception:
            log.exception("Error hook dispatch failed")

    async def _run_permission_request_hook(
        self, tc: ToolCallComplete, description: str,
    ) -> None:
        if not self.hook_engine:
            return
        await self.hook_engine.run_hooks(
            "permission_request",
            self._build_hook_context(
                "permission_request",
                tool_name=tc.tool_name,
                tool_args=tc.arguments,
                file_path=self._infer_file_path(tc.arguments),
                message=description,
                tool_call_id=self._control_tool_ids.get(tc.tool_id, tc.tool_id),
            ),
        )

    async def _run_tool_lifecycle_hooks(
        self, tc: ToolCallComplete, result: ToolResult,
    ) -> None:
        """Dispatch semantic hooks only after a tool was actually invoked."""
        if not self.hook_engine:
            return
        event = ""
        if tc.tool_name in {"WriteFile", "EditFile"} and not result.is_error:
            event = "file_change"
        elif tc.tool_name == "Bash":
            event = "command_execute"
        if not event:
            return
        await self.hook_engine.run_hooks(
            event,
            self._build_hook_context(
                event,
                tool_name=tc.tool_name,
                tool_args=tc.arguments,
                file_path=self._infer_file_path(tc.arguments),
                message=result.output,
                error=result.output if result.is_error else "",
                tool_call_id=self._control_tool_ids.get(tc.tool_id, tc.tool_id),
            ),
        )

    def _drain_hook_events(self) -> list[HookEvent]:
        if not self.hook_engine:
            return []
        return [
            HookEvent(
                hook_id=n.hook_id,
                event=n.event,
                output=n.output,
                success=n.success,
                duration_ms=n.duration_ms,
                error_type=n.error_type,
            )
            for n in self.hook_engine.drain_notifications()
        ]

    def _control_call(self, operation: Callable[[], Any]) -> Any:
        """Best-effort control-plane write; never break the agent data path."""
        if self.run_store is None:
            return None
        try:
            return operation()
        except Exception:
            log.exception("Control-plane persistence failed")
            return None

    def _latest_user_input(self, conversation: ConversationManager) -> str:
        for message in reversed(conversation.history):
            if (
                message.role == "user"
                and message.content
                and not message.content.startswith("<system-reminder>")
            ):
                return message.content
        return ""

    def _start_control_run(
        self, conversation: ConversationManager, *, input_text: str | None = None
    ) -> None:
        self._current_run_id = None
        self._current_trace_id = None
        self._current_step_id = None
        self._control_tool_ids = {}
        self.loop_guard.reset()
        if (
            self._artifact_sweep_session_id != self.session_id
            and self.result_artifact_store is not None
            and self.session_id
        ):
            try:
                self.result_artifact_store.sweep_released(
                    root_dir=self.session_dir, session_id=self.session_id
                )
                self._persisted_result_paths = {
                    state.tool_use_id: state.path
                    for state in self.result_artifact_store.list_for_session(
                        self.session_id
                    )
                    if state.state == "active"
                }
                self._artifact_sweep_session_id = self.session_id
            except Exception:
                log.exception("Failed to sweep released tool-result artifacts")
        if self.run_store is None or not self.session_id:
            # Standalone/test agents still need a stable request trace so any
            # Sub-Agent launched during this run can join the same trace.
            self._current_trace_id = self.trace_id or uuid.uuid4().hex
            return
        if self._resume_run_id is not None:
            resume_run_id = self._resume_run_id
            self._resume_run_id = None
            run = self._control_call(lambda: self.run_store.get_run(resume_run_id))
            if run is None or run.session_id != self.session_id:
                log.error("Cannot resume run %s for session %s", resume_run_id, self.session_id)
                return
            if run.status == RunStatus.INTERRUPTED:
                run = self._control_call(
                    lambda: self.run_store.transition_run(
                        resume_run_id,
                        RunStatus.RUNNING,
                        event_payload={"recovery": True},
                    )
                )
            if run is not None and run.status == RunStatus.RUNNING:
                self._current_run_id = run.id
                self._current_trace_id = run.trace_id
            return
        trace_id = self.trace_id or uuid.uuid4().hex
        run = self._control_call(
            lambda: self.run_store.create_run(
                self.session_id,
                input=input_text if input_text is not None else self._latest_user_input(conversation),
                agent_id=self.agent_id,
                parent_run_id=self.parent_run_id,
                trace_id=trace_id,
                metadata={"protocol": self.protocol},
            )
        )
        if run is None:
            return
        self._current_run_id = run.id
        self._current_trace_id = run.trace_id
        self._control_call(
            lambda: self.run_store.transition_run(run.id, RunStatus.RUNNING)
        )

    def _start_control_step(self, iteration: int) -> None:
        if self.run_store is None or self._current_run_id is None:
            return
        step = self._control_call(
            lambda: self.run_store.create_step(
                self._current_run_id,
                provider=self.provider_name,
                model=self.model,
                metadata={"iteration": iteration},
            )
        )
        if step is None:
            return
        self._current_step_id = step.id
        self._control_call(
            lambda: self.run_store.transition_step(step.id, StepStatus.RUNNING)
        )

    def _finish_control_step(
        self,
        status: StepStatus,
        *,
        response: LLMResponse | None = None,
        error: str | None = None,
        event_payload: dict[str, Any] | None = None,
    ) -> None:
        if self.run_store is None or self._current_step_id is None:
            return
        step_id = self._current_step_id
        self._control_call(
            lambda: self.run_store.transition_step(
                step_id,
                status,
                input_tokens=response.input_tokens if response else None,
                output_tokens=response.output_tokens if response else None,
                error=error,
                event_payload=event_payload,
            )
        )
        if status not in {StepStatus.RUNNING, StepStatus.BLOCKED}:
            self._current_step_id = None

    def _finish_control_run(
        self, status: RunStatus, *, error: str | None = None
    ) -> None:
        if self.run_store is None or self._current_run_id is None:
            return
        run_id = self._current_run_id
        self._control_call(
            lambda: self.run_store.transition_run(run_id, status, error=error)
        )
        if status not in {RunStatus.RUNNING, RunStatus.BLOCKED}:
            self._current_run_id = None
            self._current_trace_id = None

    def _cancel_control_tools(self) -> None:
        if self.run_store is None or self._current_run_id is None:
            return
        for tool_call in self._control_call(
            lambda: self.run_store.list_tool_calls(self._current_run_id)
        ) or []:
            if tool_call.status in {ToolCallStatus.PENDING, ToolCallStatus.RUNNING}:
                self._control_call(
                    lambda tool_call=tool_call: self.run_store.transition_tool_call(
                        tool_call.id,
                        ToolCallStatus.CANCELLED,
                        error="Agent run cancelled",
                    )
                )

    def _register_control_tool_calls(
        self, calls: list[ToolCallComplete]
    ) -> None:
        if (
            self.run_store is None
            or self._current_run_id is None
            or self._current_step_id is None
        ):
            return
        for tc in calls:
            tool = self.registry.get(tc.tool_name)
            registration = self.registry.get_registration(tc.tool_name)
            category = getattr(tool, "category", "unknown") if tool else "unknown"
            side_effect_class = {
                "read": "read",
                "write": "write",
                "command": "external",
            }.get(category, "unknown")
            stored_id = f"{self._current_run_id}:{tc.tool_id}"
            created = self._control_call(
                lambda tc=tc, stored_id=stored_id, side_effect_class=side_effect_class: (
                    self.run_store.create_tool_call(
                        self._current_run_id,
                        self._current_step_id,
                        tc.tool_name,
                        tc.arguments,
                        tool_call_id=stored_id,
                        idempotency_key=make_tool_idempotency_key(
                            self._current_run_id,
                            tc.tool_id,
                            tc.tool_name,
                            tc.arguments,
                        ),
                        side_effect_class=side_effect_class,
                        metadata={
                            "provider_tool_call_id": tc.tool_id,
                            "registry_tool_id": (
                                registration.tool_id if registration else None
                            ),
                            "tool_source": (
                                registration.source.value if registration else "unknown"
                            ),
                            "tool_scope": (
                                registration.scope_id if registration else None
                            ),
                        },
                    )
                )
            )
            if created is not None:
                self._control_tool_ids[tc.tool_id] = created.id

    def resume_run(self, run_id: str) -> None:
        """Resume an interrupted run on the next ``run`` invocation."""
        self._resume_run_id = run_id

    def _control_existing_tool_result(
        self, provider_tool_id: str
    ) -> tuple[ToolResult, float, bool] | None:
        if self.run_store is None:
            return None
        stored_id = self._control_tool_ids.get(provider_tool_id)
        if stored_id is None:
            return None
        call = self._control_call(lambda: self.run_store.get_tool_call(stored_id))
        if call is None:
            return None
        if call.status == ToolCallStatus.COMPLETED:
            output = ""
            if call.result_path:
                try:
                    output = Path(call.result_path).read_text(encoding="utf-8")
                    self._persisted_result_paths[provider_tool_id] = call.result_path
                except OSError:
                    pass
            if not output and isinstance(call.result, dict):
                output = str(call.result.get("output", ""))
            return ToolResult(output=output, is_error=False), (call.elapsed_ms or 0) / 1000, False
        if call.status == ToolCallStatus.UNCERTAIN and call.side_effect_class != "read":
            return (
                ToolResult(
                    output=(
                        "Recovery blocked: this tool may already have produced an external "
                        "side effect. Confirm or compensate before retrying."
                    ),
                    is_error=True,
                ),
                (call.elapsed_ms or 0) / 1000,
                False,
            )
        return None

    def _transition_control_tool(
        self,
        provider_tool_id: str,
        status: ToolCallStatus,
        *,
        result: ToolResult | None = None,
        elapsed: float | None = None,
        result_path: str | None = None,
    ) -> None:
        if self.run_store is None:
            return
        stored_id = self._control_tool_ids.get(provider_tool_id)
        if stored_id is None:
            return
        existing = self._control_call(lambda: self.run_store.get_tool_call(stored_id))
        if existing is not None and existing.status in {
            ToolCallStatus.COMPLETED,
            ToolCallStatus.FAILED,
            ToolCallStatus.CANCELLED,
            ToolCallStatus.DENIED,
        }:
            return
        if existing is not None and existing.status == ToolCallStatus.UNCERTAIN and status != ToolCallStatus.RUNNING:
            return
        self._control_call(
            lambda: self.run_store.transition_tool_call(
                stored_id,
                status,
                result=(
                    {"output": result.output, "is_error": result.is_error}
                    if result is not None
                    else None
                ),
                is_error=result.is_error if result is not None else None,
                error=result.output if result is not None and result.is_error else None,
                elapsed_ms=round(elapsed * 1000) if elapsed is not None else None,
                result_path=result_path,
            )
        )

    @staticmethod
    def _tool_terminal_status(result: ToolResult) -> ToolCallStatus:
        if not result.is_error:
            return ToolCallStatus.COMPLETED
        if result.output.startswith(("Permission denied:", "Hook rejected:")):
            return ToolCallStatus.DENIED
        return ToolCallStatus.FAILED

    def _control_result_path(self, provider_tool_id: str, raw_output: str) -> str | None:
        from valecode.context.manager import SINGLE_RESULT_CHAR_LIMIT

        if len(raw_output) <= SINGLE_RESULT_CHAR_LIMIT:
            return None
        return self._persisted_result_paths.get(provider_tool_id)

    @staticmethod
    def _artifact_scope(value: str) -> str:
        return uuid.uuid5(uuid.NAMESPACE_URL, value or "unbound").hex[:16]

    def _persist_tool_result(
        self, tool_use_id: str, content: str, _session_dir: Path
    ) -> Path:
        from valecode.context.manager import persist_tool_result

        scoped_dir = (
            self.session_dir
            / self._artifact_scope(self.session_id or self.agent_id)
            / self._artifact_scope(self._current_run_id or "unbound")
        )
        path = persist_tool_result(tool_use_id, content, scoped_dir)
        self._persisted_result_paths[tool_use_id] = str(path)
        if self.result_artifact_store is not None:
            try:
                self.result_artifact_store.register(
                    path,
                    tool_use_id=tool_use_id,
                    session_id=self.session_id or None,
                    run_id=self._current_run_id,
                    step_id=self._current_step_id,
                    tool_call_id=self._control_tool_ids.get(tool_use_id),
                )
            except Exception:
                log.exception("Failed to index tool-result artifact %s", path)
        return path

    def _cleanup_result_artifacts(
        self, messages: list[Any], checkpoint_id: str
    ) -> None:
        from valecode.context.manager import PERSISTED_TAG

        referenced_tool_use_ids: set[str] = set()
        referenced_paths: set[str] = set()
        for message in messages:
            for result in message.tool_results:
                referenced_tool_use_ids.add(result.tool_use_id)
                lines = result.content.splitlines()
                if lines and lines[0] == PERSISTED_TAG and len(lines) > 2:
                    referenced_paths.add(lines[2].strip())
        if self.result_artifact_store is None or not self.session_id:
            from valecode.context.manager import cleanup_tool_results

            cleanup_tool_results(self.session_dir, referenced_tool_use_ids)
            return
        states = self.result_artifact_store.reconcile_references(
            self.session_id,
            referenced_tool_use_ids,
            root_dir=self.session_dir,
            checkpoint_id=checkpoint_id,
            referenced_paths=referenced_paths,
        )
        active_paths = {
            state.tool_use_id: state.path
            for state in states
            if state.state == "active"
        }
        self._persisted_result_paths = active_paths

    def _trace_attributes(self) -> dict[str, Any]:
        return {
            "trace.id": self._current_trace_id or self.trace_id or "",
            "session.id": self.session_id,
            "run.id": self._current_run_id or "",
            "step.id": self._current_step_id or "",
            "agent.id": self.agent_id,
            "agent.parent_id": self.parent_id or "",
        }

    def _record_runtime_event(
        self, event_type: str, payload: dict[str, Any]
    ) -> None:
        if self.run_store is None:
            return
        self._control_call(
            lambda: self.run_store.events.append(
                event_type,
                session_id=self.session_id or None,
                run_id=self._current_run_id,
                step_id=self._current_step_id,
                payload=payload,
            )
        )

    def cancel(self, reason: str = "Agent run cancelled") -> None:
        self.cancellation_token.cancel(reason)

    async def _execute_registered_tool(
        self, tool_name: str, params: Any
    ) -> ToolResult:
        tool = self.registry.get(tool_name)
        return await self.execution_controller.execute_tool(
            lambda: self.registry.execute(tool_name, params),
            token=self.cancellation_token,
            tool_name=tool_name,
            use_capacity=(
                tool.uses_global_capacity if tool is not None else True
            ),
        )

    def _prepare_event(self, event: AgentEvent) -> AgentEvent:
        self._event_sequence += 1
        provider_tool_id = getattr(event, "tool_id", None)
        stored_tool_id = (
            self._control_tool_ids.get(provider_tool_id)
            if isinstance(provider_tool_id, str)
            else None
        )
        event.stamp(
            sequence=self._event_sequence,
            session_id=self.session_id or None,
            run_id=self._current_run_id,
            step_id=self._current_step_id,
            tool_call_id=stored_tool_id,
            trace_id=self._current_trace_id or self.trace_id,
        )
        envelope = event.to_envelope()
        if self.run_store is not None:
            self._control_call(
                lambda: self.run_store.events.append(
                    f"runtime.{envelope.event_type}",
                    session_id=envelope.session_id,
                    run_id=envelope.run_id,
                    step_id=envelope.step_id,
                    tool_call_id=envelope.tool_call_id,
                    idempotency_key=f"runtime:{envelope.event_id}",
                    payload=envelope.to_dict(),
                )
            )
        return event

    async def _consume_llm_stream(
        self,
        collector: StreamCollector,
        conversation: ConversationManager,
        system: str,
        tools: list[dict[str, Any]],
    ) -> AsyncIterator[AgentEvent]:
        overall_started = time.monotonic()
        retries_used = 0
        total_wait = 0.0
        while True:
            attempt_started = time.monotonic()
            first_event_at: float | None = None
            trace_context = self.tracing.span(
                "llm.stream",
                {
                    **self._trace_attributes(),
                    "provider.name": self.provider_name or "",
                    "model.name": self.model or "",
                    "retry.count": retries_used,
                },
            )
            span = trace_context.__enter__()
            try:
                llm_stream = self.execution_controller.stream(
                    self.client.stream(
                        conversation, system=system, tools=tools
                    ),
                    token=self.cancellation_token,
                )
                async for event in collector.consume(llm_stream):
                    if first_event_at is None:
                        first_event_at = time.monotonic()
                    yield event
            except asyncio.CancelledError:
                trace_context.__exit__(*sys.exc_info())
                raise
            except BaseException as exc:
                decision = self.retry_policy.decide(
                    exc, retries_used=retries_used, total_wait=total_wait
                )
                payload = {
                    "attempt": decision.attempt,
                    "category": decision.category.value,
                    "retry": decision.retry,
                    "delay": decision.delay,
                    "reason": decision.reason,
                    "error_type": type(exc).__name__,
                }
                span.set_attributes(
                    {
                        "error.type": type(exc).__name__,
                        "retry.category": decision.category.value,
                        "retry.scheduled": decision.retry,
                        "retry.delay": decision.delay,
                        "retry.reason": decision.reason,
                    }
                )
                trace_context.__exit__(*sys.exc_info())
                if not decision.retry:
                    self._record_runtime_event("llm.retry_exhausted", payload)
                    raise
                self._record_runtime_event("llm.retry_scheduled", payload)
                retries_used += 1
                total_wait += decision.delay
                # Discard partial response state; the provider request is replayed
                # from the unchanged conversation on the next attempt.
                collector.response = LLMResponse()
                yield RetryEvent(
                    reason=(
                        f"{decision.category.value} "
                        f"({decision.attempt}/{self.retry_policy.max_retries})"
                    ),
                    wait=decision.delay,
                )
                with self.tracing.span(
                    "llm.retry",
                    {
                        **self._trace_attributes(),
                        **payload,
                        "retry.total_wait": total_wait,
                    },
                ):
                    await asyncio.sleep(decision.delay)
                continue
            else:
                response = collector.response
                finished = time.monotonic()
                span.set_attributes(
                    {
                        "llm.duration_ms": round(
                            (finished - attempt_started) * 1000, 3
                        ),
                        "llm.total_duration_ms": round(
                            (finished - overall_started) * 1000, 3
                        ),
                        "llm.ttft_ms": (
                            round((first_event_at - attempt_started) * 1000, 3)
                            if first_event_at is not None
                            else -1
                        ),
                        "llm.input_tokens": response.input_tokens,
                        "llm.output_tokens": response.output_tokens,
                        "llm.cache_read_tokens": response.cache_read,
                        "llm.cache_creation_tokens": response.cache_creation,
                        "llm.stop_reason": response.stop_reason or "",
                        "retry.count": retries_used,
                    }
                )
                trace_context.__exit__(None, None, None)
                return

    def _check_tool_loop(self, calls: list[ToolCallComplete]):
        for call in calls:
            # Preserve the existing unknown-tool circuit breaker and its more
            # specific diagnostic instead of shadowing it with repetition.
            if self.registry.get(call.tool_name) is None:
                self.loop_guard.reset()
                continue
            with self.tracing.span(
                "loop_guard.evaluate",
                {
                    **self._trace_attributes(),
                    "tool.name": call.tool_name,
                    "tool.call_id": call.tool_id,
                    "tool.arguments": call.arguments,
                },
            ) as span:
                decision = self.loop_guard.observe(call)
                span.set_attributes(
                    {
                        "loop.repeat_count": decision.repeat_count,
                        "loop.repeat_limit": self.loop_guard.repeat_limit,
                        "loop.blocked": decision.blocked,
                        "loop.signature": decision.signature,
                    }
                )
            if decision.blocked:
                self._record_runtime_event(
                    "loop_guard.triggered",
                    {
                        "tool_name": decision.tool_name,
                        "signature": decision.signature,
                        "repeat_count": decision.repeat_count,
                        "limit": self.loop_guard.repeat_limit,
                        "reason": decision.reason,
                    },
                )
                return decision
        return None

    async def _auto_compact_with_trace(
        self,
        conversation: ConversationManager,
        *,
        manual: bool = False,
    ) -> CompactEvent | str | None:
        started = time.monotonic()
        with self.tracing.span(
            "context.compact",
            {**self._trace_attributes(), "compact.manual": manual},
        ) as span:
            result = await auto_compact(
                conversation,
                self.client,
                self.context_window,
                self.session_dir,
                protocol=self.protocol,
                manual=manual,
                breaker=self.compact_breaker,
                recovery=self.recovery_state,
                tool_schemas=self.registry.get_all_schemas(self.protocol),
                transcript_path=self._transcript_path,
                cleanup_callback=self._cleanup_result_artifacts,
            )
            if isinstance(result, CompactEvent) and self.hook_engine:
                await self.hook_engine.run_hooks(
                    "compact",
                    self._build_hook_context(
                        "compact",
                        message=(
                            f"Compacted context from {result.before_tokens} tokens"
                        ),
                        tool_args={
                            "before_tokens": result.before_tokens,
                            "manual": manual,
                        },
                    ),
                )
            span.set_attributes(
                {
                    "compact.duration_ms": round(
                        (time.monotonic() - started) * 1000, 3
                    ),
                    "compact.outcome": (
                        "compacted"
                        if isinstance(result, CompactEvent)
                        else "error" if isinstance(result, str) else "skipped"
                    ),
                    "compact.before_tokens": (
                        result.before_tokens if isinstance(result, CompactEvent) else 0
                    ),
                }
            )
            return result

    async def run(self, conversation: ConversationManager) -> AsyncIterator[AgentEvent]:
        if self._owns_cancellation_token:
            self.cancellation_token = CancellationToken()
        self._start_control_run(conversation)
        self._event_sequence = 0
        run_id = self._current_run_id
        trace_id = self._current_trace_id
        trace_context = self.tracing.span(
            "agent.run",
            {
                "session.id": self.session_id,
                "run.id": run_id or "",
                "agent.id": self.agent_id,
                "agent.parent_id": self.parent_id or "",
                "provider.name": self.provider_name or "",
                "model.name": self.model or "",
                "input": self._latest_user_input(conversation),
            },
            trace_id=trace_id,
        )
        run_span = trace_context.__enter__()
        completed = False
        run_status = "running"
        try:
            async for event in self._run_loop(conversation):
                event = self._prepare_event(event)
                if isinstance(event, LoopComplete):
                    completed = True
                yield event
        except asyncio.CancelledError:
            run_status = "cancelled"
            self.cancel()
            self._cancel_control_tools()
            self._finish_control_step(StepStatus.CANCELLED, error="Agent run cancelled")
            self._finish_control_run(RunStatus.CANCELLED, error="Agent run cancelled")
            raise
        except Exception as exc:
            run_status = "failed"
            message = f"{type(exc).__name__}: {exc}"
            await self._run_error_hook(exc)
            self._finish_control_step(StepStatus.FAILED, error=message)
            self._finish_control_run(RunStatus.FAILED, error=message)
            raise
        else:
            if completed:
                run_status = "completed"
                self._finish_control_run(RunStatus.COMPLETED)
            else:
                run_status = "interrupted"
                self._finish_control_step(
                    StepStatus.INTERRUPTED, error="Agent loop ended without completion"
                )
                self._finish_control_run(
                    RunStatus.INTERRUPTED, error="Agent loop ended without completion"
                )
        finally:
            run_span.set_attributes(
                {
                    "run.status": run_status,
                    "usage.input_tokens": self.total_input_tokens,
                    "usage.output_tokens": self.total_output_tokens,
                }
            )
            trace_context.__exit__(*sys.exc_info())

    async def _run_loop(self, conversation: ConversationManager) -> AsyncIterator[AgentEvent]:
        self._current_conversation = conversation
        env_context = build_environment_context(
            self.work_dir, self.active_skills, self._skill_catalog, self._agent_catalog
        )
        conversation.inject_environment(env_context)

        memory_content = self.memory_manager.load() if self.memory_manager else ""
        conversation.inject_long_term_memory(self.instructions_content, memory_content)

        if self.hook_engine:
            ctx = self._build_hook_context("session_start")
            await self.hook_engine.run_hooks("session_start", ctx)
            for he in self._drain_hook_events():
                yield he

        iteration = 0
        consecutive_unknown = 0
        max_tokens_escalated = False
        output_recoveries = 0

        while True:
            iteration += 1

            if self.max_iterations > 0 and iteration > self.max_iterations:
                yield ErrorEvent(
                    message=f"Agent reached maximum iterations ({self.max_iterations})"
                )
                break

            if iteration == 1 and self.memory_recall_task is not None:
                try:
                    await self.memory_recall_task
                except asyncio.CancelledError:
                    if not self.memory_recall_task.cancelled():
                        raise
                except Exception:
                    pass  # Best-effort recall cannot stop the main request.
                self._consume_memory_recall(conversation)

            if self.hook_engine:
                ctx = self._build_hook_context("turn_start")
                await self.hook_engine.run_hooks("turn_start", ctx)
                for he in self._drain_hook_events():
                    yield he

            for mailbox_event in self._consume_mailbox(conversation):
                yield mailbox_event
            if self.notification_fn:
                for note in self.notification_fn():
                    conversation.add_system_reminder(note)

            if self.hook_engine:
                ctx = self._build_hook_context("pre_send")
                await self.hook_engine.run_hooks("pre_send", ctx)
                for he in self._drain_hook_events():
                    yield he

            hook_prompts = (
                self.hook_engine.get_prompt_messages() if self.hook_engine else None
            )
            system = build_system_prompt(
                hook_prompts=hook_prompts,
                coordinator_mode=self.coordinator_mode,
                agent_catalog=self._agent_catalog_list or None,
            )
            system = self._system_with_todo_progress(system)

            if self.plan_mode:
                plan_path = str(self._get_plan_path())
                if self.permission_checker:
                    self.permission_checker.plan_file_path = plan_path
                plan_exists = self._get_plan_path().exists()
                plan_reminder = build_plan_mode_reminder(
                    plan_path, plan_exists, iteration
                )
                conversation.add_system_reminder(plan_reminder)

            if self.hook_engine:
                for note in self.hook_engine.drain_notifications():
                    conversation.add_system_reminder(
                        f"Hook [{note.hook_id}] {note.event}: {note.output}"
                    )

            deferred_names = self.registry.get_deferred_tool_names()
            if deferred_names:
                conversation.add_system_reminder(
                    "The following deferred tools are available via ToolSearch. "
                    "Their schemas are NOT loaded - use ToolSearch with "
                    'query "select:<name>[,<name>...]" to load tool schemas before calling them:\n'
                    + "\n".join(deferred_names)
                )

            tools = self.registry.get_all_schemas(self.protocol)

            # Layer 1: build an immutable, budgeted API view.
            api_conversation, new_records = apply_tool_result_budget(
                conversation,
                self.session_dir,
                self.replacement_state,
                persist_callback=self._persist_tool_result,
            )
            if new_records:
                append_replacement_records(self.session_dir, new_records)

            # Layer 2: compact the budgeted view. Only a successful compaction
            # replaces the durable conversation; otherwise raw results remain intact.
            compact_result = await self._auto_compact_with_trace(api_conversation)
            if isinstance(compact_result, CompactEvent):
                yield CompactNotification(
                    before_tokens=compact_result.before_tokens,
                    message=f"上下文已压缩（压缩前 {compact_result.before_tokens:,} tokens）",
                    boundary=compact_result.boundary,
                )
                conversation.replace_history(api_conversation.get_messages())
                conversation.inject_environment(env_context)
                mem = self.memory_manager.load() if self.memory_manager else ""
                conversation.inject_long_term_memory(
                    self.instructions_content, mem
                )
                api_conversation, compact_records = apply_tool_result_budget(
                    conversation,
                    self.session_dir,
                    self.replacement_state,
                    persist_callback=self._persist_tool_result,
                )
                if compact_records:
                    append_replacement_records(self.session_dir, compact_records)
            elif isinstance(compact_result, str):
                yield ErrorEvent(message=compact_result)

            self._start_control_step(iteration)
            collector = StreamCollector()
            async for event in self._consume_llm_stream(
                collector, api_conversation, system, tools
            ):
                yield event

            response = collector.response

            if self.hook_engine:
                ctx = self._build_hook_context("post_receive", message=response.text)
                await self.hook_engine.run_hooks("post_receive", ctx)
                for he in self._drain_hook_events():
                    yield he

            self.total_input_tokens += response.input_tokens
            self.total_output_tokens += response.output_tokens
            yield UsageEvent(
                input_tokens=self.total_input_tokens,
                output_tokens=self.total_output_tokens,
            )

            conv_thinking = [
                ConvThinkingBlock(thinking=tb.thinking, signature=tb.signature)
                for tb in response.thinking_blocks
            ]

            if response.stop_reason == "max_tokens":
                self._finish_control_step(
                    StepStatus.COMPLETED,
                    response=response,
                    event_payload={"stop_reason": response.stop_reason},
                )
                if not max_tokens_escalated:
                    self.client.set_max_output_tokens(MAX_TOKENS_CEILING)
                    max_tokens_escalated = True
                    if response.text:
                        conversation.add_assistant_message(
                            response.text, thinking_blocks=conv_thinking
                        )
                        conversation.add_user_message(
                            "Output token limit hit. Resume directly from where you stopped. "
                            "Do not apologize or repeat previous content. Pick up mid-thought if needed."
                        )
                    yield RetryEvent(reason="max_tokens escalation")
                    continue
                elif output_recoveries < MAX_OUTPUT_TOKENS_RECOVERIES:
                    output_recoveries += 1
                    conversation.add_assistant_message(
                        response.text, thinking_blocks=conv_thinking
                    )
                    conversation.add_user_message(
                        "Output token limit hit. Resume directly from where you stopped. "
                        "Break remaining work into smaller pieces."
                    )
                    yield RetryEvent(
                        reason=f"max_tokens recovery {output_recoveries}/{MAX_OUTPUT_TOKENS_RECOVERIES}"
                    )
                    continue
            else:
                output_recoveries = 0

            if not response.tool_calls:
                conversation.add_assistant_message(
                    response.text, thinking_blocks=conv_thinking
                )
                self._loop_count += 1
                if (
                    self._loop_count % MEMORY_EXTRACTION_INTERVAL == 0
                    and self.memory_manager
                ):
                    asyncio.ensure_future(self._extract_memories(conversation))
                if self.hook_engine:
                    ctx = self._build_hook_context("turn_end")
                    await self.hook_engine.run_hooks("turn_end", ctx)
                    ctx = self._build_hook_context("session_end")
                    await self.hook_engine.run_hooks("session_end", ctx)
                    for he in self._drain_hook_events():
                        yield he
                if self.file_history is not None:
                    summary = response.text[:60] + "..." if len(response.text) > 60 else response.text
                    self.file_history.make_snapshot(len(conversation.history), summary)
                self._finish_control_step(
                    StepStatus.COMPLETED,
                    response=response,
                    event_payload={"stop_reason": response.stop_reason},
                )
                yield LoopComplete(total_turns=iteration)
                break

            loop_decision = self._check_tool_loop(response.tool_calls)
            if loop_decision is not None:
                self._finish_control_step(
                    StepStatus.FAILED,
                    response=response,
                    error=loop_decision.reason,
                    event_payload={"loop_guard": True},
                )
                yield ErrorEvent(message=f"Agent loop stopped: {loop_decision.reason}")
                break

            tool_uses = [
                ToolUseBlock(
                    tool_use_id=tc.tool_id,
                    tool_name=tc.tool_name,
                    arguments=tc.arguments,
                )
                for tc in response.tool_calls
            ]
            conversation.add_assistant_message(
                response.text, tool_uses, thinking_blocks=conv_thinking
            )
            # 在 assistant 回复加入历史后锚定实际用量：基线（input + cache + output）
            # 覆盖到当前位置，因此下一轮迭代顶部的 auto-compact 检查只需对
            # 接下来追加的 tool results 做字符估算。
            conversation.record_usage_anchor(
                response.input_tokens,
                response.output_tokens,
                response.cache_read,
                response.cache_creation,
            )
            self._register_control_tool_calls(response.tool_calls)

            tool_results: list[ToolResultBlock] = []
            batches = partition_tool_calls(response.tool_calls, self.registry)

            for batch in batches:
                if (
                    batch.concurrent
                    and len(batch.calls) > 1
                    and self._can_execute_batch_parallel(batch.calls)
                ):
                    result_by_id: dict[str, _ToolExecResult] = {}
                    calls_to_execute: list[ToolCallComplete] = []
                    for tc in batch.calls:
                        existing = self._control_existing_tool_result(tc.tool_id)
                        if existing is not None:
                            result, elapsed, is_unknown = existing
                            result_by_id[tc.tool_id] = _ToolExecResult(
                                tool_id=tc.tool_id,
                                tool_name=tc.tool_name,
                                result=result,
                                elapsed=elapsed,
                                is_unknown=is_unknown,
                            )
                        else:
                            self._transition_control_tool(
                                tc.tool_id, ToolCallStatus.RUNNING
                            )
                            calls_to_execute.append(tc)
                    for executed in await self._execute_batch_parallel(calls_to_execute):
                        result_by_id[executed.tool_id] = executed
                    batch_results = [result_by_id[tc.tool_id] for tc in batch.calls]
                    for br in batch_results:
                        if br.is_unknown:
                            consecutive_unknown += 1
                        else:
                            consecutive_unknown = 0
                        content = self._maybe_persist_or_truncate(
                            br.tool_id, br.result.output, br.tool_name
                        )
                        self._transition_control_tool(
                            br.tool_id,
                            self._tool_terminal_status(br.result),
                            result=br.result,
                            elapsed=br.elapsed,
                            result_path=self._control_result_path(
                                br.tool_id, br.result.output
                            ),
                        )
                        tool_results.append(
                            ToolResultBlock(
                                tool_use_id=br.tool_id,
                                content=content,
                                is_error=br.result.is_error,
                            )
                        )
                        yield ToolResultEvent(
                            tool_id=br.tool_id,
                            tool_name=br.tool_name,
                            output=br.result.output,
                            is_error=br.result.is_error,
                            elapsed=br.elapsed,
                        )
                else:
                    for tc in batch.calls:
                        result: ToolResult | None = None
                        elapsed = 0.0
                        is_unknown = False
                        reused = self._control_existing_tool_result(tc.tool_id)
                        if reused is not None:
                            result, elapsed, is_unknown = reused
                        else:
                            self._transition_control_tool(
                                tc.tool_id, ToolCallStatus.RUNNING
                            )

                        if result is None and self.hook_engine:
                            file_path = self._infer_file_path(tc.arguments)
                            hook_ctx = self._build_hook_context(
                                "pre_tool_use",
                                tool_name=tc.tool_name,
                                tool_args=tc.arguments,
                                file_path=file_path,
                                tool_call_id=self._control_tool_ids.get(
                                    tc.tool_id, tc.tool_id
                                ),
                            )
                            rejection = await self.hook_engine.run_pre_tool_hooks(hook_ctx)
                            for he in self._drain_hook_events():
                                yield he
                            if rejection is not None:
                                result = ToolResult(
                                    output=f"Hook rejected: {rejection.reason}",
                                    is_error=True,
                                )
                                content = self._maybe_persist_or_truncate(
                                    tc.tool_id, result.output, tc.tool_name
                                )
                                self._transition_control_tool(
                                    tc.tool_id,
                                    ToolCallStatus.DENIED,
                                    result=result,
                                    elapsed=0.0,
                                    result_path=self._control_result_path(
                                        tc.tool_id, result.output
                                    ),
                                )
                                tool_results.append(
                                    ToolResultBlock(
                                        tool_use_id=tc.tool_id,
                                        content=content,
                                        is_error=True,
                                    )
                                )
                                yield ToolResultEvent(
                                    tool_id=tc.tool_id,
                                    tool_name=tc.tool_name,
                                    output=result.output,
                                    is_error=True,
                                    elapsed=0.0,
                                )
                                continue

                        if result is None:
                            async for item in self._execute_tool(tc):
                                if isinstance(item, RuntimeEvent):
                                    yield item
                                else:
                                    result, elapsed, is_unknown = item

                        if result is None:
                            result = ToolResult(output="Error: no result from tool", is_error=True)

                        if is_unknown:
                            consecutive_unknown += 1
                        else:
                            consecutive_unknown = 0

                        if reused is None and self.hook_engine:
                            file_path = self._infer_file_path(tc.arguments)
                            hook_ctx = self._build_hook_context(
                                "post_tool_use",
                                tool_name=tc.tool_name,
                                tool_args=tc.arguments,
                                file_path=file_path,
                                tool_call_id=self._control_tool_ids.get(
                                    tc.tool_id, tc.tool_id
                                ),
                            )
                            await self.hook_engine.run_hooks("post_tool_use", hook_ctx)
                            for he in self._drain_hook_events():
                                yield he

                        content = self._maybe_persist_or_truncate(
                            tc.tool_id, result.output, tc.tool_name
                        )
                        self._transition_control_tool(
                            tc.tool_id,
                            self._tool_terminal_status(result),
                            result=result,
                            elapsed=elapsed,
                            result_path=self._control_result_path(
                                tc.tool_id, result.output
                            ),
                        )
                        tool_results.append(
                            ToolResultBlock(
                                tool_use_id=tc.tool_id,
                                content=content,
                                is_error=result.is_error,
                            )
                        )
                        yield ToolResultEvent(
                            tool_id=tc.tool_id,
                            tool_name=tc.tool_name,
                            output=result.output,
                            is_error=result.is_error,
                            elapsed=elapsed,
                        )

            if consecutive_unknown >= 3:
                self._finish_control_step(
                    StepStatus.FAILED,
                    response=response,
                    error="Too many consecutive unknown tool calls",
                )
                yield ErrorEvent(
                    message="Agent terminated: too many consecutive unknown tool calls"
                )
                break

            exit_plan_called = any(
                tc.tool_name == "ExitPlanMode" for tc in response.tool_calls
            )
            conversation.add_tool_results_message(tool_results)
            self._finish_control_step(
                StepStatus.COMPLETED,
                response=response,
                event_payload={"tool_call_count": len(response.tool_calls)},
            )

            # Also cover callers that attach a recall task after the first send.
            self._consume_memory_recall(conversation)

            if exit_plan_called:
                yield TurnComplete(turn=iteration)
                yield LoopComplete(total_turns=iteration)
                break

            if self.hook_engine:
                ctx = self._build_hook_context("turn_end")
                await self.hook_engine.run_hooks("turn_end", ctx)
                for he in self._drain_hook_events():
                    yield he
            yield TurnComplete(turn=iteration)


    def _consume_mailbox(self, conversation: ConversationManager) -> list[MailboxEvent]:
        if not self.team_name or not self._team_manager:
            return []
        events: list[MailboxEvent] = []
        try:
            mailbox = self._team_manager.get_mailbox(self.team_name)
            if mailbox is None:
                return []
            messages = mailbox.consume(self.agent_id)
            for msg in messages:
                prefix = f"[Message from {msg.from_agent}]"
                if msg.message_type != "text":
                    prefix = f"[{msg.message_type} from {msg.from_agent}]"
                content = f"{prefix} {msg.content}"
                conversation.add_user_message(content)
                events.append(
                    MailboxEvent(
                        message_id=msg.id,
                        from_agent=msg.from_agent,
                        to_agent=msg.to_agent,
                        message_type=msg.message_type,
                        content=msg.content,
                        summary=msg.summary,
                    )
                )
        except Exception as e:
            log.debug("Mailbox consumption failed: %s", e)
        return events

    def _build_permission_description(self, tc: ToolCallComplete) -> str:
        """为 HITL 权限确认生成人类可读的操作描述。"""
        return PermissionChecker.describe_tool_action(tc.tool_name, tc.arguments)

    async def _execute_single_tool_direct(
        self, tc: ToolCallComplete
    ) -> _ToolExecResult:
        with self.tracing.span(
            "tool.execute",
            {
                **self._trace_attributes(),
                "tool.name": tc.tool_name,
                "tool.call_id": tc.tool_id,
                "tool.arguments": tc.arguments,
                "tool.concurrent": True,
            },
        ) as span:
            executed = await self._execute_single_tool_direct_untraced(tc)
            span.set_attributes(
                {
                    "tool.duration_ms": round(executed.elapsed * 1000, 3),
                    "tool.is_error": executed.result.is_error,
                    "tool.is_unknown": executed.is_unknown,
                    "tool.output": executed.result.output,
                }
            )
            if executed.result.is_error:
                span.set_error(executed.result.output)
            return executed

    async def _execute_single_tool_direct_untraced(
        self, tc: ToolCallComplete
    ) -> _ToolExecResult:
        tool = self.registry.get(tc.tool_name)
        start = time.monotonic()

        if tool is None:
            return _ToolExecResult(
                tool_id=tc.tool_id,
                tool_name=tc.tool_name,
                result=ToolResult(output=f"Error: unknown tool '{tc.tool_name}'", is_error=True),
                elapsed=time.monotonic() - start,
                is_unknown=True,
            )

        if not self.registry.is_enabled(tc.tool_name):
            return _ToolExecResult(
                tool_id=tc.tool_id,
                tool_name=tc.tool_name,
                result=ToolResult(output=f"Error: tool '{tc.tool_name}' is disabled", is_error=True),
                elapsed=time.monotonic() - start,
                is_unknown=False,
            )

        try:
            params = tool.params_model.model_validate(tc.arguments)
            result = await self._execute_registered_tool(tc.tool_name, params)
        except ValidationError as e:
            result = ToolResult(output=f"Parameter validation error: {e}", is_error=True)
        except Exception as e:
            result = ToolResult(output=f"Tool execution error: {e}", is_error=True)

        self._snapshot_for_recovery(tc, result)

        return _ToolExecResult(
            tool_id=tc.tool_id,
            tool_name=tc.tool_name,
            result=result,
            elapsed=time.monotonic() - start,
            is_unknown=False,
        )


    async def _execute_batch_parallel(
        self, calls: list[ToolCallComplete]
    ) -> list[_ToolExecResult]:
        tasks = [self._execute_single_tool_direct(tc) for tc in calls]
        return list(await asyncio.gather(*tasks))

    def _can_execute_batch_parallel(self, calls: list[ToolCallComplete]) -> bool:
        """Use the direct path only when it cannot bypass a guard.

        Permission prompts and hook notifications need the event-yielding serial
        path. Already-approved calls can still run concurrently when no hooks
        are installed; every call is checked before any one starts executing.
        """
        if self.hook_engine is not None:
            return False
        if self.permission_checker is None:
            return True
        for tc in calls:
            tool = self.registry.get(tc.tool_name)
            if tool is None or not self.registry.is_enabled(tc.tool_name):
                return False
            with self.tracing.span(
                "permission.evaluate",
                {
                    **self._trace_attributes(),
                    "tool.name": tc.tool_name,
                    "tool.call_id": tc.tool_id,
                    "permission.preflight": True,
                },
            ) as span:
                decision = self.permission_checker.check(tool, tc.arguments)
                span.set_attributes({
                    "permission.effect": decision.effect,
                    "permission.reason": decision.reason,
                })
            if decision.effect != "allow":
                return False
        return True

    async def _execute_tool(
        self, tc: ToolCallComplete
    ) -> AsyncIterator[RuntimeEvent | tuple[ToolResult, float, bool]]:
        trace_context = self.tracing.span(
            "tool.execute",
            {
                **self._trace_attributes(),
                "tool.name": tc.tool_name,
                "tool.call_id": tc.tool_id,
                "tool.arguments": tc.arguments,
                "tool.concurrent": False,
            },
        )
        span = trace_context.__enter__()
        try:
            async for item in self._execute_tool_untraced(tc):
                if not isinstance(item, RuntimeEvent):
                    result, elapsed, is_unknown = item
                    span.set_attributes(
                        {
                            "tool.duration_ms": round(elapsed * 1000, 3),
                            "tool.is_error": result.is_error,
                            "tool.is_unknown": is_unknown,
                            "tool.output": result.output,
                        }
                    )
                    if result.is_error:
                        span.set_error(result.output)
                yield item
        except BaseException:
            trace_context.__exit__(*sys.exc_info())
            raise
        else:
            trace_context.__exit__(None, None, None)

    async def _execute_tool_untraced(
        self, tc: ToolCallComplete
    ) -> AsyncIterator[RuntimeEvent | tuple[ToolResult, float, bool]]:
        tool = self.registry.get(tc.tool_name)
        start = time.monotonic()
        is_unknown = False

        if tool is None:
            result = ToolResult(
                output=f"Error: unknown tool '{tc.tool_name}'", is_error=True
            )
            is_unknown = True
            elapsed = time.monotonic() - start
            yield result, elapsed, is_unknown
            return

        if not self.registry.is_enabled(tc.tool_name):
            result = ToolResult(
                output=f"Error: tool '{tc.tool_name}' is disabled in current mode",
                is_error=True,
            )
            elapsed = time.monotonic() - start
            yield result, elapsed, is_unknown
            return

        # 权限检查
        if self.permission_checker:
            with self.tracing.span(
                "permission.evaluate",
                {
                    **self._trace_attributes(),
                    "tool.name": tc.tool_name,
                    "tool.call_id": tc.tool_id,
                },
            ) as permission_span:
                decision = self.permission_checker.check(tool, tc.arguments)
                permission_span.set_attributes(
                    {
                        "permission.effect": decision.effect,
                        "permission.reason": decision.reason,
                    }
                )

            if decision.effect == "deny":
                result = ToolResult(
                    output=f"Permission denied: {decision.reason}",
                    is_error=True,
                )
                elapsed = time.monotonic() - start
                yield result, elapsed, is_unknown
                return

            if decision.effect == "ask":
                loop = asyncio.get_running_loop()
                future: asyncio.Future[PermissionResponse] = loop.create_future()
                desc = self._build_permission_description(tc)
                await self._run_permission_request_hook(tc, desc)
                self._finish_control_step(
                    StepStatus.BLOCKED,
                    event_payload={
                        "reason": "permission",
                        "tool_name": tc.tool_name,
                    },
                )
                self._finish_control_run(RunStatus.BLOCKED)
                # 向调用方 yield 权限请求事件，由调用方处理
                yield PermissionRequest(
                    tool_name=tc.tool_name,
                    description=desc,
                    future=future,
                )
                with self.tracing.span(
                    "permission.wait",
                    {
                        **self._trace_attributes(),
                        "tool.name": tc.tool_name,
                        "tool.call_id": tc.tool_id,
                    },
                ) as wait_span:
                    response = await self.execution_controller.wait(
                        future,
                        token=self.cancellation_token,
                        timeout=self.execution_controller.limits.permission_timeout,
                        operation=f"permission response for {tc.tool_name}",
                    )
                    wait_span.set_attributes({"permission.response": response.value})
                yield PermissionDecisionEvent(
                    tool_name=tc.tool_name,
                    response=response.value,
                )
                self._finish_control_run(RunStatus.RUNNING)
                self._finish_control_step(StepStatus.RUNNING)

                if response == PermissionResponse.DENY:
                    result = ToolResult(
                        output="Permission denied: 用户拒绝了此操作",
                        is_error=True,
                    )
                    elapsed = time.monotonic() - start
                    yield result, elapsed, is_unknown
                    return

                if response in (PermissionResponse.ALLOW_SESSION, PermissionResponse.ALLOW_ALWAYS):
                    permission_name = tool.permission_name
                    self.permission_checker.add_session_allow(permission_name, tc.arguments)

        tool_started = False
        try:
            params = tool.params_model.model_validate(tc.arguments)
            tool_started = True
            result = await self._execute_registered_tool(tc.tool_name, params)
        except ValidationError as e:
            result = ToolResult(
                output=f"Parameter validation error: {e}", is_error=True
            )
        except Exception as e:
            result = ToolResult(
                output=f"Tool execution error: {e}", is_error=True
            )

        if tool_started:
            await self._run_tool_lifecycle_hooks(tc, result)

        self._snapshot_for_recovery(tc, result)

        elapsed = time.monotonic() - start
        yield result, elapsed, is_unknown

    def _snapshot_for_recovery(
        self, tc: ToolCallComplete, result: ToolResult
    ) -> None:
        """捕获 ReadFile 刚交给模型的内容，以便 Layer 2 压缩对话后
        auto_compact 能重新附加这些数据。每次 ReadFile 多一次磁盘读取，
        比从 tool 输出中反向解析行号要划算。
        """
        if result.is_error or tc.tool_name != "ReadFile":
            return
        path = tc.arguments.get("file_path") if isinstance(tc.arguments, dict) else None
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                content = fh.read()
        except OSError:
            return
        self.recovery_state.record_file_read(path, content)

    async def _extract_memories(
        self, conversation: ConversationManager
    ) -> None:
        """触发记忆提取，对齐 Go 版 inProgress + pendingContext 合并策略。

        当提取正在进行时，新的触发不会启动并发提取，而是标记 _pending_extraction。
        当前提取完成后检查该标志，如果有 pending 则立即执行一次尾随提取，
        防止多个触发器同时执行导致重复提取。
        """
        if not self.memory_manager:
            return

        # 合并策略：正在提取时暂存新请求，等当前提取完成后尾随执行
        if self._extracting:
            log.debug("[extractMemories] extraction in progress — stashing for trailing run")
            self._pending_extraction = True
            return

        self._extracting = True
        try:
            await self.memory_manager.extract(
                self.client, conversation, self.protocol
            )
        except Exception as e:
            log.debug("Memory extraction failed: %s", e)
        finally:
            self._extracting = False
            # 检查是否有尾随提取请求
            if self._pending_extraction:
                self._pending_extraction = False
                log.debug("[extractMemories] running trailing extraction for stashed context")
                # 递归调用自身处理尾随请求
                await self._extract_memories(conversation)

    async def manual_compact(
        self, conversation: ConversationManager
    ) -> CompactNotification | ErrorEvent:
        # auto_compact 会用摘要替换 conversation.history，所有 tool-result 内容
        # （原始或已替换的）都将被丢弃。这里跳过 apply_tool_result_budget —
        # 它在主循环中的唯一目的是为 LLM 调用生成 api_conv，而本路径不需要
        # 发起看到替换结果的 LLM 调用（auto_compact 内部的摘要调用操作的是原始对话）。
        result = await self._auto_compact_with_trace(conversation, manual=True)
        if isinstance(result, CompactEvent):
            env_context = build_environment_context(
            self.work_dir, self.active_skills, self._skill_catalog, self._agent_catalog
        )
            conversation.inject_environment(env_context)
            memory_content = self.memory_manager.load() if self.memory_manager else ""
            conversation.inject_long_term_memory(
                self.instructions_content, memory_content
            )
            return CompactNotification(
                before_tokens=result.before_tokens,
                message=f"上下文已压缩（压缩前 {result.before_tokens:,} tokens）",
                boundary=result.boundary,
            )
        return ErrorEvent(message=result or "压缩失败：对话历史为空或未达到压缩条件")

    async def run_to_completion(
        self, task: str, conversation: ConversationManager | None = None,
        event_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> str:
        if conversation is None:
            conversation = ConversationManager()
        if self._owns_cancellation_token:
            self.cancellation_token = CancellationToken()
        self._event_sequence = 0
        self._start_control_run(conversation, input_text=task)
        run_id = self._current_run_id
        trace_context = self.tracing.span(
            "agent.run",
            {
                "session.id": self.session_id,
                "run.id": run_id or "",
                "agent.id": self.agent_id,
                "agent.parent_id": self.parent_id or "",
                "provider.name": self.provider_name or "",
                "model.name": self.model or "",
                "input": task,
                "agent.non_interactive": True,
            },
            trace_id=self._current_trace_id,
        )
        run_span = trace_context.__enter__()
        completed = False
        run_status = "running"
        try:
            result, completed = await self._run_to_completion_loop(
                task, conversation, event_callback
            )
        except asyncio.CancelledError:
            run_status = "cancelled"
            self.cancel()
            self._cancel_control_tools()
            self._finish_control_step(StepStatus.CANCELLED, error="Agent run cancelled")
            self._finish_control_run(RunStatus.CANCELLED, error="Agent run cancelled")
            raise
        except Exception as exc:
            run_status = "failed"
            message = f"{type(exc).__name__}: {exc}"
            await self._run_error_hook(exc)
            self._finish_control_step(StepStatus.FAILED, error=message)
            self._finish_control_run(RunStatus.FAILED, error=message)
            raise
        else:
            if completed:
                run_status = "completed"
                self._finish_control_run(RunStatus.COMPLETED)
            else:
                run_status = "interrupted"
                self._finish_control_step(
                    StepStatus.INTERRUPTED, error="Agent loop ended without completion"
                )
                self._finish_control_run(
                    RunStatus.INTERRUPTED, error="Agent loop ended without completion"
                )
            return result
        finally:
            run_span.set_attributes(
                {
                    "run.status": run_status,
                    "usage.input_tokens": self.total_input_tokens,
                    "usage.output_tokens": self.total_output_tokens,
                }
            )
            trace_context.__exit__(*sys.exc_info())

    async def _run_to_completion_loop(
        self, task: str, conversation: ConversationManager | None = None,
        event_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[str, bool]:
        if conversation is None:
            conversation = ConversationManager()

            env_context = build_environment_context(
                self.work_dir, self.active_skills, self._skill_catalog, self._agent_catalog
            )
            conversation.inject_environment(env_context)

            if self.instructions_content:
                memory_content = self.memory_manager.load() if self.memory_manager else ""
                conversation.inject_long_term_memory(
                    self.instructions_content, memory_content
                )

        if task:
            conversation.add_user_message(task)

        if self.hook_engine:
            await self.hook_engine.run_hooks(
                "session_start", self._build_hook_context("session_start")
            )

        tools = self.registry.get_all_schemas(self.protocol)

        log.info(
            "[run_to_completion] agent=%s tools=%d names=%s coordinator=%s",
            self.agent_id,
            len(tools),
            [t["name"] for t in tools][:10],
            self.coordinator_mode,
        )

        last_text = ""
        completed = False

        iteration = 0
        while True:
            iteration += 1
            if self.max_iterations > 0 and iteration > self.max_iterations:
                break
            if self.hook_engine:
                ctx = self._build_hook_context("turn_start")
                await self.hook_engine.run_hooks("turn_start", ctx)

            for mailbox_event in self._consume_mailbox(conversation):
                prepared = self._prepare_event(mailbox_event)
                if event_callback:
                    event_callback(
                        {
                            "type": "runtime_event",
                            "event": prepared.to_envelope().to_dict(),
                        }
                    )
            if self.notification_fn:
                for note in self.notification_fn():
                    conversation.add_system_reminder(note)

            if self.hook_engine:
                await self.hook_engine.run_hooks(
                    "pre_send", self._build_hook_context("pre_send")
                )

            hook_prompts = (
                self.hook_engine.get_prompt_messages() if self.hook_engine else None
            )
            system = build_system_prompt(
                hook_prompts=hook_prompts,
                coordinator_mode=self.coordinator_mode,
            )

            # Build a model-only budget view without mutating durable history.
            api_conversation, pre_compact_records = apply_tool_result_budget(
                conversation,
                self.session_dir,
                self.replacement_state,
                persist_callback=self._persist_tool_result,
            )
            if pre_compact_records:
                append_replacement_records(self.session_dir, pre_compact_records)

            compact_result = await self._auto_compact_with_trace(api_conversation)
            if isinstance(compact_result, CompactEvent):
                conversation.replace_history(api_conversation.get_messages())
                conversation.inject_environment(env_context)

            deferred_names = self.registry.get_deferred_tool_names()
            if deferred_names:
                conversation.add_system_reminder(
                    "The following deferred tools are available via ToolSearch. "
                    "Their schemas are NOT loaded - use ToolSearch with "
                    'query "select:<name>[,<name>...]" to load tool schemas before calling them:\n'
                    + "\n".join(deferred_names)
                )

            # Rebuild after compaction or deferred-tool reminders.
            api_conversation, _new_records = apply_tool_result_budget(
                conversation,
                self.session_dir,
                self.replacement_state,
                persist_callback=self._persist_tool_result,
            )
            if _new_records:
                append_replacement_records(self.session_dir, _new_records)

            self._start_control_step(iteration)
            collector = StreamCollector()
            async for _event in self._consume_llm_stream(
                collector, api_conversation, self._system_with_todo_progress(system), tools
            ):
                pass

            response = collector.response
            if self.hook_engine:
                await self.hook_engine.run_hooks(
                    "post_receive",
                    self._build_hook_context(
                        "post_receive", message=response.text
                    ),
                )
            self.total_input_tokens += response.input_tokens
            self.total_output_tokens += response.output_tokens

            if event_callback:
                event_callback({
                    "type": "usage",
                    "usage": {
                        "inputTokens": self.total_input_tokens,
                        "outputTokens": self.total_output_tokens,
                    },
                })

            if response.text:
                last_text = response.text
                if event_callback:
                    event_callback({
                        "type": "stream_text",
                        "text": response.text,
                    })

            log.info(
                "[run_to_completion] agent=%s iter=%d tool_calls=%d text_len=%d stop=%s",
                self.agent_id, iteration, len(response.tool_calls),
                len(response.text), response.stop_reason,
            )

            if not response.tool_calls:
                conversation.add_assistant_message(response.text)
                if self.hook_engine:
                    await self.hook_engine.run_hooks(
                        "turn_end", self._build_hook_context("turn_end")
                    )
                    await self.hook_engine.run_hooks(
                        "session_end", self._build_hook_context("session_end")
                    )
                if self.file_history is not None:
                    summary = response.text[:60] + "..." if len(response.text) > 60 else response.text
                    self.file_history.make_snapshot(len(conversation.history), summary)
                self._finish_control_step(
                    StepStatus.COMPLETED,
                    response=response,
                    event_payload={"stop_reason": response.stop_reason},
                )
                completed = True
                break

            loop_decision = self._check_tool_loop(response.tool_calls)
            if loop_decision is not None:
                self._finish_control_step(
                    StepStatus.FAILED,
                    response=response,
                    error=loop_decision.reason,
                    event_payload={"loop_guard": True},
                )
                break

            tool_uses = [
                ToolUseBlock(
                    tool_use_id=tc.tool_id,
                    tool_name=tc.tool_name,
                    arguments=tc.arguments,
                )
                for tc in response.tool_calls
            ]
            conversation.add_assistant_message(response.text, tool_uses)
            # assistant 回复已在历史中，锚定实际用量；下一轮迭代只需对
            # 下方追加的 tool results 做字符估算。
            conversation.record_usage_anchor(
                response.input_tokens,
                response.output_tokens,
                response.cache_read,
                response.cache_creation,
            )
            self._register_control_tool_calls(response.tool_calls)

            tool_results: list[ToolResultBlock] = []
            for tc in response.tool_calls:
                if event_callback:
                    event_callback({
                        "type": "tool_use",
                        "toolName": tc.tool_name,
                        "args": tc.arguments,
                    })
                existing = self._control_existing_tool_result(tc.tool_id)
                if existing is not None:
                    result, elapsed, _ = existing
                else:
                    self._transition_control_tool(tc.tool_id, ToolCallStatus.RUNNING)
                    started = time.monotonic()
                    result = await self._execute_tool_noninteractive(tc)
                    elapsed = time.monotonic() - started
                content = self._maybe_persist_or_truncate(
                    tc.tool_id, result.output, tc.tool_name
                )
                self._transition_control_tool(
                    tc.tool_id,
                    self._tool_terminal_status(result),
                    result=result,
                    elapsed=elapsed,
                    result_path=self._control_result_path(tc.tool_id, result.output),
                )
                tool_results.append(
                    ToolResultBlock(
                        tool_use_id=tc.tool_id,
                        content=content,
                        is_error=result.is_error,
                    )
                )

            conversation.add_tool_results_message(tool_results)
            self._finish_control_step(
                StepStatus.COMPLETED,
                response=response,
                event_payload={"tool_call_count": len(response.tool_calls)},
            )

            if self.hook_engine:
                ctx = self._build_hook_context("turn_end")
                await self.hook_engine.run_hooks("turn_end", ctx)

        return last_text, completed

    async def _execute_tool_noninteractive(
        self, tc: ToolCallComplete
    ) -> ToolResult:
        started = time.monotonic()
        with self.tracing.span(
            "tool.execute",
            {
                **self._trace_attributes(),
                "tool.name": tc.tool_name,
                "tool.call_id": tc.tool_id,
                "tool.arguments": tc.arguments,
                "tool.non_interactive": True,
            },
        ) as span:
            result = await self._execute_tool_noninteractive_untraced(tc)
            span.set_attributes(
                {
                    "tool.duration_ms": round(
                        (time.monotonic() - started) * 1000, 3
                    ),
                    "tool.is_error": result.is_error,
                    "tool.output": result.output,
                }
            )
            if result.is_error:
                span.set_error(result.output)
            return result

    async def _execute_tool_noninteractive_untraced(
        self, tc: ToolCallComplete
    ) -> ToolResult:
        tool = self.registry.get(tc.tool_name)

        if tool is None:
            return ToolResult(
                output=f"Error: unknown tool '{tc.tool_name}'", is_error=True
            )

        if not self.registry.is_enabled(tc.tool_name):
            return ToolResult(
                output=f"Error: tool '{tc.tool_name}' is disabled",
                is_error=True,
            )

        if self.hook_engine:
            file_path = self._infer_file_path(tc.arguments)
            hook_ctx = self._build_hook_context(
                "pre_tool_use",
                tool_name=tc.tool_name,
                tool_args=tc.arguments,
                file_path=file_path,
                tool_call_id=self._control_tool_ids.get(tc.tool_id, tc.tool_id),
            )
            rejection = await self.hook_engine.run_pre_tool_hooks(hook_ctx)
            if rejection is not None:
                return ToolResult(
                    output=f"Hook rejected: {rejection.reason}",
                    is_error=True,
                )

        if self.permission_checker:
            with self.tracing.span(
                "permission.evaluate",
                {
                    **self._trace_attributes(),
                    "tool.name": tc.tool_name,
                    "tool.call_id": tc.tool_id,
                    "permission.non_interactive": True,
                },
            ) as permission_span:
                decision = self.permission_checker.check(tool, tc.arguments)
                permission_span.set_attributes(
                    {
                        "permission.effect": decision.effect,
                        "permission.reason": decision.reason,
                    }
                )
            if decision.effect == "deny":
                return ToolResult(
                    output=f"Permission denied: {decision.reason}",
                    is_error=True,
                )
            if decision.effect == "ask":
                await self._run_permission_request_hook(
                    tc, self._build_permission_description(tc)
                )
                if self.permission_mode == PermissionMode.BYPASS:
                    pass  # BYPASS 模式自动批准
                else:
                    return ToolResult(
                        output="Permission denied: non-interactive agent cannot prompt user",
                        is_error=True,
                    )

        tool_started = False
        try:
            params = tool.params_model.model_validate(tc.arguments)
            tool_started = True
            result = await self._execute_registered_tool(tc.tool_name, params)
        except ValidationError as e:
            result = ToolResult(
                output=f"Parameter validation error: {e}", is_error=True
            )
        except Exception as e:
            result = ToolResult(
                output=f"Tool execution error: {e}", is_error=True
            )

        if tool_started:
            await self._run_tool_lifecycle_hooks(tc, result)

        if self.hook_engine:
            file_path = self._infer_file_path(tc.arguments)
            hook_ctx = self._build_hook_context(
                "post_tool_use",
                tool_name=tc.tool_name,
                tool_args=tc.arguments,
                file_path=file_path,
                tool_call_id=self._control_tool_ids.get(tc.tool_id, tc.tool_id),
            )
            await self.hook_engine.run_hooks("post_tool_use", hook_ctx)

        return result

    def _maybe_persist_or_truncate(
        self, tool_use_id: str, text: str, tool_name: str | None = None
    ) -> str:
        from valecode.context.manager import (
            SINGLE_RESULT_CHAR_LIMIT,
            make_persisted_preview,
        )

        if len(text) > SINGLE_RESULT_CHAR_LIMIT:
            fp = self._persist_tool_result(tool_use_id, text, self.session_dir)
            return make_persisted_preview(text, fp)
        registration = (
            self.registry.get_registration(tool_name) if tool_name is not None else None
        )
        output_limit = (
            registration.output_limit if registration is not None else MAX_OUTPUT_CHARS
        )
        if len(text) > output_limit:
            return text[:output_limit] + "\n… (output truncated)"
        return text
