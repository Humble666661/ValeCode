from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from valecode.commands.registry import Command, CommandContext, CommandType

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from valecode.agents.trace import TraceManager
    from valecode.persistence import RunTraceState


@dataclass(frozen=True)
class _DisplayNode:
    key: str
    label: str
    parent_key: str | None
    agent_type: str
    status: str
    input_tokens: int
    output_tokens: int
    tool_call_count: int
    elapsed: str


def _format_elapsed(start: float, end: float | None) -> str:
    elapsed = (end or time.monotonic()) - start
    if elapsed >= 60:
        return f"{elapsed / 60:.1f}m"
    return f"{elapsed:.0f}s"


def _status_icon(status: str) -> str:
    return {
        "pending": "○",
        "running": "⏳",
        "completed": "✓",
        "failed": "✗",
        "interrupted": "!",
        "cancelled": "−",
        "blocked": "⊘",
    }.get(status, "?")


def _format_persisted_elapsed(node: RunTraceState) -> str:
    try:
        start = datetime.fromisoformat(node.started_at or node.created_at)
        end = (
            datetime.fromisoformat(node.completed_at)
            if node.completed_at
            else datetime.now(UTC)
        )
        elapsed = max(0.0, (end - start).total_seconds())
    except (TypeError, ValueError):
        return "?"
    if elapsed >= 60:
        return f"{elapsed / 60:.1f}m"
    return f"{elapsed:.0f}s"


def _collect_nodes(
    ctx: CommandContext, trace_manager: TraceManager,
) -> list[_DisplayNode]:
    live_nodes = trace_manager.list_nodes()
    persisted: list[RunTraceState] = []
    session_id = getattr(ctx.session, "session_id", "")
    run_store = getattr(ctx.session_manager, "run_store", None)
    if session_id and run_store is not None:
        try:
            persisted = run_store.list_trace_nodes(session_id=session_id)
        except Exception:
            log.exception("Unable to load persisted Agent trace")
            persisted = []

    by_run = {node.run_id: node for node in persisted}
    persisted_agent_ids = {node.agent_id for node in persisted if node.agent_id}
    latest_run_by_agent: dict[str, str] = {}
    display: list[_DisplayNode] = []
    for node in persisted:
        if node.agent_id:
            latest_run_by_agent[node.agent_id] = node.run_id
        parent = by_run.get(node.parent_run_id or "")
        display.append(
            _DisplayNode(
                key=node.run_id,
                label=node.agent_id or node.run_id,
                parent_key=parent.run_id if parent is not None else None,
                agent_type=node.agent_type,
                status=node.status.value,
                input_tokens=node.input_tokens,
                output_tokens=node.output_tokens,
                tool_call_count=node.tool_call_count,
                elapsed=_format_persisted_elapsed(node),
            )
        )

    for node in live_nodes:
        if node.agent_id in persisted_agent_ids:
            continue
        parent_key = (
            latest_run_by_agent.get(node.parent_id or "")
            or (f"live:{node.parent_id}" if node.parent_id else None)
        )
        display.append(
            _DisplayNode(
                key=f"live:{node.agent_id}",
                label=node.agent_id,
                parent_key=parent_key,
                agent_type=node.agent_type,
                status=node.status,
                input_tokens=node.input_tokens,
                output_tokens=node.output_tokens,
                tool_call_count=node.tool_call_count,
                elapsed=_format_elapsed(node.start_time, node.end_time),
            )
        )
    return display


def create_trace_command(trace_manager: TraceManager, lead_agent_id: str = "") -> Command:


    async def handler(ctx: CommandContext) -> None:
        nodes = _collect_nodes(ctx, trace_manager)
        if not nodes:
            ctx.ui.add_system_message("没有 Agent 追踪记录")
            return

        by_key = {node.key: node for node in nodes}
        parent_map: dict[str | None, list[_DisplayNode]] = {}
        for n in nodes:
            parent_map.setdefault(n.parent_key, []).append(n)

        lines = ["Agent 追踪树:"]
        visited: set[str] = set()

        def _render(parent_id: str | None, indent: int) -> None:
            children = parent_map.get(parent_id, [])
            for n in children:
                if n.key in visited:
                    continue
                visited.add(n.key)
                icon = _status_icon(n.status)
                tokens = (
                    f"↑{n.input_tokens} ↓{n.output_tokens}"
                    if n.input_tokens or n.output_tokens
                    else ""
                )
                tools = f" 工具×{n.tool_call_count}" if n.tool_call_count else ""
                prefix = "  " * indent
                lines.append(
                    f"{prefix}{icon} [{n.label[:8]}] {n.agent_type} — "
                    f"{n.status} ({n.elapsed}) {tokens}{tools}".rstrip()
                )
                _render(n.key, indent + 1)

        roots = [
            n for n in nodes
            if n.parent_key is None or n.parent_key not in by_key
        ]
        if not roots:
            roots = nodes[:1]

        if lead_agent_id:
            lines.append(f"  Lead: {lead_agent_id[:8]}")

        for root in roots:
            if root.key in visited:
                continue
            visited.add(root.key)
            icon = _status_icon(root.status)
            tokens = (
                f"↑{root.input_tokens} ↓{root.output_tokens}"
                if root.input_tokens or root.output_tokens
                else ""
            )
            tools = f" 工具×{root.tool_call_count}" if root.tool_call_count else ""
            lines.append(
                f"  {icon} [{root.label[:8]}] {root.agent_type} — "
                f"{root.status} ({root.elapsed}) {tokens}{tools}".rstrip()
            )
            _render(root.key, 2)

        total_in = sum(n.input_tokens for n in nodes)
        total_out = sum(n.output_tokens for n in nodes)
        total_tools = sum(n.tool_call_count for n in nodes)
        lines.append(
            f"\n  合计: {len(nodes)} runs, ↑{total_in} ↓{total_out}, "
            f"工具×{total_tools}"
        )

        ctx.ui.add_system_message("\n".join(lines))

    return Command(
        name="trace",
        description="查看智能体父子追踪树",
        type=CommandType.LOCAL,
        handler=handler,
        aliases=["tree"],
        usage="/trace",
    )
