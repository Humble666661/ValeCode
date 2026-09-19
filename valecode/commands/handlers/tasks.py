from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from valecode.commands.registry import Command, CommandContext, CommandType

if TYPE_CHECKING:
    from valecode.agents.task_manager import TaskManager
    from valecode.persistence import TaskState


def _format_elapsed(start: float, end: float | None) -> str:
    elapsed = (end or time.monotonic()) - start
    if elapsed >= 60:
        return f"{elapsed / 60:.1f}m"
    return f"{elapsed:.0f}s"


def _format_status(status: str) -> str:
    icons = {
        "queued": "○",
        "blocked": "⊘",
        "leased": "⏳",
        "running": "⏳",
        "retrying": "↻",
        "completed": "✓",
        "succeeded": "✓",
        "failed": "✗",
        "cancelled": "⊘",
    }
    return f"{icons.get(status, '?')} {status}"


def _format_persisted_elapsed(task: TaskState) -> str:
    try:
        start = datetime.fromisoformat(task.created_at)
        end = (
            datetime.fromisoformat(task.completed_at)
            if task.completed_at
            else datetime.now(UTC)
        )
        elapsed = max(0.0, (end - start).total_seconds())
    except (TypeError, ValueError):
        return "?"
    if elapsed >= 60:
        return f"{elapsed / 60:.1f}m"
    return f"{elapsed:.0f}s"


def _persisted_result(task: TaskState) -> str:
    if isinstance(task.result, dict) and "output" in task.result:
        return str(task.result["output"])
    if task.result is not None:
        return str(task.result)
    return task.error or ""


def _current_session_id(ctx: CommandContext) -> str | None:
    value = getattr(ctx.session, "session_id", None)
    return value if isinstance(value, str) and value else None


def create_tasks_handler(task_manager: TaskManager):


    async def handler(ctx: CommandContext) -> None:
        args = ctx.args.strip()
        parts = args.split(maxsplit=1) if args else []
        subcmd = parts[0] if parts else ""

        if subcmd == "info":
            if len(parts) < 2:
                ctx.ui.add_system_message("用法: /tasks info <task-id>")
                return
            task_id = parts[1].strip()
            bg = task_manager.get(task_id)
            if bg is not None:
                elapsed = _format_elapsed(bg.start_time, bg.end_time)
                lines = [
                    f"任务详情: {task_id}",
                    f"  名称:    {bg.name}",
                    f"  状态:    {_format_status(bg.status)}",
                    f"  耗时:    {elapsed}",
                    f"  Tokens:  ↑{bg.progress.input_tokens} ↓{bg.progress.output_tokens}",
                ]
                result = bg.result
                result_path = ""
            else:
                get_persisted = getattr(task_manager, "get_persisted", None)
                state = get_persisted(task_id) if callable(get_persisted) else None
                session_id = _current_session_id(ctx)
                if state is None or (
                    session_id is not None and state.session_id != session_id
                ):
                    ctx.ui.add_system_message(f"未找到任务: {task_id}")
                    return
                lines = [
                    f"任务详情: {task_id}",
                    f"  名称:    {state.input.get('name') or task_id}",
                    f"  状态:    {_format_status(state.status.value)}",
                    f"  耗时:    {_format_persisted_elapsed(state)}",
                    f"  尝试:    {state.attempt_count}/{state.max_attempts}",
                    f"  Tokens:  ↑{state.input_tokens} ↓{state.output_tokens}",
                ]
                result = _persisted_result(state)
                result_path = state.result_path or ""
            if result:
                result_preview = result[:2000]
                if len(result) > 2000:
                    result_preview += "\n... (truncated)"
                lines.append(f"  结果:\n{result_preview}")
            if result_path:
                lines.append(f"  完整结果: {result_path}")
            ctx.ui.add_system_message("\n".join(lines))
            return

        if subcmd == "cancel":
            if len(parts) < 2:
                ctx.ui.add_system_message("用法: /tasks cancel <task-id>")
                return
            task_id = parts[1].strip()
            bg = task_manager.get(task_id)
            if bg is None:
                get_persisted = getattr(task_manager, "get_persisted", None)
                state = get_persisted(task_id) if callable(get_persisted) else None
                session_id = _current_session_id(ctx)
                if state is not None and (
                    session_id is not None and state.session_id != session_id
                ):
                    ctx.ui.add_system_message(f"无法取消任务: {task_id}（不属于当前会话）")
                    return
            if task_manager.cancel(task_id):
                ctx.ui.add_system_message(f"已取消任务: {task_id}")
            else:
                ctx.ui.add_system_message(
                    f"无法取消任务: {task_id}（可能不存在或已完成）"
                )
            return

        # 默认：列出所有任务
        tasks = task_manager.list_tasks()
        live_ids = {task.id for task in tasks}
        list_persisted = getattr(task_manager, "list_persisted", None)
        persisted = (
            list_persisted(session_id=_current_session_id(ctx))
            if callable(list_persisted)
            else []
        )
        persisted = [task for task in persisted if task.id not in live_ids]
        if not tasks and not persisted:
            ctx.ui.add_system_message("没有后台任务")
            return

        lines = ["后台任务列表:"]
        for bg in tasks:
            elapsed = _format_elapsed(bg.start_time, bg.end_time)
            lines.append(
                f"  [{bg.id}] {bg.name:<20} {_format_status(bg.status):<14} {elapsed}"
            )
        for state in persisted:
            name = str(state.input.get("name") or state.id)
            lines.append(
                f"  [{state.id}] {name:<20} "
                f"{_format_status(state.status.value):<14} "
                f"{_format_persisted_elapsed(state)}"
            )
        ctx.ui.add_system_message("\n".join(lines))

    return handler


def create_tasks_command(task_manager: TaskManager) -> Command:
    return Command(
        name="tasks",
        description="管理后台任务",
        type=CommandType.LOCAL,
        handler=create_tasks_handler(task_manager),
        aliases=["task"],
        usage="/tasks [info|cancel] [task-id]",
    )
