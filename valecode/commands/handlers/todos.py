"""Show the current session's model-visible task progress."""

from __future__ import annotations

from valecode.commands.registry import Command, CommandContext, CommandType
from valecode.tools.todo_write import TodoWrite


async def handle_todos(ctx: CommandContext) -> None:
    agent = ctx.agent
    tool = agent.registry.get("TodoWrite") if agent is not None else None
    if not isinstance(tool, TodoWrite):
        ctx.ui.add_system_message("当前会话未启用任务进度。")
        return
    ctx.ui.add_system_message(tool.current_summary() or "暂无待办事项。")


TODOS_COMMAND = Command(
    name="todos",
    aliases=[],
    description="查看当前会话的任务进度",
    usage="/todos",
    type=CommandType.LOCAL,
    handler=handle_todos,
)
