from __future__ import annotations

import inspect

from valecode.commands.registry import Command, CommandContext, CommandType


async def handle_exit(ctx: CommandContext) -> None:
    """通过当前界面提供的退出回调安全关闭应用。"""
    exit_app = ctx.config.get("exit_app")
    if exit_app is None:
        ctx.ui.add_system_message("当前运行模式不支持 /exit")
        return

    result = exit_app()
    if inspect.isawaitable(result):
        await result


EXIT_COMMAND = Command(
    name="exit",
    aliases=["quit", "q"],
    description="安全退出 ValeCode",
    usage="/exit",
    type=CommandType.LOCAL_UI,
    handler=handle_exit,
)
