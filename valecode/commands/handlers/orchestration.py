"""Shared explicit commands; never infer execution from ordinary conversation."""
from __future__ import annotations

import shlex
from valecode.commands.registry import Command, CommandType
from valecode.tools.workflow import WorkflowParams, WorkflowTool


def create_orchestration_command(runtime, name="workflow", tool_type=WorkflowTool, params_type=WorkflowParams):
    usage = f'/{name} run "工作区内 YAML 路径" | list | status <id> | resume <id> | retry <id> <节点> --confirm | cancel'
    async def handler(ctx):
        try:
            if getattr(ctx.session, "session_id", "") != runtime.scope()[0]:
                raise ValueError("没有匹配的活动会话")
            parts = shlex.split(ctx.args, posix=True)
            action = parts[0] if parts else "list"
            values = {"action": action}
            if action in {"list", "cancel"} and len(parts) <= 1:
                pass
            elif action == "run" and len(parts) == 2:
                values["path"] = parts[1]
            elif action in {"status", "resume"} and len(parts) == 2:
                values["instance_id"] = parts[1]
            elif action == "retry" and len(parts) == 4 and parts[-1] == "--confirm":
                values.update(instance_id=parts[1], node_id=parts[2], confirm_retry=True)
            else:
                raise ValueError(usage + "；重试可能重复写入，必须显式加 --confirm")
            changing = action in {"run", "resume", "retry"}
            if changing and (runtime._running or getattr(ctx.ui, "_streaming", False)):
                raise ValueError("请先等待当前执行结束")
            if changing:
                ctx.ui._streaming = True
                ctx.ui.add_system_message("开始执行；进度持续保存，可用 status 查看或 cancel 中断。")
            try:
                if action == "run":
                    identity = runtime.create(runtime.load(values["path"]))
                    ctx.ui.add_system_message(f"执行 ID：{identity}")
                    values = {"action": "resume", "instance_id": identity}
                result = await tool_type(runtime).execute(params_type(**values))
                ctx.ui.add_system_message(result.output)
            finally:
                if changing:
                    ctx.ui._streaming = False
        except (ValueError, KeyError, OSError) as exc:
            ctx.ui.add_system_message(str(exc))
    return Command(name=name, description="执行和恢复持久工作流" if name == "workflow" else "执行目标、独立验收与有界续跑",
        type=CommandType.LOCAL, handler=handler, usage=usage)
