"""Explicit workflow entry point; child Agents never receive this tool."""
from __future__ import annotations

import json
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field
from valecode.tools.base import Tool, ToolResult


class WorkflowParams(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    action: Literal["run", "list", "status", "resume", "retry", "cancel"]
    path: str = Field(default="", max_length=1000)
    instance_id: str = Field(default="", max_length=100)
    node_id: str = Field(default="", max_length=64)
    confirm_retry: bool = False


class WorkflowTool(Tool):
    name = "Workflow"
    description = "仅在用户明确要求执行工作流时使用。run 加载工作区 YAML；list/status 查看；resume 保留已完成节点；retry 仅对失败节点且需用户明确同意重复副作用 confirm_retry=true；cancel 中断当前执行。不会自动重放写节点。"
    params_model = WorkflowParams
    category = "command"
    uses_global_capacity = False
    execution_timeout = 0

    def __init__(self, runtime):
        self.runtime = runtime

    async def execute(self, params):
        try:
            if params.action == "list":
                result = self.runtime.store.list(*self.runtime.scope(), "workflow")
            elif params.action == "status":
                result = self.runtime.status(params.instance_id)
            elif params.action == "cancel":
                await self.runtime.cancel()
                result = {"status": "interrupted", "note": "中断节点需明确确认后才能重试"}
            else:
                if self.runtime._running:
                    raise ValueError("已有工作流正在执行")
                identity = params.instance_id
                if params.action == "run":
                    identity = self.runtime.create(self.runtime.load(params.path))
                elif params.action == "retry":
                    self.runtime.retry(identity, params.node_id, params.confirm_retry)
                result = await self.runtime.run(identity)
            return ToolResult(json.dumps(result, ensure_ascii=False))
        except (ValueError, KeyError, OSError, RuntimeError) as exc:
            return ToolResult(str(exc), is_error=True)
