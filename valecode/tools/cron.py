from __future__ import annotations

import json
from dataclasses import replace
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from valecode.tools.base import Tool, ToolResult


class CreateParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=128)
    prompt: str = Field(min_length=1, max_length=20000)
    subagent_type: str
    schedule_type: Literal["once", "interval", "cron"]
    schedule_spec: dict
    timezone: str = "UTC"


class EmptyParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


class IdParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schedule_id: str


class UpdateParams(IdParams):
    action: Literal["pause", "resume"]


class CronTool(Tool):
    category = "command"
    def __init__(self, runtime):
        self.runtime = runtime

    def session_id(self):
        session = self.runtime.agent_tool._parent_agent.session_id
        if not session:
            raise ValueError("A current session is required")
        return session


class CronCreate(CronTool):
    name = "CronCreate"
    description = "创建当前会话的持久定时 Agent 任务。仅会话打开时触发；默认权限、不会无人值守绕过审批。interval 提供 every_seconds（>=60），once 提供 ISO run_at，cron 提供五段 cron；timezone 使用 IANA 名称。"
    params_model = CreateParams
    uses_global_capacity = False

    async def execute(self, params):
        try:
            session_id = self.session_id()
            tool = self.runtime.agent_tool
            definition = tool._agent_loader.get(params.subagent_type)
            if definition is None or definition.isolation:
                raise ValueError("A known definition-based agent without worktree isolation is required")
            spec = tool._resume_spec(replace(definition, permission_mode="default"), None)
            result = self.runtime.store.create(session_id=session_id, name=params.name, prompt=params.prompt,
                work_dir=tool._parent_agent.work_dir, kind=params.schedule_type, spec=params.schedule_spec,
                timezone=params.timezone, agent_spec=spec)
            return ToolResult(json.dumps({"schedule_id": result["id"], "next_run": result["next_run"], "status": result["status"], "timezone": result["timezone"]}, ensure_ascii=False))
        except (ValueError, KeyError) as exc:
            return ToolResult(str(exc), is_error=True)


class CronList(CronTool):
    name = "CronList"
    description = "列出当前会话的持久定时计划与最近执行状态。"
    params_model = EmptyParams
    category = "read"

    async def execute(self, params):
        plans = self.runtime.store.list(self.session_id())
        with self.runtime.store.database.reader() as db:
            for plan in plans:
                latest = db.execute("SELECT t.id,t.status,t.error FROM schedule_occurrences o JOIN tasks t ON t.id=o.task_id WHERE o.schedule_id=? ORDER BY o.scheduled_for DESC LIMIT 1", (plan["id"],)).fetchone()
                plan["latest_task"] = dict(latest) if latest else None
                plan.pop("agent", None)
        return ToolResult(json.dumps(plans, ensure_ascii=False))


class CronUpdate(CronTool):
    name = "CronUpdate"
    description = "暂停或恢复当前会话的定时计划；暂停取消尚未领取的执行，运行中任务继续。恢复从当前时间计算，不补发历史。"
    params_model = UpdateParams
    async def execute(self, params):
        try:
            result = self.runtime.store.change(params.schedule_id, self.session_id(), params.action)
            return ToolResult(f'{result["id"]}: {result["status"]}')
        except (ValueError, KeyError) as exc:
            return ToolResult(str(exc), is_error=True)


class CronDelete(CronTool):
    name = "CronDelete"
    description = "删除当前会话的定时计划并停止未来触发；保留历史执行记录，不强制终止运行中任务。"
    params_model = IdParams
    async def execute(self, params):
        try:
            result = self.runtime.store.change(params.schedule_id, self.session_id(), "delete")
            return ToolResult(f'{result["id"]}: deleted')
        except (ValueError, KeyError) as exc:
            return ToolResult(str(exc), is_error=True)


def build_cron_tools(runtime):
    return [CronCreate(runtime), CronList(runtime), CronUpdate(runtime), CronDelete(runtime)]
