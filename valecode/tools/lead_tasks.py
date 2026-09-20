from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from valecode.tools.base import Tool, ToolResult
from valecode.tools.task_create import TaskCreateParams, TaskCreateTool
from valecode.tools.task_get import TaskGetParams, TaskGetTool
from valecode.tools.task_list import TaskListParams, TaskListTool
from valecode.tools.task_update import TaskUpdateParams, TaskUpdateTool

if TYPE_CHECKING:
    from valecode.teams.manager import TeamManager


class LeadTaskCreateParams(TaskCreateParams):
    team_name: str


class LeadTaskGetParams(TaskGetParams):
    team_name: str


class LeadTaskListParams(TaskListParams):
    team_name: str


class LeadTaskUpdateParams(TaskUpdateParams):
    team_name: str


class _LeadTaskTool(Tool):
    def __init__(
        self,
        team_manager: TeamManager,
        lead_agent_id: str,
    ) -> None:
        self._team_manager = team_manager
        self._lead_agent_id = lead_agent_id

    def _resolve_team(self, team_name: str) -> tuple[str | None, ToolResult | None]:
        team = self._team_manager.get_team(team_name)
        if team is None:
            return None, ToolResult(
                output=f"Team '{team_name}' not found", is_error=True
            )
        if team.lead_agent_id != self._lead_agent_id:
            return None, ToolResult(
                output=f"Current agent is not the lead of team '{team.name}'",
                is_error=True,
            )
        return team.name, None


class LeadTaskCreateTool(_LeadTaskTool):
    name = "TaskCreate"
    description = (
        "Create a shared task for a team led by the current agent. "
        "Use the exact team_name returned by TeamCreate."
    )
    params_model = LeadTaskCreateParams
    category = "command"
    is_concurrency_safe = True

    async def execute(self, params: BaseModel) -> ToolResult:
        p: LeadTaskCreateParams = params  # type: ignore[assignment]
        team_name, error = self._resolve_team(p.team_name)
        if error is not None:
            return error
        delegate = TaskCreateTool(
            self._team_manager, team_name or "", self._lead_agent_id
        )
        return await delegate.execute(
            TaskCreateParams(
                title=p.title,
                description=p.description,
                assignee=p.assignee,
                blocks=p.blocks,
                blocked_by=p.blocked_by,
                priority=p.priority,
                progress=p.progress,
            )
        )


class LeadTaskGetTool(_LeadTaskTool):
    name = "TaskGet"
    description = "Get one shared task from a team led by the current agent."
    params_model = LeadTaskGetParams
    category = "read"
    is_concurrency_safe = True

    async def execute(self, params: BaseModel) -> ToolResult:
        p: LeadTaskGetParams = params  # type: ignore[assignment]
        team_name, error = self._resolve_team(p.team_name)
        if error is not None:
            return error
        return await TaskGetTool(self._team_manager, team_name or "").execute(
            TaskGetParams(task_id=p.task_id)
        )


class LeadTaskListTool(_LeadTaskTool):
    name = "TaskList"
    description = (
        "List shared tasks from a team led by the current agent, optionally "
        "filtered by status or assignee."
    )
    params_model = LeadTaskListParams
    category = "read"
    is_concurrency_safe = True

    async def execute(self, params: BaseModel) -> ToolResult:
        p: LeadTaskListParams = params  # type: ignore[assignment]
        team_name, error = self._resolve_team(p.team_name)
        if error is not None:
            return error
        return await TaskListTool(self._team_manager, team_name or "").execute(
            TaskListParams(
                status=p.status, assignee=p.assignee, priority=p.priority
            )
        )


class LeadTaskUpdateTool(_LeadTaskTool):
    name = "TaskUpdate"
    description = (
        "Update or assign a shared task for a team led by the current agent. "
        "Only teammates may atomically claim status=in_progress for themselves."
    )
    params_model = LeadTaskUpdateParams
    category = "command"
    is_concurrency_safe = True

    async def execute(self, params: BaseModel) -> ToolResult:
        p: LeadTaskUpdateParams = params  # type: ignore[assignment]
        team_name, error = self._resolve_team(p.team_name)
        if error is not None:
            return error
        if p.status == "in_progress":
            return ToolResult(
                output=(
                    "The team lead may assign a pending task, but only the "
                    "teammate performing it may claim status=in_progress"
                ),
                is_error=True,
            )
        return await TaskUpdateTool(self._team_manager, team_name or "").execute(
            TaskUpdateParams(
                task_id=p.task_id,
                status=p.status,
                assignee=p.assignee,
                description=p.description,
                add_blocks=p.add_blocks,
                add_blocked_by=p.add_blocked_by,
                priority=p.priority,
                progress=p.progress,
            )
        )


def build_lead_task_tools(
    team_manager: TeamManager,
    lead_agent_id: str,
) -> list[Tool]:
    return [
        LeadTaskCreateTool(team_manager, lead_agent_id),
        LeadTaskGetTool(team_manager, lead_agent_id),
        LeadTaskListTool(team_manager, lead_agent_id),
        LeadTaskUpdateTool(team_manager, lead_agent_id),
    ]
