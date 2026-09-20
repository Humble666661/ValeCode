from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from valecode.tools.agent_tool import AgentToolParams
from valecode.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from valecode.tools.agent_tool import AgentTool
    from valecode.teams.manager import TeamManager


class TaskDispatchParams(BaseModel):
    team_name: str
    task_id: str
    subagent_type: str
    description: str = "Execute shared task"
    name: str | None = None
    model: str | None = None
    instructions: str = ""


class TaskDispatchTool(Tool):
    name = "TaskDispatch"
    description = (
        "Explicitly dispatch one dependency-ready shared task to a durable "
        "background Agent. The execution record is linked back to the board; "
        "success completes the task and final failure or cancellation blocks it."
    )
    params_model = TaskDispatchParams
    category = "command"
    is_concurrency_safe = False
    uses_global_capacity = False

    def __init__(
        self,
        team_manager: TeamManager,
        lead_agent_id: str,
        agent_tool: AgentTool,
    ) -> None:
        self._team_manager = team_manager
        self._lead_agent_id = lead_agent_id
        self._agent_tool = agent_tool

    async def execute(self, params: BaseModel) -> ToolResult:
        p: TaskDispatchParams = params  # type: ignore[assignment]
        team = self._team_manager.get_team(p.team_name)
        if team is None:
            return ToolResult(output=f"Team '{p.team_name}' not found", is_error=True)
        if team.lead_agent_id != self._lead_agent_id:
            return ToolResult(
                output=f"Current agent is not the lead of team '{team.name}'",
                is_error=True,
            )
        store = self._team_manager.get_task_store(team.name)
        if store is None:
            return ToolResult(
                output=f"Task store not found for team '{team.name}'", is_error=True
            )
        task = store.get(p.task_id)
        if task is None:
            return ToolResult(output=f"Task '{p.task_id}' not found", is_error=True)

        worker_name = p.name or p.subagent_type
        original_assignee = task.assignee
        try:
            store.claim(task.id, worker_name)
        except (KeyError, TimeoutError, ValueError) as exc:
            return ToolResult(output=str(exc), is_error=True)

        prompt_parts = [
            f"Execute shared task {task.id} for team '{team.name}'.",
            f"Title: {task.title}",
        ]
        if task.description:
            prompt_parts.append(f"Description: {task.description}")
        if p.instructions.strip():
            prompt_parts.append(f"Additional instructions: {p.instructions.strip()}")
        prompt_parts.append(
            "Return a concise implementation/result summary. The runtime will "
            "synchronize the terminal execution outcome back to the task board."
        )
        agent_params = AgentToolParams(
            prompt="\n".join(prompt_parts),
            description=p.description,
            subagent_type=p.subagent_type,
            model=p.model,
            run_in_background=True,
            name=worker_name,
        )
        try:
            result = await self._agent_tool.execute_for_board(
                agent_params,
                team_name=team.name,
                task_id=task.id,
            )
        except Exception as exc:
            store.update(task.id, status="pending", assignee=original_assignee)
            return ToolResult(output=f"Task dispatch failed: {exc}", is_error=True)
        if result.is_error:
            store.update(task.id, status="pending", assignee=original_assignee)
            return result
        return ToolResult(
            output=(
                f"Shared task {task.id} dispatched to '{worker_name}'.\n"
                f"{result.output}"
            )
        )
