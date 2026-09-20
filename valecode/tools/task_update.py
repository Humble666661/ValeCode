
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from valecode.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from valecode.teams.manager import TeamManager


class TaskUpdateParams(BaseModel):
    task_id: str
    status: str | None = None
    assignee: str | None = None
    description: str | None = None
    add_blocks: list[str] | None = None
    add_blocked_by: list[str] | None = None
    priority: Literal["low", "medium", "high"] | None = None
    progress: int | None = Field(default=None, ge=0, le=100)


VALID_STATUSES = {"pending", "in_progress", "completed", "blocked"}


class TaskUpdateTool(Tool):
    name = "TaskUpdate"
    description = (
        "Update a shared task's status, assignee, description, or dependencies. "
        "Setting status=in_progress atomically claims a dependency-ready task "
        "for the current teammate. Use add_blocks/add_blocked_by to add "
        "dependency relations."
    )
    params_model = TaskUpdateParams
    category = "command"
    is_concurrency_safe = True


    def __init__(
        self,
        team_manager: TeamManager,
        team_name: str,
        agent_name: str = "",
    ) -> None:
        self._team_manager = team_manager
        self._team_name = team_name
        self._agent_name = agent_name


    async def execute(self, params: BaseModel) -> ToolResult:
        p: TaskUpdateParams = params  # type: ignore[assignment]

        if p.status and p.status not in VALID_STATUSES:
            return ToolResult(
                output=f"Invalid status '{p.status}'. Must be one of: {', '.join(sorted(VALID_STATUSES))}",
                is_error=True,
            )

        store = self._team_manager.get_task_store(self._team_name)
        if store is None:
            return ToolResult(output=f"Task store not found for team '{self._team_name}'", is_error=True)

        try:
            if p.status == "in_progress":
                if not self._agent_name:
                    return ToolResult(
                        output="Current teammate identity is unavailable; task claim denied",
                        is_error=True,
                    )
                if p.assignee is not None and p.assignee != self._agent_name:
                    return ToolResult(
                        output=(
                            "A teammate can only claim a task for itself; "
                            f"current teammate is '{self._agent_name}'"
                        ),
                        is_error=True,
                    )
                if (
                    p.description is not None
                    or p.add_blocks
                    or p.add_blocked_by
                    or p.priority is not None
                    or p.progress is not None
                ):
                    return ToolResult(
                        output=(
                            "Claim the task first, then update its fields or dependencies "
                            "in a separate TaskUpdate call"
                        ),
                        is_error=True,
                    )
                task = store.claim(p.task_id, self._agent_name)
            else:
                task = store.update(
                    task_id=p.task_id,
                    status=p.status,
                    assignee=p.assignee,
                    description=p.description,
                    add_blocks=p.add_blocks,
                    add_blocked_by=p.add_blocked_by,
                    priority=p.priority,
                    progress=p.progress,
                )
        except (KeyError, TimeoutError, ValueError) as exc:
            return ToolResult(output=str(exc), is_error=True)

        if task is None:
            return ToolResult(output=f"Task '{p.task_id}' not found", is_error=True)

        changes: list[str] = []
        if p.status:
            changes.append(f"status → {p.status}")
        if p.assignee is not None:
            changes.append(f"assignee → {p.assignee or '(unassigned)'}")
        if p.description is not None:
            changes.append("description updated")
        if p.add_blocks:
            changes.append(f"blocks += {', '.join(p.add_blocks)}")
        if p.add_blocked_by:
            changes.append(f"blocked_by += {', '.join(p.add_blocked_by)}")
        if p.priority is not None:
            changes.append(f"priority → {p.priority}")
        if p.progress is not None:
            changes.append(f"progress → {task.progress}%")

        return ToolResult(
            output=f"Task {task.id} updated: {'; '.join(changes) if changes else 'no changes'}"
        )
