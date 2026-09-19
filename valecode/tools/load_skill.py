from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from valecode.skills.content import render_skill_content
from valecode.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from valecode.agent import Agent
    from valecode.skills.loader import SkillLoader


class LoadSkillParams(BaseModel):
    name: str = Field(description="The name of the skill to load")


class LoadSkill(Tool):
    name = "LoadSkill"
    description = (
        "Load and activate a skill by name. "
        "Returns the full SOP body so you can follow its instructions."
    )
    params_model = LoadSkillParams
    category = "read"
    is_concurrency_safe = False
    is_system_tool = True


    def __init__(self) -> None:
        self._loader: SkillLoader | None = None
        self._agent: Agent | None = None


    def set_loader(self, loader: SkillLoader) -> None:
        self._loader = loader

    def set_agent(self, agent: Agent) -> None:
        self._agent = agent


    async def execute(self, params: BaseModel) -> ToolResult:
        assert isinstance(params, LoadSkillParams)

        if self._loader is None or self._agent is None:
            return ToolResult(
                output="Error: LoadSkill not properly initialized",
                is_error=True,
            )

        skill = self._loader.get(params.name)
        if skill is None:
            available = ", ".join(n for n, _ in self._loader.get_catalog())
            return ToolResult(
                output=f"Error: unknown skill '{params.name}'. Available skills: {available}",
                is_error=True,
            )

        rendered = render_skill_content(skill)
        self._agent.activate_skill(skill.name, rendered, skill.permission_rules)
        recovery = getattr(self._agent, "recovery_state", None)
        if recovery is not None:
            recovery.record_skill_invocation(skill.name, rendered)

        return ToolResult(output=rendered)
