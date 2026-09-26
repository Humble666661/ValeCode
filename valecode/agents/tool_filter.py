
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from valecode.tools import ToolRegistry, ToolSource

if TYPE_CHECKING:
    from valecode.agents.parser import AgentDef
    from valecode.teams.manager import TeamManager

ALL_AGENT_DISALLOWED_TOOLS: frozenset[str] = frozenset({
    "TaskOutput",
    "ExitPlanMode",
    "EnterPlanMode",
    "Agent",
    "AskUserQuestion",
    "TaskStop",
    "Workflow",
    "TodoWrite",
    "TaskCreate",
    "TaskGet",
    "TaskList",
    "TaskUpdate",
    "TaskDispatch",
    "CronCreate",
    "CronDelete",
    "CronList",
    "CronUpdate",
})

CUSTOM_AGENT_DISALLOWED_TOOLS: frozenset[str] = frozenset({
    "TaskOutput",
    "ExitPlanMode",
    "EnterPlanMode",
    "Agent",
    "AskUserQuestion",
    "TaskStop",
    "Workflow",
})

ASYNC_AGENT_ALLOWED_TOOLS: frozenset[str] = frozenset({
    "ReadFile",
    "WebSearch",
    "Grep",
    "WebFetch",
    "Glob",
    "Bash",
    "EditFile",
    "WriteFile",
    "NotebookEdit",
    "Skill",
    "LoadSkill",
    "SyntheticOutput",
    "ToolSearch",
    "EnterWorktree",
    "ExitWorktree",
})

TEAMMATE_COORDINATION_TOOLS: frozenset[str] = frozenset({
    "TaskCreate",
    "TaskGet",
    "TaskList",
    "TaskUpdate",
    "SendMessage",
})

IN_PROCESS_TEAMMATE_ALLOWED_TOOLS: frozenset[str] = (
    ASYNC_AGENT_ALLOWED_TOOLS | TEAMMATE_COORDINATION_TOOLS
)

COORDINATOR_MODE_ALLOWED_TOOLS: frozenset[str] = frozenset({
    "CronCreate",
    "CronList",
    "CronUpdate",
    "CronDelete",
    "Agent",
    "SendMessage",
    "TaskCreate",
    "TaskGet",
    "TaskList",
    "TaskUpdate",
    "TaskDispatch",
    "TaskStop",
    "SyntheticOutput",
    "TeamCreate",
    "TeamDelete",
    "ReadFile",
    "Glob",
    "Grep",
    "Bash",
})


def resolve_agent_tools(
    parent_registry: ToolRegistry,
    definition: AgentDef,
    is_background: bool = False,
) -> ToolRegistry:
    all_tools = {t.name: t for t in parent_registry.list_tools()}

    # 第 0 层：MCP 工具始终放行，先分离出来再做后续过滤
    mcp_tools = {
        name: tool
        for name, tool in all_tools.items()
        if (
            (registration := parent_registry.get_registration(name)) is not None
            and registration.source == ToolSource.MCP
        )
    }
    all_tools = {name: tool for name, tool in all_tools.items() if name not in mcp_tools}

    # 第 1 层：全局禁用工具
    for name in ALL_AGENT_DISALLOWED_TOOLS:
        all_tools.pop(name, None)

    # 第 2 层：自定义 agent 额外限制
    if definition.source in ("project", "user", "plugin"):
        for name in CUSTOM_AGENT_DISALLOWED_TOOLS:
            all_tools.pop(name, None)

    # 第 3 层：后台任务白名单
    if is_background:
        all_tools = {
            name: tool
            for name, tool in all_tools.items()
            if name in ASYNC_AGENT_ALLOWED_TOOLS
        }

    # 第 4 层：按 agent 定义中的禁用/允许列表过滤
    if definition.disallowed_tools:
        for name in definition.disallowed_tools:
            all_tools.pop(name, None)

    if definition.tools:
        allowed_set = set(definition.tools)
        all_tools = {
            name: tool
            for name, tool in all_tools.items()
            if name in allowed_set
        }

    filtered = ToolRegistry()
    for name in (*mcp_tools, *all_tools):
        parent_registry.copy_registration_to(filtered, name)
    return filtered


def build_teammate_tools(
    parent_registry: ToolRegistry,
    team_manager: TeamManager,
    team_name: str,
    agent_id: str,
    agent_name: str,
    backend_type: str,
    definition: AgentDef | None = None,
) -> ToolRegistry:
    from valecode.teams.models import BackendType
    from valecode.tools.send_message import SendMessageTool
    from valecode.tools.task_create import TaskCreateTool
    from valecode.tools.task_get import TaskGetTool
    from valecode.tools.task_list import TaskListTool
    from valecode.tools.task_update import TaskUpdateTool

    if backend_type in {item.value for item in BackendType}:
        all_tools = {t.name: t for t in parent_registry.list_tools()}
        filtered = {
            name: tool
            for name, tool in all_tools.items()
            if name in IN_PROCESS_TEAMMATE_ALLOWED_TOOLS
            or (parent_registry.get_registration(name).source == ToolSource.MCP)
        }
    else:
        filtered = {t.name: t for t in parent_registry.list_tools()}
        filtered.pop("TeamCreate", None)
        filtered.pop("TeamDelete", None)

    # 应用 agent 定义中的工具限制
    if definition is not None:
        if definition.disallowed_tools:
            for name in definition.disallowed_tools:
                filtered.pop(name, None)
        if definition.tools:
            allowed_set = set(definition.tools) | TEAMMATE_COORDINATION_TOOLS
            filtered = {
                name: tool
                for name, tool in filtered.items()
                if name in allowed_set
            }

    for name in TEAMMATE_COORDINATION_TOOLS:
        filtered.pop(name, None)

    coordination_tools = [
        TaskCreateTool(team_manager, team_name, agent_name),
        TaskGetTool(team_manager, team_name),
        TaskListTool(team_manager, team_name),
        TaskUpdateTool(team_manager, team_name, agent_name),
        SendMessageTool(team_manager, team_name, agent_id, agent_name),
    ]

    registry = ToolRegistry()
    for name in filtered:
        parent_registry.copy_registration_to(registry, name)
    for tool in coordination_tools:
        registry.register(tool)

    return registry


def clone_registry_for_fork(parent_registry: ToolRegistry) -> ToolRegistry:
    """Fork 专用：复制父注册表的工具，但不共享父会话 TodoWrite。

    遇到 AgentTool 实例时浅复制并标记 query_source，
    确保 fork 子 Agent 不能再次 fork（运行时拦截），
    同时保持工具定义与父 Agent 字节一致以命中 prompt cache。
    """
    import copy

    from valecode.tools.agent_tool import FORK_QUERY_SOURCE

    forked = ToolRegistry()
    for tool in parent_registry.list_tools():
        if tool.name in {"TodoWrite", "CronCreate", "CronList", "CronUpdate", "CronDelete"}:
            continue
        if tool.name == "Agent" and hasattr(tool, "query_source"):
            clone = copy.copy(tool)
            clone.query_source = FORK_QUERY_SOURCE
            registration = parent_registry.get_registration(tool.name)
            forked.register(
                clone,
                source=registration.source if registration else ToolSource.SESSION,
                scope_id=registration.scope_id if registration else None,
            )
        else:
            parent_registry.copy_registration_to(forked, tool.name)
    return forked


def apply_coordinator_filter(registry: ToolRegistry) -> ToolRegistry:
    all_tools = {t.name: t for t in registry.list_tools()}
    filtered = ToolRegistry()
    for name, tool in all_tools.items():
        registration = registry.get_registration(name)
        if (
            registration is not None and registration.source == ToolSource.MCP
        ) or name in COORDINATOR_MODE_ALLOWED_TOOLS:
            registry.copy_registration_to(filtered, name)
    return filtered
