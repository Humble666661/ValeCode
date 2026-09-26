"""Explicit, host-independent assembly for the shared execution components.

UI rendering, provider discovery and transport startup remain host adapters.
This factory reads no application config and installs no process globals.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class HarnessOptions:
    enable_fork: bool = False
    enable_verification: bool = False
    enable_coordinator: bool = False
    interactive: bool = False
    teammate_mode: str = "in-process"
    background_tasks: object = None
    worktree_config: object = None


@dataclass
class HarnessComponents:
    agent_loader: object
    agent_tool: object
    task_manager: object
    team_manager: object
    worktree_manager: object
    trace_manager: object
    cron_runtime: object

    async def close(self):
        await close_resources([
            ("cron", self.cron_runtime.close),
            ("teams", self.team_manager.close),
            ("tasks", self.task_manager.shutdown),
        ])


async def close_resources(resources):
    """One failing resource cannot abandon all subsequently owned resources."""
    for label, close in resources:
        if close is not None:
            try:
                await close()
            except Exception:
                log.exception("Failed to close %s", label)


def sync_worktree_context(agent, manager):
    from valecode.permissions import PathSandbox
    session = manager.get_current_session()
    target = session.worktree_path if session is not None else manager.repo_root
    if agent.work_dir != target:
        agent.work_dir = target
        agent._plan_path_cache = None
    if agent.permission_checker is not None:
        agent.permission_checker.sandbox = PathSandbox(target)
        agent.permission_checker.plan_file_path = ""


def assemble_harness(agent, session_manager, provider, registry, *, options=None,
    task_manager=None, trace_manager=None, team_manager=None, worktree_manager=None,
    cron_ready=None, cron_start=False):
    from valecode.agents.durable_task_manager import DurableTaskManager
    from valecode.agents.loader import AgentLoader
    from valecode.agents.trace import TraceManager
    from valecode.teams.manager import TeamManager
    from valecode.tools.agent_tool import AgentTool
    from valecode.tools.enter_worktree import EnterWorktreeTool
    from valecode.tools.exit_worktree import ExitWorktreeTool
    from valecode.tools.team_create import TeamCreateTool
    from valecode.tools.team_delete import TeamDeleteTool
    from valecode.tools.lead_tasks import build_lead_task_tools
    from valecode.worktree.manager import WorktreeManager
    from valecode.runtime.cron import install_cron

    settings = options or HarnessOptions()
    loader = AgentLoader(agent.work_dir, enable_verification=settings.enable_verification)
    loader.load_all()
    traces = trace_manager if trace_manager is not None else TraceManager()
    worktrees = worktree_manager if worktree_manager is not None else WorktreeManager(
        repo_root=agent.work_dir, symlink_directories=getattr(settings.worktree_config, "symlink_directories", []))
    tasks = task_manager if task_manager is not None else DurableTaskManager.from_config(
        session_manager.task_store, settings.background_tasks)
    teams = team_manager if team_manager is not None else TeamManager(
        worktree_manager=worktrees, trace_manager=traces, task_store=session_manager.task_store)
    teams._worktree_manager = worktrees
    tool = AgentTool(loader, tasks, traces, agent, enable_fork=settings.enable_fork,
        provider_config=provider, worktree_manager=worktrees, team_manager=teams)
    registry.register(tool)
    on_change = lambda: sync_worktree_context(agent, worktrees)
    registry.register(EnterWorktreeTool(worktrees, on_change=on_change))
    registry.register(ExitWorktreeTool(worktrees, on_change=on_change))
    registry.register(TeamCreateTool(teams, agent, teammate_mode=settings.teammate_mode,
        is_interactive=settings.interactive, enable_coordinator_mode=settings.enable_coordinator))
    registry.register(TeamDeleteTool(teams, agent))
    for task_tool in build_lead_task_tools(teams, agent.agent_id, tool):
        registry.register(task_tool)
    agent._team_manager = teams
    catalog = loader.list_agents()
    if catalog:
        lines = ["## Available Sub-Agent Types", "", "Use Agent with subagent_type to delegate:", ""]
        lines.extend(f"- **{kind}**: {description}" for kind, description in catalog)
        if settings.enable_fork:
            lines.extend(["", "Leave subagent_type empty to fork the current conversation."])
        lines.extend(["", "Background results arrive automatically; do not poll, sleep or duplicate work."])
        agent.set_agent_catalog("\n".join(lines), catalog_list=catalog)
    cron = install_cron(tool, registry, ready=cron_ready, start=False)
    tasks.start_maintenance()
    if cron_start:
        cron.start()
    return HarnessComponents(loader, tool, tasks, teams, worktrees, traces, cron)
