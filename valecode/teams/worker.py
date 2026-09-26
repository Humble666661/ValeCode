"""Independent teammate runtime, bootstrapped from private launch control state."""
from __future__ import annotations

import asyncio
import os
from dataclasses import replace
from pathlib import Path

from valecode.teams.worker_launch import WorkerLaunch


class WorkerTeamView:
    """Read fresh shared membership without rewriting the parent's JSON cache."""
    def __init__(self, data, database):
        from valecode.persistence import TeamStore, TaskStore
        from valecode.teams.mailbox import Mailbox
        from valecode.teams.shared_task import DurableSharedTaskStore
        self.data = data
        self.teams = TeamStore(database)
        self.mailbox = Mailbox(data["mailbox_dir"])
        self.board = DurableSharedTaskStore(TaskStore(database), data["team_name"])

    def get_team(self, name):
        from valecode.teams.models import AgentTeam, TeammateInfo
        if name != self.data["team_name"]:
            return None
        row = self.teams.get_team(name)
        if row is None or row.status != "active":
            return None
        team = AgentTeam(name, row.lead_agent_id)
        team.members = [TeammateInfo(m.name, m.agent_id, m.agent_type, m.model, m.worktree_path, m.backend_type, m.is_active) for m in self.teams.list_members(name)]
        return team

    def get_mailbox(self, name):
        return self.mailbox if name == self.data["team_name"] else None

    def get_task_store(self, name):
        return self.board if name == self.data["team_name"] else None

    def get_pane_id(self, agent_id):
        return None  # Polling mailbox workers need no terminal key injection.


async def run_worker(path: str | Path) -> int:
    from valecode.agent import (
        Agent, ErrorEvent, MailboxEvent, PermissionRequest, PermissionResponse,
        StreamText, ToolUseEvent, UsageEvent,
    )
    from valecode.client import create_client
    from valecode.config import load_config
    from valecode.conversation import ConversationManager
    from valecode.hooks import HookEngine, load_hooks
    from valecode.memory import load_instructions
    from valecode.mcp import MCPManager
    from valecode.permissions import (
        DangerousCommandDetector, PathSandbox, PermissionChecker, PermissionMode, RuleEngine,
    )
    from valecode.persistence import Database, RunStore
    from valecode.tools import create_default_registry, ToolSource
    from valecode.tools.agent_tool import AgentTool, TEAMMATE_ADDENDUM
    from valecode.agents.tool_filter import build_teammate_tools
    from valecode.teams.mailbox import create_message
    from valecode.tools.impl.tool_search import ToolSearchTool

    launch = WorkerLaunch(path)
    data = launch.claim()
    original_cwd = Path.cwd()
    hooks = None
    mcp = None
    registry = None
    full = None
    heartbeat = None
    status = "failed"
    progress = {"tool_count": 0, "token_count": 0, "last_message": ""}
    database = Database(launch.root / ".valecode" / "control.db")
    # Do NOT instantiate SessionManager: its startup recovery would interrupt
    # the still-running parent/peer runs in this shared database.
    view = WorkerTeamView(data, database)
    agent = None
    try:
        os.chdir(launch.root)
        config = load_config()
        provider = next((p for p in config.providers if p.name == data["provider_name"]), None)
        if provider is None:
            raise ValueError("Selected parent provider is no longer configured")
        provider = replace(provider, model=data["model"])
        definition = AgentTool._definition_from_resume_spec(data["definition"])
        assert definition is not None
        loaded_hooks = load_hooks(config.raw_hooks)
        hooks = HookEngine(loaded_hooks) if loaded_hooks else None
        full = create_default_registry(load_plugins=True)
        registry = full  # Initialization failure still releases created tools.
        if config.mcp_servers:
            mcp = MCPManager()
            mcp.load_configs(config.mcp_servers)
            await mcp.register_all_tools(full)
        # The worker can only use tools actually exposed at launch time.
        ceiling = set(data["allowed_tools"])
        base = type(full)()
        for tool in full.list_tools():
            if tool.name in ceiling:
                full.copy_registration_to(base, tool.name)
        registry = build_teammate_tools(base, view, data["team_name"], data["agent_id"], data["member_name"], "in-process", definition)
        if "ToolSearch" in ceiling:
            registry.register(ToolSearchTool(registry))
        # Protect parent control files even if the worktree resides under /tmp.
        deny = list(PathSandbox._DEFAULT_DENY_WRITE) + [
            str(launch.root / ".valecode" / "pane-workers"),
            str(launch.root / ".valecode" / "control.db"),
            str(launch.root / ".valecode" / "permissions.yaml"),
            str(launch.root / ".valecode" / "permissions.local.yaml"),
        ]
        checker = PermissionChecker(DangerousCommandDetector(), PathSandbox(data["work_dir"], deny_write=deny),
            RuleEngine(Path.home() / ".valecode" / "permissions.yaml", launch.root / ".valecode" / "permissions.yaml", launch.root / ".valecode" / "permissions.local.yaml"),
            mode=PermissionMode(data["permission_mode"]))
        os.chdir(data["work_dir"])
        if data["sandbox"]["enabled"]:
            from valecode.sandbox import attach_sandbox
            attached, reason = attach_sandbox(registry, checker, data["work_dir"],
                network_enabled=data["sandbox"]["network_enabled"], auto_allow=data["sandbox"]["auto_allow"])
            if not attached:
                raise ValueError("Required worker sandbox unavailable: " + reason)
            bash = registry.get("Bash")
            bash.sandbox_config.deny_write.extend(deny)
        agent = Agent(create_client(provider), registry, provider.protocol, work_dir=data["work_dir"],
            max_iterations=definition.max_turns, permission_checker=checker,
            context_window=provider.get_context_window(),
            instructions_content=load_instructions(data["work_dir"]) + "\n" + definition.system_prompt + TEAMMATE_ADDENDUM,
            hook_engine=hooks, run_store=RunStore(database), provider_name=provider.name, model=provider.model)
        agent.agent_id = data["agent_id"]
        agent.parent_id = data["lead_id"]
        agent.parent_run_id = data["parent_run_id"]
        agent.session_id = data["session_id"]
        agent.trace_id = data["trace_id"]
        agent.team_name = data["team_name"]
        agent.agent_type = definition.agent_type
        agent._team_manager = view
        registry.bind_session(data["session_id"])
        conversation = ConversationManager()
        current_status = "running"

        async def monitor():
            while True:
                if not launch.parent_alive() or view.get_team(data["team_name"]) is None:
                    agent.cancel("Parent stopped or heartbeat expired")
                    return
                launch.update_state(current_status, **progress)
                await asyncio.sleep(1)
        heartbeat = asyncio.create_task(monitor())
        prompt = data["prompt"]
        while launch.parent_alive():
            current_status = "running"
            launch.update_state("running", **progress)
            view.teams.set_member_active(data["team_name"], data["agent_id"], True)
            conversation.add_user_message(prompt)
            result = ""
            failed = False
            async for event in agent.run(conversation):
                if isinstance(event, PermissionRequest):
                    event.future.set_result(PermissionResponse.DENY)
                elif isinstance(event, ToolUseEvent):
                    progress["tool_count"] += 1
                    progress["last_activity"] = event.tool_name
                elif isinstance(event, UsageEvent):
                    progress["input_tokens"] = agent.total_input_tokens
                    progress["output_tokens"] = agent.total_output_tokens
                    progress["token_count"] = agent.total_input_tokens + agent.total_output_tokens
                elif isinstance(event, StreamText):
                    result += event.text
                    progress["last_message"] = result[-512:]
                elif isinstance(event, MailboxEvent) and event.message_type == "shutdown_request":
                    agent.cancel("Teammate shutdown requested")
                elif isinstance(event, ErrorEvent):
                    failed = True
                    progress["error"] = event.message[:1000]
            if failed:
                raise RuntimeError(progress.get("error", "Worker run failed"))
            current_status = "idle"
            view.teams.set_member_active(data["team_name"], data["agent_id"], False)
            launch.update_state("idle", **progress)
            view.mailbox.write(data["lead_id"], create_message(data["agent_id"], data["lead_id"],
                result[:20000], summary=f'{data["member_name"]} completed'))
            print(result, flush=True)
            while launch.parent_alive():
                messages = view.mailbox.consume(data["agent_id"])
                if any(m.message_type == "shutdown_request" for m in messages):
                    status = "stopped"
                    return 0
                if messages:
                    prompt = "\n\n".join(f"[Message from {m.from_agent}] {m.content}" for m in messages)
                    break
                await asyncio.sleep(0.5)
            else:
                break
        status = "stopped"
        return 0
    except asyncio.CancelledError:
        status = "stopped"
        return 0
    except Exception as exc:
        progress["error"] = str(exc)[:1000]
        return 1
    finally:
        if heartbeat is not None:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
        try:
            launch.update_state(status, **progress)
            with database.transaction(immediate=True) as db:
                db.execute("UPDATE team_members SET is_active=0,status='stopped' WHERE team_name=? AND agent_id=?", (data["team_name"], data["agent_id"]))
        except Exception:
            import logging
            logging.getLogger(__name__).exception("Worker terminal state could not be saved")
        resources = []
        if full is not None:
            resources.append(lambda: full.release_source(ToolSource.PLUGIN))
        if registry is not None:
            resources.append(registry.release_session)
        if mcp is not None:
            resources.append(mcp.shutdown)
        if hooks is not None:
            resources.append(hooks.shutdown)
        try:
            for close in resources:
                try:
                    await close()
                except Exception:
                    import logging
                    logging.getLogger(__name__).exception("Worker cleanup failed")
        finally:
            os.chdir(original_cwd)
