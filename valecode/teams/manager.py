from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from valecode.teams.backend_detect import BackendDetectionError, detect_backend
from valecode.teams.mailbox import Mailbox, create_message
from valecode.teams.models import (
    AgentTeam,
    BackendType,
    TeammateInfo,
    resolve_team_dir,
    use_fallback_team_root,
    unique_team_name,
)
from valecode.teams.progress import TeammateProgress
from valecode.teams.registry import AgentNameRegistry
from valecode.teams.shared_task import DurableSharedTaskStore, SharedTaskStore
from valecode.worktree.paths import canonical_path, is_path_within, require_path_within
from valecode.persistence import TeamStore

if TYPE_CHECKING:
    from valecode.agent import Agent

log = logging.getLogger(__name__)


class TeamError(Exception):
    pass


class TeamManager:
    def __init__(
        self,
        worktree_manager: Any = None,
        trace_manager: Any = None,
        task_store: Any = None,
    ) -> None:
        self._teams: dict[str, AgentTeam] = {}
        self._task_stores: dict[str, SharedTaskStore] = {}
        self._mailboxes: dict[str, Mailbox] = {}
        self._pane_ids: dict[str, str] = {}  # agent_id -> pane_id (tmux/iterm2)
        self._inprocess_tasks: dict[str, tuple[Any, str]] = {}
        self._detected_backend: BackendType | None = None
        self._worktree_manager = worktree_manager
        self._trace_manager = trace_manager
        self._teammate_team_map: dict[str, str] = {}  # agent_id -> team_name
        self._durable_task_store = task_store
        task_database = getattr(task_store, "database", None)
        self._team_store = TeamStore(task_database) if task_database is not None else None

    def detect_backend(
        self,
        teammate_mode: str = "",
        is_interactive: bool = True,
    ) -> BackendType:
        if self._detected_backend is None:
            self._detected_backend = detect_backend(teammate_mode, is_interactive)
        return self._detected_backend


    def create_team(
        self,
        name: str,
        lead_agent_id: str,
        description: str = "",
        teammate_mode: str = "",
        is_interactive: bool = True,
    ) -> AgentTeam:
        backend = self.detect_backend(teammate_mode, is_interactive)
        slug = unique_team_name(name)
        team_dir = resolve_team_dir(slug)
        try:
            team_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            team_dir = use_fallback_team_root() / slug
            team_dir.mkdir(parents=True, exist_ok=True)

        config_path = str(team_dir / "config.json")
        team = AgentTeam(
            name=slug,
            lead_agent_id=lead_agent_id,
            config_path=config_path,
            description=description,
        )
        team.save()

        task_store = (
            DurableSharedTaskStore(self._durable_task_store, slug)
            if self._durable_task_store is not None
            else SharedTaskStore(team_dir / "tasks.json")
        )
        task_store.init_empty()

        mailbox_dir = team_dir / "mailbox"
        mailbox_dir.mkdir(parents=True, exist_ok=True)
        mailbox = Mailbox(mailbox_dir)

        self._teams[slug] = team
        self._task_stores[slug] = task_store
        self._mailboxes[slug] = mailbox
        if self._team_store is not None:
            self._team_store.upsert_team(
                slug,
                lead_agent_id,
                description=description,
                backend_type=backend.value,
            )

        log.info("Created team '%s' at %s (backend=%s)", slug, team_dir, backend.value)
        return team
    def get_team(self, name: str) -> AgentTeam | None:
        if name in self._teams:
            return self._teams[name]
        team_dir = resolve_team_dir(name)
        config_path = team_dir / "config.json"
        if config_path.exists():
            team = AgentTeam.load(str(config_path))
            self._teams[name] = team
            if self._team_store is not None:
                current_state = self._team_store.get_team(team.name)
                self._team_store.upsert_team(
                    team.name,
                    team.lead_agent_id,
                    description=team.description,
                    backend_type=(
                        current_state.backend_type if current_state is not None else ""
                    ),
                )
                for member in team.members:
                    self._team_store.upsert_member(team.name, member)
            return team
        if self._team_store is not None:
            state = self._team_store.get_team(name)
            if state is not None and state.status == "active":
                team = AgentTeam(
                    name=state.name,
                    lead_agent_id=state.lead_agent_id,
                    description=state.description,
                    config_path=str(config_path),
                )
                for member in self._team_store.list_members(name):
                    team.add_member(
                        TeammateInfo(
                            name=member.name,
                            agent_id=member.agent_id,
                            agent_type=member.agent_type,
                            model=member.model,
                            worktree_path=member.worktree_path,
                            backend_type=member.backend_type,
                            is_active=member.is_active,
                        )
                    )
                self._teams[name] = team
                return team
        return None

    def get_task_store(self, team_name: str) -> SharedTaskStore | None:
        if team_name in self._task_stores:
            return self._task_stores[team_name]
        team_dir = resolve_team_dir(team_name)
        tasks_path = team_dir / "tasks.json"
        if self._durable_task_store is not None and (team_dir / "config.json").exists():
            store = DurableSharedTaskStore(self._durable_task_store, team_name)
            self._task_stores[team_name] = store
            return store
        if tasks_path.exists():
            store = SharedTaskStore(tasks_path)
            self._task_stores[team_name] = store
            return store
        return None

    def get_mailbox(self, team_name: str) -> Mailbox | None:
        if team_name in self._mailboxes:
            return self._mailboxes[team_name]
        team_dir = resolve_team_dir(team_name)
        mailbox_dir = team_dir / "mailbox"
        if mailbox_dir.exists():
            mailbox = Mailbox(mailbox_dir)
            self._mailboxes[team_name] = mailbox
            return mailbox
        return None

    def register_member(
        self,
        team_name: str,
        member: TeammateInfo,
    ) -> None:
        team = self.get_team(team_name)
        if team is None:
            raise TeamError(f"Team '{team_name}' not found")
        team.add_member(member)
        team.save()

        AgentNameRegistry.instance().register(member.name, member.agent_id)
        self._teammate_team_map[member.agent_id] = team_name
        if self._team_store is not None:
            self._team_store.upsert_member(team_name, member)
        log.info("Registered member '%s' (agent=%s) in team '%s'", member.name, member.agent_id, team_name)

    def set_member_idle(self, team_name: str, member_name: str) -> None:
        team = self.get_team(team_name)
        if team is None:
            return
        member = team.get_member(member_name)
        if member is None or member.is_active is False:
            return
        team.set_member_active(member_name, False)
        team.save()
        if self._team_store is not None:
            self._team_store.set_member_active(team_name, member_name, False)

        mailbox = self.get_mailbox(team_name)
        if mailbox:
            msg = create_message(
                from_agent=member_name,
                to_agent=team.lead_agent_id,
                content=f"Teammate '{member_name}' is now idle (run_to_completion finished).",
                summary=f"{member_name} idle",
                message_type="text",
            )
            mailbox.write(team.lead_agent_id, msg)

    def set_member_active(self, team_name: str, member_name: str) -> None:
        team = self.get_team(team_name)
        if team is None:
            return
        team.set_member_active(member_name, True)
        team.save()
        if self._team_store is not None:
            self._team_store.set_member_active(team_name, member_name, True)

    def register_inprocess_task(
        self, agent_id: str, task_manager: Any, task_id: str
    ) -> None:
        self._inprocess_tasks[agent_id] = (task_manager, task_id)

    def register_pane_id(self, agent_id: str, pane_id: str) -> None:
        self._pane_ids[agent_id] = pane_id


    def get_pane_id(self, agent_id: str) -> str | None:
        return self._pane_ids.get(agent_id)

    def delete_team(self, team_name: str) -> None:
        team = self.get_team(team_name)
        if team is None:
            raise TeamError(f"Team '{team_name}' not found")

        active = [m for m in team.members if m.is_active is not False]
        if active:
            names = ", ".join(m.name for m in active)
            raise TeamError(f"Cannot delete team: active members: {names}")

        for member in list(team.members):
            AgentNameRegistry.instance().unregister(member.name)

            runtime = self._inprocess_tasks.pop(member.agent_id, None)
            if runtime is not None:
                task_manager, task_id = runtime
                task_manager.cancel(task_id)

            pane_id = self._pane_ids.pop(member.agent_id, None)
            if pane_id:
                self._kill_pane(pane_id, member.backend_type)

            if member.worktree_path:
                self._cleanup_worktree(member.worktree_path)

            if self._trace_manager:
                self._trace_manager.remove(member.agent_id)

        mailbox = self.get_mailbox(team_name)
        if mailbox:
            mailbox.cleanup_all()

        team_dir = resolve_team_dir(team_name)
        self._remove_dir(team_dir)

        self._teams.pop(team_name, None)
        self._task_stores.pop(team_name, None)
        self._mailboxes.pop(team_name, None)
        if self._team_store is not None:
            self._team_store.mark_deleted(team_name)

        log.info("Deleted team '%s'", team_name)

    def list_teams(self) -> list[str]:
        names = list(self._teams.keys())
        if self._team_store is not None:
            for team in self._team_store.list_active():
                if team.name not in names:
                    names.append(team.name)
        return names

    def get_team_for_teammate(self, agent_id: str) -> str | None:
        if agent_id in self._teammate_team_map:
            return self._teammate_team_map[agent_id]
        for name, team in self._teams.items():
            for m in team.members:
                if m.agent_id == agent_id:
                    return name
        return None


    def drain_lead_mailbox(self) -> list[str]:
        notes: list[str] = []
        for team_name in list(self._teams.keys()):
            team = self.get_team(team_name)
            if team is None:
                continue
            mailbox = self.get_mailbox(team_name)
            if mailbox is None:
                continue
            msgs = mailbox.consume(team.lead_agent_id)
            if not msgs:
                continue
            parts = [f'<team-notification team="{team_name}">']
            for m in msgs:
                parts.append(f"from={m.from_agent}: {m.content}")
            parts.append("</team-notification>")
            notes.append("\n".join(parts))
        return notes

    def get_all_teammate_progress(self) -> list[TeammateProgress]:
        """Collect progress objects attached to every registered teammate."""
        results: list[TeammateProgress] = []
        for team in self._teams.values():
            for member in team.members:
                if hasattr(member, "progress") and member.progress is not None:
                    results.append(member.progress)
        return results

    def on_teammate_completed(self, agent_id: str) -> None:
        self._inprocess_tasks.pop(agent_id, None)
        team_name = self.get_team_for_teammate(agent_id)
        if team_name is None:
            return
        team = self.get_team(team_name)
        if team is None:
            return
        member = next((m for m in team.members if m.agent_id == agent_id), None)
        if member:
            self.set_member_idle(team_name, member.name)


    def _kill_pane(self, pane_id: str, backend_type: str) -> None:
        try:
            if backend_type == BackendType.TMUX.value:
                from valecode.teams.spawn_tmux import kill_pane
                kill_pane(pane_id)
        except Exception as e:
            log.warning("Failed to kill pane %s: %s", pane_id, e)

    def _cleanup_worktree(self, worktree_path: str) -> None:
        manager = self._worktree_manager
        if manager is None or not is_path_within(worktree_path, manager.worktree_dir):
            log.warning("Refusing unmanaged team worktree cleanup: %s", worktree_path)
            return
        try:
            registration = manager._registered_worktrees().get(
                canonical_path(worktree_path)
            )
            if registration is None:
                log.warning("Team worktree is not registered: %s", worktree_path)
                return
            result = manager._run_git(
                ["worktree", "remove", "--force", "--", worktree_path]
            )
            if result.returncode != 0:
                log.warning(
                    "git worktree remove failed for %s: %s",
                    worktree_path,
                    result.stderr.strip(),
                )
                return
            branch_ref = registration.get("branch", "")
            if branch_ref.startswith("refs/heads/"):
                manager._run_git(
                    ["branch", "-D", "--", branch_ref.removeprefix("refs/heads/")]
                )
        except Exception as e:
            log.warning("git worktree remove failed for %s: %s", worktree_path, e)

    def _remove_dir(self, path: Path) -> None:
        import shutil
        try:
            team_root = resolve_team_dir("boundary-probe").parent
            safe_path = require_path_within(path, team_root, label="team directory")
            if safe_path.exists():
                shutil.rmtree(safe_path, ignore_errors=True)
        except ValueError as e:
            log.warning("Refusing unsafe team directory removal %s: %s", path, e)
        except Exception as e:
            log.warning("Failed to remove directory %s: %s", path, e)
