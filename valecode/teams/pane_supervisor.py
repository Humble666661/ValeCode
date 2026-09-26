"""Parent-owned leases for independently running teammates; never replay work."""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

from valecode.teams.progress import ToolActivity
from valecode.teams.worker_launch import WorkerLaunch, TERMINAL

log = logging.getLogger(__name__)


@dataclass
class OwnedWorker:
    team_name: str
    member: object
    launch: WorkerLaunch
    pane_id: str
    last_status: str = ""


class PaneSupervisor:
    def __init__(self, manager):
        self.manager = manager
        self.workers: dict[str, OwnedWorker] = {}
        self._task = None
        self._closed = False
        self._poll_lock = asyncio.Lock()

    def register(self, team_name, member, launch, pane_id):
        if self._closed or member.agent_id in self.workers:
            raise ValueError("Worker already owned or supervisor closed")
        self.workers[member.agent_id] = OwnedWorker(team_name, member, launch, pane_id)
        if self._task is None:
            self._task = asyncio.create_task(self._monitor())

    def _apply(self, handle, state):
        member = handle.member
        status = state["status"]
        member.is_active = status in {"starting", "running"}
        if member.progress is not None:
            progress = member.progress
            with progress._lock:
                progress.status = status
                progress.tool_use_count = state.get("tool_count", 0)
                progress.token_count = state.get("token_count", 0)
                progress.last_message = state.get("last_message") or state.get("error")
                if state.get("last_activity"):
                    progress.last_activity = ToolActivity(state["last_activity"], state["last_activity"])
        trace = self.manager._trace_manager
        if trace is not None:
            trace.update(member.agent_id, input_tokens=state.get("input_tokens", 0),
                output_tokens=state.get("output_tokens", 0), tool_call_count=state.get("tool_count", 0), status=status)
            if status in TERMINAL:
                trace.complete(member.agent_id, "failed" if status == "failed" else "cancelled")
        if handle.last_status != status:
            team = self.manager._teams.get(handle.team_name)
            if team is not None:
                team.save()
            store = self.manager._team_store
            if store is not None:
                with store.database.transaction(immediate=True) as db:
                    db.execute("UPDATE team_members SET is_active=?,status=? WHERE team_name=? AND agent_id=?",
                        (int(member.is_active), "stopped" if status == "failed" else status, handle.team_name, member.agent_id))
            handle.last_status = status

    async def poll(self):
        async with self._poll_lock:
            if not self._closed:
                await self._poll_owned()

    async def _poll_owned(self):
        for agent_id, handle in list(self.workers.items()):
            try:
                state = handle.launch.state()
                age = time.time() - state["timestamp"]
                if not -2 <= age < 30:
                    raise ValueError("Worker heartbeat expired")
                self._apply(handle, state)
                if state["status"] in TERMINAL:
                    await asyncio.to_thread(self.manager._kill_pane, handle.pane_id, handle.member.backend_type)
                    self.workers.pop(agent_id, None)
                    self.manager._pane_ids.pop(agent_id, None)
                else:
                    handle.launch.heartbeat_parent()
            except Exception as exc:
                log.warning("Stopping worker %s: %s", agent_id, exc)
                try:
                    handle.launch.heartbeat_parent(stop=True)
                    self._apply(handle, {"status": "failed", "error": str(exc)[:1000]})
                except Exception:
                    log.exception("Failed to persist worker failure")
                await asyncio.to_thread(self.manager._kill_pane, handle.pane_id, handle.member.backend_type)
                self.workers.pop(agent_id, None)
                self.manager._pane_ids.pop(agent_id, None)

    async def _monitor(self):
        while not self._closed:
            await self.poll()
            await asyncio.sleep(1)

    def stop_member(self, agent_id):
        handle = self.workers.pop(agent_id, None)
        if handle is not None:
            handle.launch.heartbeat_parent(stop=True)

    async def close(self):
        if self._closed:
            return
        self._closed = True
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        owned = list(self.workers.values())
        self.workers.clear()
        for handle in owned:
            try:
                handle.launch.heartbeat_parent(stop=True)
            except Exception:
                log.exception("Failed to signal worker stop")
        if owned:
            # Let the model stream/tool cancellation unwind before closing panes.
            await asyncio.sleep(2)
        for handle in owned:
            await asyncio.to_thread(self.manager._kill_pane, handle.pane_id, handle.member.backend_type)
            self.manager._pane_ids.pop(handle.member.agent_id, None)
            try:
                self._apply(handle, {"status": "stopped"})
            except Exception:
                log.exception("Failed to save stopped worker")
        # Closing the lead is not deleting a team: preserve all worktree edits.
