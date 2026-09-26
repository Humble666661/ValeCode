"""One session's timer bridge into the existing durable worker lifecycle."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from valecode.persistence.schedule_store import ScheduleStore

log = logging.getLogger(__name__)


class CronRuntime:
    def __init__(self, agent_tool, *, poll_interval: float = 5.0, ready=None):
        self.agent_tool = agent_tool
        self.store = ScheduleStore(agent_tool._task_manager.task_store.database)
        self.poll_interval = max(0.05, poll_interval)
        self._task: asyncio.Task | None = None
        self._closed = False
        self.ready = ready or (lambda: True)

    def start(self):
        if self._closed:
            raise RuntimeError("Cron runtime is closed")
        if self._task is None:
            self._task = asyncio.create_task(self._loop())

    def tick(self, *, now: datetime | None = None) -> list[str]:
        if self._closed:
            return []
        parent = self.agent_tool._parent_agent
        if not parent.session_id or not self.ready():
            return []
        admitted = self.store.admit_due(parent.session_id, parent.work_dir, now=now)
        self.agent_tool.recover_persisted_tasks(parent.session_id)
        return admitted

    async def _loop(self):
        try:
            while True:
                await asyncio.sleep(self.poll_interval)
                try:
                    self.tick()
                except Exception:
                    log.exception("Cron tick failed; persisted plans remain recoverable")
        except asyncio.CancelledError:
            return

    async def close(self):
        self._closed = True
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None


def install_cron(agent_tool, registry, *, ready=None, start=True) -> CronRuntime:
    from valecode.tools.cron import build_cron_tools
    runtime = CronRuntime(agent_tool, ready=ready)
    for tool in build_cron_tools(runtime):
        registry.register(tool)
    if start:
        runtime.start()
    return runtime
