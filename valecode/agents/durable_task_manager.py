from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from valecode.agents.task_manager import BackgroundTask, TaskManager
from valecode.persistence import TaskState, TaskStatus, TaskStore

if TYPE_CHECKING:
    from valecode.agent import Agent

log = logging.getLogger(__name__)

RESULT_INLINE_LIMIT = 20_000


class DurableTaskManager(TaskManager):
    """SQLite-backed scheduler; memory objects are process-local handles only."""

    def __init__(
        self,
        task_store: TaskStore,
        *,
        worker_id: str | None = None,
        lease_seconds: float = 30.0,
        heartbeat_interval: float = 10.0,
        max_concurrency: int = 8,
        per_team_concurrency: int = 4,
        retry_base_seconds: float = 1.0,
        retry_max_seconds: float = 30.0,
        maintenance_interval: float = 10.0,
    ) -> None:
        super().__init__()
        self.task_store = task_store
        self.worker_id = worker_id or f"worker-{uuid.uuid4().hex[:12]}"
        self.lease_seconds = lease_seconds
        self.heartbeat_interval = min(
            heartbeat_interval, max(lease_seconds / 2, 0.05)
        )
        self.retry_base_seconds = retry_base_seconds
        self.retry_max_seconds = retry_max_seconds
        self.maintenance_interval = max(0.05, maintenance_interval)
        self._maintenance_task: asyncio.Task[None] | None = None
        self._global_capacity = asyncio.Semaphore(max(1, max_concurrency))
        self._team_limit = max(1, per_team_concurrency)
        self._team_capacity: dict[str, asyncio.Semaphore] = {}
        self.recovered_tasks = self.task_store.recover_expired_leases()

    def start_maintenance(self) -> None:
        """Start periodic lease recovery once an event loop is available."""
        if self._maintenance_task is not None and not self._maintenance_task.done():
            return
        self._maintenance_task = asyncio.create_task(self._maintenance_loop())

    async def _maintenance_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self.maintenance_interval)
                try:
                    recovered = self.task_store.recover_expired_leases()
                except Exception:
                    log.exception("Unable to recover expired background task leases")
                    continue
                if recovered:
                    log.info(
                        "Recovered %d expired background task lease(s)",
                        len(recovered),
                    )
        except asyncio.CancelledError:
            return

    def launch(
        self,
        agent: Agent,
        task: str,
        name: str = "",
        fork_conversation: Any = None,
        *,
        dependencies: list[str] | None = None,
        max_attempts: int = 3,
    ) -> str:
        with agent.tracing.span(
            "task.schedule",
            {
                "session.id": agent.session_id,
                "run.id": agent._current_run_id or "",
                "agent.id": agent.agent_id,
                "task.name": name,
                "task.dependencies": dependencies or [],
                "task.max_attempts": max_attempts,
            },
        ) as span:
            task_id = uuid.uuid4().hex[:8]
            bg = BackgroundTask(
                id=task_id,
                name=name or task_id,
                agent=agent,
                task=task,
                status="queued",
            )
            self._tasks[task_id] = bg
            self.task_store.create(
                {"task": task, "name": bg.name},
                task_id=task_id,
                session_id=agent.session_id or None,
                run_id=agent._current_run_id,
                team_name=agent.team_name or None,
                max_attempts=max_attempts,
                dependencies=dependencies,
                metadata={
                    "forked": fork_conversation is not None,
                    "worktree_path": (
                        agent.work_dir
                        if isinstance(getattr(agent, "work_dir", None), str)
                        else ""
                    ),
                },
            )
            handle = asyncio.create_task(self._run_durable(task_id, fork_conversation))
            self._async_tasks[task_id] = handle
            bg.cancel = handle.cancel
            span.set_attributes({"task.id": task_id})
            return task_id

    async def _claim_when_ready(self, bg: BackgroundTask) -> bool:
        while True:
            state = self.task_store.get(bg.id)
            if state is None or state.status in {TaskStatus.CANCELLED, TaskStatus.FAILED}:
                return False
            claimed = self.task_store.claim(
                bg.id, self.worker_id, lease_seconds=self.lease_seconds
            )
            if claimed is not None:
                self.task_store.mark_running(bg.id, self.worker_id)
                bg.status = "running"
                return True
            bg.status = (
                "blocked" if not self.task_store.dependencies_ready(bg.id) else "queued"
            )
            await asyncio.sleep(0.1)

    async def _heartbeat_loop(self, task_id: str) -> None:
        try:
            while True:
                await asyncio.sleep(self.heartbeat_interval)
                if not self.task_store.heartbeat(
                    task_id, self.worker_id, lease_seconds=self.lease_seconds
                ):
                    return
        except asyncio.CancelledError:
            return

    async def _run_durable(
        self, task_id: str, fork_conversation: Any = None
    ) -> None:
        bg = self._tasks.get(task_id)
        if bg is None:
            return
        team_name = getattr(bg.agent, "team_name", "") or ""
        team_capacity = None
        if team_name:
            team_capacity = self._team_capacity.setdefault(
                team_name, asyncio.Semaphore(self._team_limit)
            )
        with bg.agent.tracing.span(
            "task.run",
            {
                "task.id": task_id,
                "task.name": bg.name,
                "session.id": bg.agent.session_id,
                "agent.id": bg.agent.agent_id,
                "team.name": team_name,
            },
        ) as span:
            try:
                async with self._global_capacity:
                    if team_capacity is None:
                        await self._execute_attempts(bg, fork_conversation)
                    else:
                        async with team_capacity:
                            await self._execute_attempts(bg, fork_conversation)
            except asyncio.CancelledError:
                self._cancel_persisted(bg)
            finally:
                bg.end_time = time.monotonic()
                bg.progress.input_tokens = bg.agent.total_input_tokens
                bg.progress.output_tokens = bg.agent.total_output_tokens
                self._async_tasks.pop(task_id, None)
                await self._notify_queue.put(task_id)
                span.set_attributes(
                    {
                        "task.status": bg.status,
                        "usage.input_tokens": bg.progress.input_tokens,
                        "usage.output_tokens": bg.progress.output_tokens,
                    }
                )

    async def _execute_attempts(
        self, bg: BackgroundTask, fork_conversation: Any = None
    ) -> None:
        while await self._claim_when_ready(bg):
            heartbeat = asyncio.create_task(self._heartbeat_loop(bg.id))
            try:
                state = self.task_store.get(bg.id)
                with bg.agent.tracing.span(
                    "task.attempt",
                    {
                        "task.id": bg.id,
                        "task.attempt": state.attempt_count if state else 0,
                        "task.max_attempts": state.max_attempts if state else 0,
                    },
                ) as attempt_span:
                    try:
                        if fork_conversation is not None:
                            result = await bg.agent.run_to_completion(
                                "", fork_conversation
                            )
                            fork_conversation = None
                        else:
                            result = await bg.agent.run_to_completion(bg.task)
                    except BaseException as exc:
                        attempt_span.record_exception(exc)
                        attempt_span.set_attributes({"task.attempt.status": "failed"})
                        raise
                    attempt_span.set_attributes({"task.attempt.status": "succeeded"})
                bg.result = result
                bg.status = "completed"
                self._succeed(bg)
                await self._teammate_idle_loop(bg)
                return
            except asyncio.CancelledError:
                self._cancel_persisted(bg)
                raise
            except Exception as exc:
                log.error("Background task %s failed: %s", bg.id, exc)
                if not self._fail_or_requeue(bg, exc):
                    return
                state = self.task_store.get(bg.id)
                delay = self._retry_delay(state.attempt_count if state else 1)
                bg.status = "retrying"
                await asyncio.sleep(delay)
            finally:
                heartbeat.cancel()
                await asyncio.gather(heartbeat, return_exceptions=True)

    async def _teammate_idle_loop(self, bg: BackgroundTask) -> None:
        if not bg.agent.team_name or not bg.agent._team_manager:
            return
        mailbox = bg.agent._team_manager.get_mailbox(bg.agent.team_name)
        if not mailbox:
            return
        from valecode.teams.mailbox import create_message

        mailbox.write(
            "lead",
            create_message(
                from_agent=bg.name,
                to_agent="lead",
                content=f"[idle] {bg.name}: completed initial task",
                summary=f"{bg.name} idle",
            ),
        )
        for _ in range(60):
            await asyncio.sleep(1)
            messages = mailbox.consume(bg.agent.agent_id)
            if not messages:
                continue
            prompt = "\n\n".join(
                f"[Message from {message.from_agent}] {message.content}"
                for message in messages
            )
            bg.result = await bg.agent.run_to_completion(prompt)
            mailbox.write(
                "lead",
                create_message(
                    from_agent=bg.name,
                    to_agent="lead",
                    content=f"[idle] {bg.name}: completed follow-up",
                    summary=f"{bg.name} idle",
                ),
            )

    def _retry_delay(self, attempt: int) -> float:
        return min(
            self.retry_max_seconds,
            self.retry_base_seconds * (2 ** max(attempt - 1, 0)),
        )

    def _result_payload(self, bg: BackgroundTask) -> tuple[dict[str, str], str | None]:
        if len(bg.result) <= RESULT_INLINE_LIMIT:
            return {"output": bg.result}, None
        directory = self.task_store.database.path.parent / "task-results"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{bg.id}.txt"
        path.write_text(bg.result, encoding="utf-8")
        return {
            "output": bg.result[:2_000] + "\n… (full result stored in file)"
        }, str(path)

    def _succeed(self, bg: BackgroundTask) -> None:
        state = self.task_store.get(bg.id)
        if state is None:
            return
        payload, path = self._result_payload(bg)
        self.task_store.finish_attempt(bg.id, state.attempt_count, status="succeeded")
        self.task_store.transition(
            bg.id,
            TaskStatus.SUCCEEDED,
            result=payload,
            result_path=path,
            error=None,
            lease_owner=None,
            lease_expires_at=None,
            heartbeat_at=None,
            input_tokens=bg.agent.total_input_tokens,
            output_tokens=bg.agent.total_output_tokens,
        )

    def _cancel_persisted(self, bg: BackgroundTask) -> None:
        bg.status = "cancelled"
        bg.result = "Task was cancelled"
        state = self.task_store.get(bg.id)
        if state is None or state.status == TaskStatus.CANCELLED:
            return
        if state.attempt_count:
            try:
                self.task_store.finish_attempt(
                    bg.id, state.attempt_count, status="cancelled", error=bg.result
                )
            except KeyError:
                pass
        self.task_store.transition(
            bg.id,
            TaskStatus.CANCELLED,
            error=bg.result,
            lease_owner=None,
            lease_expires_at=None,
            heartbeat_at=None,
        )

    def _fail_or_requeue(self, bg: BackgroundTask, exc: Exception) -> bool:
        bg.result = f"Error: {exc}"
        state = self.task_store.get(bg.id)
        if state is None:
            bg.status = "failed"
            return False
        self.task_store.finish_attempt(
            bg.id, state.attempt_count, status="failed", error=str(exc)
        )
        if state.attempt_count >= state.max_attempts:
            bg.status = "failed"
            self.task_store.transition(
                bg.id,
                TaskStatus.FAILED,
                result={"output": bg.result},
                error=str(exc),
                lease_owner=None,
                lease_expires_at=None,
                heartbeat_at=None,
                input_tokens=bg.agent.total_input_tokens,
                output_tokens=bg.agent.total_output_tokens,
            )
            return False
        delay = self._retry_delay(state.attempt_count)
        next_retry = (datetime.now(UTC) + timedelta(seconds=delay)).isoformat(
            timespec="milliseconds"
        )
        self.task_store.transition(
            bg.id,
            TaskStatus.QUEUED,
            error=str(exc),
            lease_owner=None,
            lease_expires_at=None,
            heartbeat_at=None,
            next_retry_at=next_retry,
        )
        return True

    def adopt_running(
        self,
        agent: Agent,
        task_description: str,
        partial_result: str = "",
        name: str = "",
    ) -> str:
        task_id = self.launch(agent, task_description, name=name)
        self._tasks[task_id].result = partial_result
        return task_id

    def adopt_persisted(self, task_id: str, agent: Agent) -> str:
        state = self.task_store.get(task_id)
        if state is None:
            raise KeyError(f"Task not found: {task_id}")
        if state.status != TaskStatus.QUEUED:
            raise ValueError(f"Task is not available for adoption: {state.status.value}")
        bg = BackgroundTask(
            id=task_id,
            name=str(state.input.get("name", task_id)),
            agent=agent,
            task=str(state.input.get("task", "")),
            status="queued",
        )
        self._tasks[task_id] = bg
        handle = asyncio.create_task(self._run_durable(task_id))
        self._async_tasks[task_id] = handle
        bg.cancel = handle.cancel
        return task_id

    def get_persisted(self, task_id: str) -> TaskState | None:
        return self.task_store.get(task_id)

    def list_persisted(self, *, session_id: str | None = None) -> list[TaskState]:
        return self.task_store.list(session_id=session_id)

    def cancel(self, task_id: str) -> bool:
        bg = self._tasks.get(task_id)
        if bg is not None and bg.status in {
            "queued", "blocked", "running", "retrying",
        }:
            handle = self._async_tasks.get(task_id)
            if handle is not None and not handle.done():
                handle.cancel()
                return True

        # A recovered queued task may have no process-local handle. It must
        # still be cancellable from the durable task view.
        state = self.task_store.get(task_id)
        if state is None or state.status != TaskStatus.QUEUED:
            return False
        self.task_store.transition(
            task_id,
            TaskStatus.CANCELLED,
            error="Task was cancelled before adoption",
        )
        return True

    async def shutdown(self) -> None:
        if self._maintenance_task is not None:
            self._maintenance_task.cancel()
            await asyncio.gather(self._maintenance_task, return_exceptions=True)
            self._maintenance_task = None
        await super().shutdown()
