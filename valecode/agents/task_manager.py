from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from valecode.agent import Agent

log = logging.getLogger(__name__)


@dataclass
class ProgressInfo:
    tool_call_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    last_activity: str = ""


@dataclass
class BackgroundTask:
    id: str
    name: str
    agent: Agent
    task: str
    status: str = "running"
    result: str = ""
    start_time: float = field(default_factory=time.monotonic)
    end_time: float | None = None
    cancel: Callable[[], None] | None = None
    progress: ProgressInfo = field(default_factory=ProgressInfo)
    board_team_name: str = ""
    board_task_id: str = ""
    board_synced: bool = False


class TaskManager:


    def __init__(self) -> None:
        self._tasks: dict[str, BackgroundTask] = {}
        self._notify_queue: asyncio.Queue[str] = asyncio.Queue()
        self._async_tasks: dict[str, asyncio.Task[None]] = {}


    def launch(
        self,
        agent: Agent,
        task: str,
        name: str = "",
        fork_conversation: Any = None,
        *,
        resume_spec: dict[str, Any] | None = None,
        board_team_name: str = "",
        board_task_id: str = "",
    ) -> str:
        # The in-memory manager does not need a reconstruction descriptor, but
        # accepts it so callers can use the same launch contract as the durable
        # manager.
        del resume_spec
        task_id = uuid.uuid4().hex[:8]
        bg = BackgroundTask(
            id=task_id,
            name=name or task_id,
            agent=agent,
            task=task,
            board_team_name=board_team_name,
            board_task_id=board_task_id,
        )
        self._tasks[task_id] = bg

        async_task = asyncio.create_task(
            self._run_background(task_id, fork_conversation)
        )
        self._async_tasks[task_id] = async_task

        bg.cancel = async_task.cancel
        return task_id


    async def _run_background(
        self, task_id: str, fork_conversation: Any = None
    ) -> None:
        bg = self._tasks.get(task_id)
        if bg is None:
            return

        try:
            if fork_conversation is not None:
                result = await bg.agent.run_to_completion("", fork_conversation)
            else:
                result = await bg.agent.run_to_completion(bg.task)
            bg.result = result
            bg.status = "completed"
            self._sync_board_task(bg, "completed")

            if bg.agent.team_name and bg.agent._team_manager:
                mailbox = bg.agent._team_manager.get_mailbox(bg.agent.team_name)
                if mailbox:
                    from valecode.teams.mailbox import create_message
                    msg = create_message(
                        from_agent=bg.name,
                        to_agent="lead",
                        content=f"[idle] {bg.name}: completed initial task",
                        summary=f"{bg.name} idle",
                    )
                    mailbox.write("lead", msg)

                    for _ in range(60):
                        await asyncio.sleep(1)
                        msgs = mailbox.consume(bg.agent.agent_id)
                        if not msgs:
                            continue
                        prompt = "\n\n".join(
                            f"[Message from {m.from_agent}] {m.content}" for m in msgs
                        )
                        result = await bg.agent.run_to_completion(prompt)
                        bg.result = result
                        msg = create_message(
                            from_agent=bg.name,
                            to_agent="lead",
                            content=f"[idle] {bg.name}: completed follow-up",
                            summary=f"{bg.name} idle",
                        )
                        mailbox.write("lead", msg)

        except asyncio.CancelledError:
            bg.status = "cancelled"
            bg.result = "Task was cancelled"
            self._sync_board_task(bg, "cancelled")
        except Exception as e:
            log.error("Background task %s failed: %s", task_id, e)
            bg.status = "failed"
            bg.result = f"Error: {e}"
            self._sync_board_task(bg, "failed")
        finally:
            bg.end_time = time.monotonic()
            bg.progress.input_tokens = bg.agent.total_input_tokens
            bg.progress.output_tokens = bg.agent.total_output_tokens
            self._async_tasks.pop(task_id, None)
            await self._notify_queue.put(task_id)

    @staticmethod
    def _sync_board_task(bg: BackgroundTask, outcome: str) -> None:
        """Project a terminal execution outcome back to its shared board item."""
        if bg.board_synced or not bg.board_team_name or not bg.board_task_id:
            return
        manager = getattr(bg.agent, "_team_manager", None)
        if manager is None:
            log.error(
                "Cannot sync board task %s/%s: TeamManager unavailable",
                bg.board_team_name,
                bg.board_task_id,
            )
            return
        try:
            store = manager.get_task_store(bg.board_team_name)
            if store is None:
                raise ValueError(f"Team '{bg.board_team_name}' task store not found")
            committed = store.finish_claim(
                bg.board_task_id,
                bg.name,
                succeeded=outcome == "completed",
            )
            if not committed:
                log.warning(
                    "Ignored stale board result for %s/%s owned by %s",
                    bg.board_team_name,
                    bg.board_task_id,
                    bg.name,
                )
            bg.board_synced = True
        except Exception:
            log.exception(
                "Unable to sync board task %s/%s after %s",
                bg.board_team_name,
                bg.board_task_id,
                outcome,
            )


    def adopt_running(
        self,
        agent: Agent,
        task_description: str,
        partial_result: str = "",
        name: str = "",
    ) -> str:
        task_id = uuid.uuid4().hex[:8]
        bg = BackgroundTask(
            id=task_id,
            name=name or task_id,
            agent=agent,
            task=task_description,
            result=partial_result,
        )
        self._tasks[task_id] = bg

        async_task = asyncio.create_task(self._continue_background(task_id))
        self._async_tasks[task_id] = async_task
        bg.cancel = async_task.cancel
        return task_id


    async def _continue_background(self, task_id: str) -> None:
        bg = self._tasks.get(task_id)
        if bg is None:
            return

        try:
            result = await bg.agent.run_to_completion(bg.task)
            bg.result = (bg.result + "\n" + result).strip() if bg.result else result
            bg.status = "completed"
        except asyncio.CancelledError:
            bg.status = "cancelled"
        except Exception as e:
            log.error("Background task %s failed: %s", task_id, e)
            bg.status = "failed"
            bg.result = f"Error: {e}"
        finally:
            bg.end_time = time.monotonic()
            bg.progress.input_tokens = bg.agent.total_input_tokens
            bg.progress.output_tokens = bg.agent.total_output_tokens
            self._async_tasks.pop(task_id, None)
            await self._notify_queue.put(task_id)

    def get(self, task_id: str) -> BackgroundTask | None:
        return self._tasks.get(task_id)

    def list_tasks(self) -> list[BackgroundTask]:
        return list(self._tasks.values())

    def cancel(self, task_id: str) -> bool:
        bg = self._tasks.get(task_id)
        if bg is None or bg.status != "running":
            return False
        async_task = self._async_tasks.get(task_id)
        if async_task and not async_task.done():
            async_task.cancel()
            bg.status = "cancelled"
            bg.result = "Task was cancelled"
            self._sync_board_task(bg, "cancelled")
            return True
        return False

    def poll_completed(self) -> list[BackgroundTask]:
        completed: list[BackgroundTask] = []
        while not self._notify_queue.empty():
            try:
                task_id = self._notify_queue.get_nowait()
                bg = self._tasks.get(task_id)
                if bg is not None:
                    completed.append(bg)
            except asyncio.QueueEmpty:
                break
        return completed

    async def shutdown(self) -> None:
        """Cancel and await every process-local background task."""
        active = [
            (task_id, task)
            for task_id, task in self._async_tasks.items()
            if not task.done()
        ]
        handles = [task for _, task in active]
        for task in handles:
            task.cancel()
        if handles:
            await asyncio.gather(*handles, return_exceptions=True)
        # A task cancelled before its coroutine gets a first timeslice cannot
        # execute its ``finally`` block, so clean those handles explicitly.
        for task_id, task in active:
            if task.done():
                self._async_tasks.pop(task_id, None)
                bg = self._tasks.get(task_id)
                if bg is not None and bg.status == "running":
                    bg.status = "cancelled"
                    bg.result = "Task was cancelled"
                    bg.end_time = time.monotonic()
                    self._sync_board_task(bg, "cancelled")
