from __future__ import annotations

import json
import os
import random
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from valecode.persistence import TaskState, TaskStore


class SharedTaskLockTimeout(TimeoutError):
    """Raised instead of mutating a task board without owning its lock."""


class SharedTaskClaimError(ValueError):
    """Raised when a shared task cannot safely be claimed."""


TASK_BOARD_LOCK_ATTEMPTS = 50


@dataclass
class SharedTask:
    id: str
    title: str
    description: str = ""
    status: str = "pending"  # pending | in_progress | completed | blocked
    assignee: str = ""
    blocks: list[str] = field(default_factory=list)
    blocked_by: list[str] = field(default_factory=list)
    created_by: str = ""


    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SharedTask:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def _normalize_task_ids(values: list[str] | None) -> list[str]:
    result: list[str] = []
    for value in values or []:
        task_id = str(value).strip()
        if not task_id:
            raise ValueError("Task dependency IDs cannot be empty")
        if task_id not in result:
            result.append(task_id)
    return result


def _apply_dependency_relations(
    tasks: dict[str, SharedTask],
    task_id: str,
    *,
    add_blocks: list[str] | None = None,
    add_blocked_by: list[str] | None = None,
) -> None:
    """Validate and apply both sides of task dependency relations."""
    task = tasks[task_id]
    blocks = _normalize_task_ids(add_blocks)
    blocked_by = _normalize_task_ids(add_blocked_by)
    referenced = blocks + blocked_by
    missing = [item for item in referenced if item not in tasks]
    if missing:
        raise ValueError(
            f"Unknown shared task dependencies: {', '.join(dict.fromkeys(missing))}"
        )
    if task_id in referenced:
        raise ValueError(f"Task '{task_id}' cannot depend on itself")

    for blocked_id in blocks:
        blocked = tasks[blocked_id]
        if blocked_id not in task.blocks:
            task.blocks.append(blocked_id)
        if task_id not in blocked.blocked_by:
            blocked.blocked_by.append(task_id)
    for dependency_id in blocked_by:
        dependency = tasks[dependency_id]
        if dependency_id not in task.blocked_by:
            task.blocked_by.append(dependency_id)
        if task_id not in dependency.blocks:
            dependency.blocks.append(task_id)

    visiting: set[str] = set()
    visited: set[str] = set()

    def _visit(current_id: str) -> None:
        if current_id in visiting:
            raise ValueError("Shared task dependencies cannot contain a cycle")
        if current_id in visited:
            return
        visiting.add(current_id)
        current = tasks[current_id]
        for dependency_id in current.blocked_by:
            if dependency_id not in tasks:
                raise ValueError(
                    f"Unknown shared task dependencies: {dependency_id}"
                )
            _visit(dependency_id)
        visiting.remove(current_id)
        visited.add(current_id)

    for current_id in tasks:
        _visit(current_id)


class SharedTaskStore:


    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._lock_path = self._path.with_name(f"{self._path.name}.lock")
        self._next_id = 1
        self._tasks: dict[str, SharedTask] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            self._next_id = 1
            self._tasks = {}
            return
        data = json.loads(self._path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("tasks", []), list):
            raise ValueError("Corrupt shared task board")
        self._next_id = int(data.get("next_id", 1))
        self._tasks = {}
        for t in data.get("tasks", []):
            if not isinstance(t, dict):
                raise ValueError("Corrupt shared task board")
            task = SharedTask.from_dict(t)
            self._tasks[task.id] = task

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "next_id": self._next_id,
            "tasks": [t.to_dict() for t in self._tasks.values()],
        }
        temp_path = self._path.with_name(
            f".{self._path.name}.{uuid.uuid4().hex}.tmp"
        )
        try:
            with temp_path.open("x", encoding="utf-8", newline="\n") as handle:
                json.dump(data, handle, indent=2, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self._path)
        finally:
            temp_path.unlink(missing_ok=True)

    def _mutate(self, fn: Callable[[], Any]) -> Any:
        """Reload and mutate the board while holding a bounded file lock."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        owner_token = uuid.uuid4().hex
        last_error: OSError | None = None
        for _ in range(TASK_BOARD_LOCK_ATTEMPTS):
            try:
                fd = os.open(
                    str(self._lock_path),
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o644,
                )
                try:
                    os.write(fd, owner_token.encode("ascii"))
                    os.fsync(fd)
                finally:
                    os.close(fd)
                break
            except FileExistsError:
                try:
                    info = self._lock_path.stat()
                    if time.time() - info.st_mtime > 10:
                        stale_token = self._lock_path.read_text(
                            encoding="ascii", errors="replace"
                        )
                        latest = self._lock_path.stat()
                        if (
                            latest.st_mtime_ns == info.st_mtime_ns
                            and self._lock_path.read_text(
                                encoding="ascii", errors="replace"
                            )
                            == stale_token
                        ):
                            self._lock_path.unlink(missing_ok=True)
                except OSError:
                    pass
                time.sleep((5 + random.randint(0, 45)) / 1000)
            except OSError as exc:
                last_error = exc
                break
        else:
            raise SharedTaskLockTimeout("Timed out acquiring shared task board lock")
        if last_error is not None:
            raise last_error

        try:
            self._load()
            result = fn()
            self._save()
            return result
        finally:
            try:
                if self._lock_path.read_text(encoding="ascii") == owner_token:
                    self._lock_path.unlink(missing_ok=True)
            except OSError:
                pass

    def _claim_loaded(self, task_id: str, assignee: str) -> SharedTask:
        task = self._tasks.get(task_id)
        if task is None:
            raise KeyError(f"Task not found: {task_id}")
        if not assignee.strip():
            raise SharedTaskClaimError("A non-empty assignee is required to claim a task")
        if task.status == "in_progress":
            if task.assignee == assignee:
                return task
            raise SharedTaskClaimError(
                f"Task '{task_id}' is already claimed by '{task.assignee or 'unknown'}'"
            )
        if task.status == "completed":
            raise SharedTaskClaimError(f"Task '{task_id}' is already completed")
        if task.status not in {"pending", "blocked"}:
            raise SharedTaskClaimError(
                f"Task '{task_id}' cannot be claimed from status '{task.status}'"
            )
        if task.assignee and task.assignee != assignee:
            raise SharedTaskClaimError(
                f"Task '{task_id}' is assigned to '{task.assignee}'"
            )
        incomplete = [
            dependency_id
            for dependency_id in task.blocked_by
            if dependency_id not in self._tasks
            or self._tasks[dependency_id].status != "completed"
        ]
        if incomplete:
            raise SharedTaskClaimError(
                f"Task '{task_id}' is blocked by incomplete tasks: {', '.join(incomplete)}"
            )
        task.assignee = assignee
        task.status = "in_progress"
        return task

    def create(
        self,
        title: str,
        description: str = "",
        assignee: str = "",
        blocks: list[str] | None = None,
        blocked_by: list[str] | None = None,
        created_by: str = "",
    ) -> SharedTask:
        def _create() -> SharedTask:
            task_id = str(self._next_id)
            self._next_id += 1
            task = SharedTask(
                id=task_id,
                title=title,
                description=description,
                assignee=assignee,
                created_by=created_by,
            )
            self._tasks[task_id] = task
            _apply_dependency_relations(
                self._tasks,
                task_id,
                add_blocks=blocks,
                add_blocked_by=blocked_by,
            )
            return task

        return self._mutate(_create)

    def get(self, task_id: str) -> SharedTask | None:
        self._load()
        return self._tasks.get(task_id)


    def list_tasks(
        self,
        status: str | None = None,
        assignee: str | None = None,
    ) -> list[SharedTask]:
        self._load()
        result = list(self._tasks.values())
        if status:
            result = [t for t in result if t.status == status]
        if assignee:
            result = [t for t in result if t.assignee == assignee]
        return result


    def update(
        self,
        task_id: str,
        status: str | None = None,
        assignee: str | None = None,
        description: str | None = None,
        add_blocks: list[str] | None = None,
        add_blocked_by: list[str] | None = None,
    ) -> SharedTask | None:
        def _update() -> SharedTask | None:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            if status != "in_progress" and assignee is not None:
                task.assignee = assignee
            if description is not None:
                task.description = description
            _apply_dependency_relations(
                self._tasks,
                task_id,
                add_blocks=add_blocks,
                add_blocked_by=add_blocked_by,
            )
            if status == "in_progress":
                self._claim_loaded(
                    task_id, assignee if assignee is not None else task.assignee
                )
            elif status is not None:
                task.status = status
            return task

        return self._mutate(_update)

    def claim(self, task_id: str, assignee: str) -> SharedTask:
        """Atomically claim one ready task for *assignee*."""
        return self._mutate(lambda: self._claim_loaded(task_id, assignee))

    def init_empty(self) -> None:
        def _clear() -> None:
            self._tasks.clear()
            self._next_id = 1

        self._mutate(_clear)


class DurableSharedTaskStore:
    """Team task-board adapter backed by the control-plane TaskStore."""

    def __init__(self, task_store: TaskStore, team_name: str) -> None:
        self._store = task_store
        self._team_name = team_name

    def _database_id(self, display_id: str) -> str:
        return f"shared:{self._team_name}:{display_id}"

    @staticmethod
    def _to_shared(state: TaskState) -> SharedTask:
        data = state.input
        return SharedTask(
            id=str(state.metadata.get("shared_id", state.id)),
            title=str(data.get("title", "")),
            description=str(data.get("description", "")),
            status=str(data.get("board_status", "pending")),
            assignee=str(data.get("assignee", "")),
            blocks=list(data.get("blocks", [])),
            blocked_by=list(data.get("blocked_by", [])),
            created_by=str(data.get("created_by", "")),
        )

    def _states(self) -> list[TaskState]:
        return [
            state
            for state in self._store.list(team_name=self._team_name, limit=10_000)
            if state.metadata.get("kind") == "shared_team_task"
        ]

    def create(
        self,
        title: str,
        description: str = "",
        assignee: str = "",
        blocks: list[str] | None = None,
        blocked_by: list[str] | None = None,
        created_by: str = "",
    ) -> SharedTask:
        blocks = _normalize_task_ids(blocks)
        blocked_by = _normalize_task_ids(blocked_by)
        dependencies = [
            self._database_id(item)
            for item in blocked_by
        ]
        for _ in range(50):
            existing_ids = [int(s.metadata["shared_id"]) for s in self._states()]
            display_id = str(max(existing_ids, default=0) + 1)
            validation_tasks = {task.id: task for task in self.list_tasks()}
            validation_tasks[display_id] = SharedTask(id=display_id, title=title)
            _apply_dependency_relations(
                validation_tasks,
                display_id,
                add_blocks=blocks,
                add_blocked_by=blocked_by,
            )
            try:
                state = self._store.create(
                    {
                        "title": title,
                        "description": description,
                        "assignee": assignee,
                        "blocks": blocks,
                        "blocked_by": blocked_by,
                        "created_by": created_by,
                        "board_status": "pending",
                    },
                    task_id=self._database_id(display_id),
                    team_name=self._team_name,
                    dependencies=dependencies,
                    metadata={"kind": "shared_team_task", "shared_id": display_id},
                )
                break
            except sqlite3.IntegrityError:
                if self._store.get(self._database_id(display_id)) is None:
                    raise
        else:
            raise RuntimeError("Could not allocate a shared task ID after 50 attempts")
        # ``blocks`` is the inverse relationship: add this task as a dependency
        # of every already-existing target.
        for blocked_id in blocks or []:
            target = self._store.get(self._database_id(blocked_id))
            if target is not None:
                target_input = dict(target.input)
                target_blocked_by = list(target_input.get("blocked_by", []))
                if display_id not in target_blocked_by:
                    target_blocked_by.append(display_id)
                    target_input["blocked_by"] = target_blocked_by
                self._store.update_details(
                    target.id,
                    input=target_input,
                    add_dependencies=[state.id],
                )
        return self._to_shared(state)

    def get(self, task_id: str) -> SharedTask | None:
        state = self._store.get(self._database_id(task_id))
        if state is None or state.metadata.get("kind") != "shared_team_task":
            return None
        return self._to_shared(state)

    def list_tasks(
        self, status: str | None = None, assignee: str | None = None
    ) -> list[SharedTask]:
        tasks = [self._to_shared(state) for state in self._states()]
        if status:
            tasks = [task for task in tasks if task.status == status]
        if assignee:
            tasks = [task for task in tasks if task.assignee == assignee]
        return tasks

    def update(
        self,
        task_id: str,
        status: str | None = None,
        assignee: str | None = None,
        description: str | None = None,
        add_blocks: list[str] | None = None,
        add_blocked_by: list[str] | None = None,
    ) -> SharedTask | None:
        add_blocks = _normalize_task_ids(add_blocks)
        add_blocked_by = _normalize_task_ids(add_blocked_by)
        database_id = self._database_id(task_id)
        state = self._store.get(database_id)
        if state is None or state.metadata.get("kind") != "shared_team_task":
            return None
        validation_tasks = {task.id: task for task in self.list_tasks()}
        _apply_dependency_relations(
            validation_tasks,
            task_id,
            add_blocks=add_blocks,
            add_blocked_by=add_blocked_by,
        )
        data = dict(state.input)
        if status is not None and status != "in_progress":
            data["board_status"] = status
        if assignee is not None:
            data["assignee"] = assignee
        if description is not None:
            data["description"] = description
        for field_name, additions in (
            ("blocks", add_blocks),
            ("blocked_by", add_blocked_by),
        ):
            values = list(data.get(field_name, []))
            for value in additions or []:
                if value not in values:
                    values.append(value)
            data[field_name] = values
        dependencies = [
            self._database_id(item)
            for item in add_blocked_by or []
        ]
        if status == "in_progress":
            if description is not None or add_blocks or add_blocked_by:
                raise SharedTaskClaimError(
                    "Claim a task separately from changing its description or dependencies"
                )
            updated = self._store.claim_board_task(
                database_id, assignee or str(data.get("assignee", ""))
            )
        else:
            updated = self._store.update_details(
                database_id, input=data, add_dependencies=dependencies
            )
            if status is not None:
                updated = self._store.update_board_status(database_id, status)
        for blocked_id in add_blocks or []:
            target = self._store.get(self._database_id(blocked_id))
            if target is None:
                continue
            target_input = dict(target.input)
            target_blocked_by = list(target_input.get("blocked_by", []))
            if task_id not in target_blocked_by:
                target_blocked_by.append(task_id)
                target_input["blocked_by"] = target_blocked_by
            self._store.update_details(
                target.id,
                input=target_input,
                add_dependencies=[database_id],
            )
        return self._to_shared(updated)

    def claim(self, task_id: str, assignee: str) -> SharedTask:
        state = self._store.claim_board_task(self._database_id(task_id), assignee)
        return self._to_shared(state)

    def init_empty(self) -> None:
        # Team names are unique, so a newly created team has no matching rows.
        return
