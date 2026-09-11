from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from valecode.persistence import TaskState, TaskStore


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


class SharedTaskStore:


    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._next_id = 1
        self._tasks: dict[str, SharedTask] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        data = json.loads(self._path.read_text(encoding="utf-8"))
        self._next_id = data.get("next_id", 1)
        for t in data.get("tasks", []):
            task = SharedTask.from_dict(t)
            self._tasks[task.id] = task

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "next_id": self._next_id,
            "tasks": [t.to_dict() for t in self._tasks.values()],
        }
        self._path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def create(
        self,
        title: str,
        description: str = "",
        assignee: str = "",
        blocks: list[str] | None = None,
        blocked_by: list[str] | None = None,
        created_by: str = "",
    ) -> SharedTask:
        task_id = str(self._next_id)
        self._next_id += 1
        task = SharedTask(
            id=task_id,
            title=title,
            description=description,
            assignee=assignee,
            blocks=blocks or [],
            blocked_by=blocked_by or [],
            created_by=created_by,
        )
        self._tasks[task_id] = task
        self._save()
        return task

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
        self._load()
        task = self._tasks.get(task_id)
        if task is None:
            return None
        if status is not None:
            task.status = status
        if assignee is not None:
            task.assignee = assignee
        if description is not None:
            task.description = description
        if add_blocks:
            for bid in add_blocks:
                if bid not in task.blocks:
                    task.blocks.append(bid)
        if add_blocked_by:
            for bid in add_blocked_by:
                if bid not in task.blocked_by:
                    task.blocked_by.append(bid)
        self._save()
        return task

    def init_empty(self) -> None:
        self._tasks.clear()
        self._next_id = 1
        self._save()


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
        existing_ids = [int(s.metadata["shared_id"]) for s in self._states()]
        display_id = str(max(existing_ids, default=0) + 1)
        dependencies = [
            self._database_id(item)
            for item in blocked_by or []
            if self._store.get(self._database_id(item)) is not None
        ]
        state = self._store.create(
            {
                "title": title,
                "description": description,
                "assignee": assignee,
                "blocks": blocks or [],
                "blocked_by": blocked_by or [],
                "created_by": created_by,
                "board_status": "pending",
            },
            task_id=self._database_id(display_id),
            team_name=self._team_name,
            dependencies=dependencies,
            metadata={"kind": "shared_team_task", "shared_id": display_id},
        )
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
        database_id = self._database_id(task_id)
        state = self._store.get(database_id)
        if state is None or state.metadata.get("kind") != "shared_team_task":
            return None
        data = dict(state.input)
        if status is not None:
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
            if self._store.get(self._database_id(item)) is not None
        ]
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

    def init_empty(self) -> None:
        # Team names are unique, so a newly created team has no matching rows.
        return
