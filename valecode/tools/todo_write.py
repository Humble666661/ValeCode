"""Session-scoped task progress for the current coding request.

This is deliberately not the Team task board or a background scheduler. The
agent replaces its short checklist atomically; the latest state is injected
into subsequent model calls even if conversation history is compacted.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path
from typing import Callable, Literal

from pydantic import BaseModel, Field, model_validator

from valecode.tools.base import Tool, ToolResult


class TodoItem(BaseModel):
    content: str = Field(min_length=1, max_length=240)
    status: Literal["pending", "in_progress", "completed"]


class TodoWriteParams(BaseModel):
    todos: list[TodoItem] = Field(max_length=30)

    @model_validator(mode="after")
    def validate_progress(self) -> TodoWriteParams:
        if sum(item.status == "in_progress" for item in self.todos) > 1:
            raise ValueError("Only one task may be in_progress")
        normalized = [item.content.strip().casefold() for item in self.todos]
        if any(not content for content in normalized) or len(set(normalized)) != len(normalized):
            raise ValueError("Task contents must be non-empty and unique")
        return self


def format_todos(todos: list[TodoItem]) -> str:
    if not todos:
        return "暂无待办事项。"
    completed = sum(item.status == "completed" for item in todos)
    marks = {"pending": " ", "in_progress": ">", "completed": "x"}
    lines = [f"任务进度：{completed}/{len(todos)} 已完成"]
    lines.extend(
        f"{index}. [{marks[item.status]}] {item.content}"
        for index, item in enumerate(todos, 1)
    )
    return "\n".join(lines)


class TodoStore:
    def __init__(self, work_dir: str | Path, session_id: str) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", session_id):
            raise ValueError("Invalid session ID for task progress")
        root = Path(work_dir).resolve()
        state_dir = (root / ".valecode" / "todos").resolve()
        if not state_dir.is_relative_to(root):
            raise ValueError("Task progress directory escapes the project")
        self.path = state_dir / f"{session_id}.json"

    def load(self) -> list[TodoItem]:
        if self.path.is_symlink():
            raise ValueError("Task progress file must not be a symlink")
        if not self.path.exists():
            return []
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("version") != 1:
            raise ValueError("Unsupported task progress format")
        return TodoWriteParams.model_validate({"todos": raw.get("todos")}).todos

    def save(self, todos: list[TodoItem]) -> None:
        if self.path.is_symlink():
            raise ValueError("Task progress file must not be a symlink")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(
                json.dumps(
                    {"version": 1, "todos": [item.model_dump() for item in todos]},
                    ensure_ascii=False, indent=2,
                ),
                encoding="utf-8",
            )
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)


class TodoWrite(Tool):
    name = "TodoWrite"
    description = (
        "Track progress for a multi-step coding request. Replace the current "
        "session's full checklist on each update; use pending, in_progress, "
        "or completed (only after the work and verification are done). "
        "Keep at most one item in_progress. Skip trivial or informational tasks. "
        "The state persists and is visible in later model turns."
    )
    params_model = TodoWriteParams
    category = "command"
    is_concurrency_safe = False
    is_system_tool = True

    def __init__(self, context: Callable[[], tuple[str | Path, str]]) -> None:
        self._context = context

    def _store(self) -> TodoStore:
        work_dir, session_id = self._context()
        return TodoStore(work_dir, session_id)

    def current_summary(self) -> str:
        try:
            todos = self._store().load()
        except (OSError, ValueError) as exc:
            return f"任务进度不可用：{exc}"
        return format_todos(todos) if todos else ""

    async def execute(self, params: TodoWriteParams) -> ToolResult:
        try:
            self._store().save(params.todos)
        except (OSError, ValueError) as exc:
            return ToolResult(output=f"无法保存任务进度：{exc}", is_error=True)
        return ToolResult(output=format_todos(params.todos))
