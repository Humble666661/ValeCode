from __future__ import annotations

import asyncio
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from valecode.tools.base import Tool, ToolResult


class QuestionItem(BaseModel):
    type: Literal["text", "radio", "select", "checkbox"] = Field(description="Question input type")
    name: str = Field(min_length=1, max_length=64, description="Question identifier")
    message: str = Field(min_length=1, max_length=2000, description="Question text to display")
    options: list[str] = Field(
        default_factory=list,
        max_length=30,
        description="Options for radio/select/checkbox types",
    )


class AskUserParams(BaseModel):
    questions: list[QuestionItem] = Field(
        min_length=1, max_length=10, description="List of questions to ask the user"
    )

    @model_validator(mode="after")
    def distinct_names(self):
        if len({q.name for q in self.questions}) != len(self.questions):
            raise ValueError("Question names must be unique")
        if any(len(option) > 1000 for q in self.questions for option in q.options):
            raise ValueError("Question option exceeds 1000 characters")
        return self


class AskUserEvent:


    def __init__(
        self,
        questions: list[dict[str, Any]],
        future: asyncio.Future[dict[str, str]],
    ) -> None:
        self.questions = questions
        self.future = future


class AskUserTool(Tool):
    name = "AskUserQuestion"
    description = (
        "Ask the user one or more questions when you need information "
        "that cannot be determined from code or context alone. Supports "
        "text input, radio (single select), select, and checkbox (multi select) "
        "question types."
    )
    params_model = AskUserParams
    category: str = "read"
    is_system_tool = True
    should_defer = True


    def __init__(self, on_request=None) -> None:
        self._pending_event: AskUserEvent | None = None
        self._on_request = on_request

    async def execute(self, params: AskUserParams) -> ToolResult:
        questions_data = [q.model_dump() for q in params.questions]

        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, str]] = loop.create_future()

        self._pending_event = AskUserEvent(questions=questions_data, future=future)

        try:
            if self._on_request is None:
                return ToolResult(output="This host has no interactive question adapter", is_error=True)
            await self._on_request(self._pending_event)
            answers = await asyncio.wait_for(future, timeout=300)
        except asyncio.TimeoutError:
            return ToolResult(
                output="User did not respond within 5 minutes", is_error=True
            )
        finally:
            self._pending_event = None

        lines = []
        for q in params.questions:
            answer = answers.get(q.name, "(no answer)")
            lines.append(f"{q.name}: {answer}")

        return ToolResult(output="\n".join(lines))
