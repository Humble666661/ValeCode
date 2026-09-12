from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from valecode.tools import (
    ToolConflictError,
    ToolRegistry,
    ToolSource,
    create_default_registry,
)
from valecode.tools.base import Tool, ToolResult


class Params(BaseModel):
    value: str = ""


class NamedTool(Tool):
    description = "test tool"
    params_model = Params

    def __init__(
        self,
        name: str,
        marker: str,
        *,
        deferred: bool = False,
        output_limit: int = 10_000,
    ) -> None:
        self.name = name
        self.marker = marker
        self.should_defer = deferred
        self.max_output_chars = output_limit
        self.events: list[str] = []
        self.closed = False

    async def before_execute(self, params: BaseModel) -> None:
        self.events.append("before")

    async def execute(self, params: BaseModel) -> ToolResult:
        self.events.append("execute")
        return ToolResult(self.marker)

    async def after_execute(
        self, params: BaseModel, result: ToolResult
    ) -> ToolResult:
        self.events.append("after")
        return ToolResult(result.output + ":after")

    async def on_error(self, params: BaseModel, error: BaseException) -> None:
        self.events.append(f"error:{type(error).__name__}")

    async def close(self) -> None:
        self.closed = True


def test_layer_precedence_is_independent_of_registration_order() -> None:
    registry = ToolRegistry()
    session = NamedTool("Shared", "session")
    builtin = NamedTool("Shared", "builtin")
    mcp = NamedTool("Shared", "mcp")
    plugin = NamedTool("Shared", "plugin")

    registry.register(session, source=ToolSource.SESSION, scope_id="session:s1")
    registry.register(builtin, source=ToolSource.BUILTIN, scope_id="builtin")
    registry.register(mcp, source=ToolSource.MCP, scope_id="mcp:server")
    registry.register(plugin, source=ToolSource.PLUGIN, scope_id="plugin:p1")

    assert registry.get("Shared") is session
    registration = registry.get_registration("Shared")
    assert registration is not None
    assert registration.tool_id == "session:session:s1:Shared"
    assert registration.source == ToolSource.SESSION
    assert len(registry.list_conflicts()) == 3
    assert registry.list_conflicts()[-1].winner_tool_id == registration.tool_id


@pytest.mark.asyncio
async def test_releasing_scope_closes_tool_and_reveals_lower_layer() -> None:
    registry = ToolRegistry()
    builtin = NamedTool("Shared", "builtin")
    session = NamedTool("Shared", "session")
    registry.register(builtin, source=ToolSource.BUILTIN, scope_id="builtin")
    registry.bind_session("abc")
    registration = registry.register(session)

    assert registry.get("Shared") is session
    assert await registry.release_session() == [registration.tool_id]
    assert session.closed
    assert registry.get("Shared") is builtin


@pytest.mark.asyncio
async def test_releasing_scope_closes_replaced_instance() -> None:
    registry = ToolRegistry()
    first = NamedTool("Replaceable", "first")
    second = NamedTool("Replaceable", "second")
    registry.register(first, source=ToolSource.PLUGIN, scope_id="plugin:p1")
    registry.register(second, source=ToolSource.PLUGIN, scope_id="plugin:p1")

    assert registry.get("Replaceable") is second
    assert await registry.release_scope("plugin:p1") == [
        "plugin:plugin:p1:Replaceable"
    ]
    assert first.closed
    assert second.closed


def test_strict_registration_rejects_conflict() -> None:
    registry = ToolRegistry()
    registry.register(NamedTool("Shared", "one"), source=ToolSource.PLUGIN)
    with pytest.raises(ToolConflictError, match="Shared"):
        registry.register(
            NamedTool("Shared", "two"),
            source=ToolSource.MCP,
            allow_override=False,
        )


@pytest.mark.asyncio
async def test_execute_runs_uniform_lifecycle() -> None:
    registry = ToolRegistry()
    tool = NamedTool("Lifecycle", "value")
    registry.register(tool)

    result = await registry.execute("Lifecycle", Params())
    assert result.output == "value:after"
    assert tool.events == ["before", "execute", "after"]


@pytest.mark.asyncio
async def test_error_hook_and_close_failures_do_not_mask_cleanup() -> None:
    class BrokenTool(NamedTool):
        async def execute(self, params: BaseModel) -> ToolResult:
            raise RuntimeError("original")

        async def on_error(self, params: BaseModel, error: BaseException) -> None:
            self.events.append("error")
            raise ValueError("hook failed")

        async def close(self) -> None:
            raise OSError("close failed")

    registry = ToolRegistry()
    tool = BrokenTool("Broken", "unused")
    registration = registry.register(tool, scope_id="session:test")
    with pytest.raises(RuntimeError, match="original"):
        await registry.execute("Broken", Params())
    assert tool.events == ["before", "error"]

    assert await registry.release_scope("session:test") == [registration.tool_id]
    assert registry.get("Broken") is None
    assert registry.list_release_errors() == [(registration.tool_id, "close failed")]


def test_source_metadata_deferred_schema_and_output_policy() -> None:
    registry = create_default_registry()
    builtin = registry.get_registration("ReadFile")
    assert builtin is not None and builtin.source == ToolSource.BUILTIN

    deferred = NamedTool("RemoteSearch", "ok", deferred=True, output_limit=321)
    registration = registry.register(
        deferred, source=ToolSource.MCP, scope_id="mcp:search"
    )
    assert registration.permission_name == "RemoteSearch"
    assert registration.output_limit == 321
    assert "RemoteSearch" in registry.get_deferred_tool_names()
    assert "RemoteSearch" not in {
        schema["name"] for schema in registry.get_all_schemas()
    }
    registry.mark_discovered("RemoteSearch")
    assert "RemoteSearch" in {
        schema["name"] for schema in registry.get_all_schemas()
    }


def test_agent_filter_preserves_registration_source() -> None:
    from valecode.agents.parser import AgentDef
    from valecode.agents.tool_filter import resolve_agent_tools

    parent = ToolRegistry()
    parent.register(
        NamedTool("RemoteWithoutPrefix", "ok"),
        source=ToolSource.MCP,
        scope_id="mcp:server",
    )

    child = resolve_agent_tools(
        parent,
        AgentDef(agent_type="worker", when_to_use="test"),
    )
    registration = child.get_registration("RemoteWithoutPrefix")
    assert registration is not None
    assert registration.source == ToolSource.MCP
    assert registration.scope_id == "mcp:server"
