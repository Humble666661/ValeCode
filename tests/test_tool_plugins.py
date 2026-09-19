from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from valecode.tools import ToolRegistry, ToolSource
from valecode.tools.base import Tool, ToolResult
from valecode.tools.plugins import load_plugin_tools


class _Params(BaseModel):
    pass


class _PluginTool(Tool):
    name = "PluginTool"
    description = "plugin test tool"
    params_model = _Params

    def __init__(self, name: str | None = None) -> None:
        if name is not None:
            self.name = name
        self.closed = False

    async def execute(self, params: _Params) -> ToolResult:
        return ToolResult("ok")

    async def close(self) -> None:
        self.closed = True


class _EntryPoint:
    def __init__(
        self, name: str, value: str, loaded: Any = None,
        error: Exception | None = None,
    ) -> None:
        self.name = name
        self.value = value
        self.module = value.split(":", 1)[0]
        self.dist = SimpleNamespace(name="demo-dist")
        self._loaded = loaded
        self._error = error

    def load(self) -> Any:
        if self._error is not None:
            raise self._error
        return self._loaded


def test_loads_instance_class_and_factory_in_deterministic_scopes() -> None:
    registry = ToolRegistry()
    entries = [
        _EntryPoint("z_factory", "demo:factory", lambda: [_PluginTool("Factory")]),
        _EntryPoint("a_instance", "demo:instance", _PluginTool("Instance")),
        _EntryPoint("m_class", "demo:class", _PluginTool),
    ]

    result = load_plugin_tools(registry, entry_points=entries)

    assert result.issues == []
    assert [item.name for item in result.registrations] == [
        "Instance", "PluginTool", "Factory",
    ]
    assert all(item.source == ToolSource.PLUGIN for item in result.registrations)
    assert result.registrations[0].scope_id == "plugin:demo-dist:a_instance"


def test_broken_plugin_is_reported_without_blocking_valid_plugins() -> None:
    registry = ToolRegistry()
    result = load_plugin_tools(registry, entry_points=[
        _EntryPoint("broken", "bad:tool", error=RuntimeError("boom")),
        _EntryPoint("invalid", "bad:value", loaded="not a tool"),
        _EntryPoint("valid", "good:tool", loaded=_PluginTool("Valid")),
    ])

    assert registry.get("Valid") is not None
    assert [issue.entry_point for issue in result.issues] == [
        "broken (bad:tool)", "invalid (bad:value)",
    ]
    assert "RuntimeError: boom" in result.issues[0].error


@pytest.mark.asyncio
async def test_release_source_closes_plugin_and_reveals_builtin() -> None:
    registry = ToolRegistry()
    builtin = _PluginTool("Shared")
    plugin = _PluginTool("Shared")
    registry.register(builtin, source=ToolSource.BUILTIN, scope_id="builtin")
    load_plugin_tools(
        registry,
        entry_points=[_EntryPoint("override", "demo:override", plugin)],
    )
    assert registry.get("Shared") is plugin

    released = await registry.release_source(ToolSource.PLUGIN)

    assert released == ["plugin:plugin:demo-dist:override:Shared"]
    assert plugin.closed is True
    assert builtin.closed is False
    assert registry.get("Shared") is builtin
