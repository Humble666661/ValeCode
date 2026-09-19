from __future__ import annotations

import inspect
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from importlib import metadata
from typing import Any

from valecode.tools.base import Tool
from valecode.tools.registry import ToolRegistration, ToolRegistry, ToolSource

log = logging.getLogger(__name__)

PLUGIN_ENTRY_POINT_GROUP = "valecode.tools"


@dataclass(frozen=True)
class PluginLoadIssue:
    entry_point: str
    error: str


@dataclass
class PluginLoadResult:
    registrations: list[ToolRegistration] = field(default_factory=list)
    issues: list[PluginLoadIssue] = field(default_factory=list)


def _entry_scope(entry_point: Any) -> str:
    distribution = getattr(getattr(entry_point, "dist", None), "name", "")
    owner = distribution or getattr(entry_point, "module", "") or "unknown"
    return f"plugin:{owner}:{entry_point.name}"


def _materialize_tools(loaded: Any) -> list[Tool]:
    value = loaded
    if inspect.isclass(value) and issubclass(value, Tool):
        value = value()
    elif not isinstance(value, Tool) and callable(value):
        value = value()

    if inspect.isawaitable(value):
        raise TypeError("async plugin factories are not supported")
    if isinstance(value, Tool):
        tools = [value]
    elif isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
        tools = list(value)
    else:
        raise TypeError(
            "entry point must provide a Tool, Tool class, or synchronous Tool iterable"
        )

    if not tools or any(not isinstance(tool, Tool) for tool in tools):
        raise TypeError("plugin factory returned an empty or invalid tool collection")
    names = [tool.name for tool in tools]
    if len(names) != len(set(names)):
        raise ValueError("plugin returned duplicate tool names")
    return tools


def load_plugin_tools(
    registry: ToolRegistry, *, entry_points: Iterable[Any] | None = None,
) -> PluginLoadResult:
    """Load trusted Python tool plugins registered under ``valecode.tools``.

    Loading is fail-open per entry point so one broken optional plugin does not
    prevent ValeCode from starting. Loaded tools still use the normal registry,
    permission checker, hook, timeout, and output-budget paths.
    """
    if entry_points is None:
        entry_points = metadata.entry_points(group=PLUGIN_ENTRY_POINT_GROUP)
    result = PluginLoadResult()
    ordered = sorted(
        entry_points,
        key=lambda item: (str(item.name), str(getattr(item, "value", ""))),
    )
    for entry_point in ordered:
        label = f"{entry_point.name} ({getattr(entry_point, 'value', '')})"
        try:
            tools = _materialize_tools(entry_point.load())
            scope_id = _entry_scope(entry_point)
            for tool in tools:
                result.registrations.append(registry.register(
                    tool,
                    source=ToolSource.PLUGIN,
                    scope_id=scope_id,
                ))
            log.info("Loaded ValeCode tool plugin %s with %d tool(s)", label, len(tools))
        except Exception as exc:
            issue = PluginLoadIssue(label, f"{type(exc).__name__}: {exc}")
            result.issues.append(issue)
            log.warning("Unable to load ValeCode tool plugin %s: %s", label, exc)
    return result
