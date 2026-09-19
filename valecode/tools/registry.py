from __future__ import annotations

import inspect
import itertools
import logging
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import StrEnum
from typing import Any

from pydantic import BaseModel

from valecode.tools.base import Tool, ToolResult

log = logging.getLogger(__name__)


class ToolSource(StrEnum):
    BUILTIN = "builtin"
    PLUGIN = "plugin"
    MCP = "mcp"
    SESSION = "session"


_SOURCE_PRIORITY = {
    ToolSource.BUILTIN: 10,
    ToolSource.PLUGIN: 20,
    ToolSource.MCP: 30,
    ToolSource.SESSION: 40,
}


@dataclass(frozen=True)
class ToolRegistration:
    tool_id: str
    tool: Tool
    source: ToolSource
    scope_id: str
    sequence: int
    permission_name: str
    output_limit: int

    @property
    def name(self) -> str:
        return self.tool.name

    @property
    def priority(self) -> int:
        return _SOURCE_PRIORITY[self.source]


@dataclass(frozen=True)
class ToolConflict:
    name: str
    previous_tool_id: str
    replacement_tool_id: str
    winner_tool_id: str


class ToolConflictError(ValueError):
    pass


class ToolRegistry:
    """Layered registry with deterministic override and scope cleanup.

    Precedence is Built-in < Plugin < MCP < Session. Registrations are retained
    underneath the active entry, so releasing an upper scope reveals the prior
    implementation without rebuilding the registry.
    """

    def __init__(self) -> None:
        self._registrations: dict[str, list[ToolRegistration]] = {}
        self._retired: list[ToolRegistration] = []
        self._disabled: set[str] = set()
        self._discovered: set[str] = set()
        self._sequence = itertools.count(1)
        self._conflicts: list[ToolConflict] = []
        self._release_errors: list[tuple[str, str]] = []
        self._session_scope_id = "session"

    def bind_session(self, session_id: str) -> None:
        self._session_scope_id = f"session:{session_id}"

    @staticmethod
    def _source(value: ToolSource | str) -> ToolSource:
        try:
            return value if isinstance(value, ToolSource) else ToolSource(value)
        except ValueError as exc:
            raise ValueError(f"Unknown tool source: {value}") from exc

    def _default_scope(self, source: ToolSource) -> str:
        if source == ToolSource.SESSION:
            return self._session_scope_id
        return source.value

    @staticmethod
    def _tool_id(source: ToolSource, scope_id: str, name: str) -> str:
        return f"{source.value}:{scope_id}:{name}"

    @staticmethod
    def _sort_key(registration: ToolRegistration) -> tuple[int, int]:
        return registration.priority, registration.sequence

    def register(
        self,
        tool: Tool,
        *,
        source: ToolSource | str = ToolSource.SESSION,
        scope_id: str | None = None,
        allow_override: bool = True,
    ) -> ToolRegistration:
        resolved_source = self._source(source)
        resolved_scope = scope_id or self._default_scope(resolved_source)
        sequence = next(self._sequence)
        registration = ToolRegistration(
            tool_id=self._tool_id(resolved_source, resolved_scope, tool.name),
            tool=tool,
            source=resolved_source,
            scope_id=resolved_scope,
            sequence=sequence,
            permission_name=tool.permission_name,
            output_limit=tool.max_output_chars,
        )
        entries = self._registrations.setdefault(tool.name, [])
        previous = self.get_registration(tool.name)
        same_slot = next(
            (
                entry
                for entry in entries
                if entry.source == resolved_source
                and entry.scope_id == resolved_scope
            ),
            None,
        )
        if (previous is not None or same_slot is not None) and not allow_override:
            raise ToolConflictError(
                f"Tool '{tool.name}' conflicts with "
                f"{(same_slot or previous).tool_id}"
            )
        if same_slot is not None:
            entries.remove(same_slot)
            # register() is synchronous while resource cleanup may be async.
            # Keep the replaced instance reachable until its scope is released.
            self._retired.append(same_slot)
        entries.append(registration)
        active = self.get_registration(tool.name)
        if previous is not None:
            self._conflicts.append(
                ToolConflict(
                    name=tool.name,
                    previous_tool_id=previous.tool_id,
                    replacement_tool_id=registration.tool_id,
                    winner_tool_id=active.tool_id if active else registration.tool_id,
                )
            )
        if previous is None or active is None or previous.tool is not active.tool:
            self._discovered.discard(tool.name)
        return registration

    def unregister(
        self,
        name: str,
        *,
        source: ToolSource | str | None = None,
        scope_id: str | None = None,
    ) -> list[ToolRegistration]:
        previous = self.get_registration(name)
        entries = self._registrations.get(name, [])
        resolved_source = self._source(source) if source is not None else None
        removed = [
            entry
            for entry in entries
            if (resolved_source is None or entry.source == resolved_source)
            and (scope_id is None or entry.scope_id == scope_id)
        ]
        kept = [entry for entry in entries if entry not in removed]
        if kept:
            self._registrations[name] = kept
        else:
            self._registrations.pop(name, None)
            self._disabled.discard(name)
        active = self.get_registration(name)
        if previous is None or active is None or previous.tool is not active.tool:
            self._discovered.discard(name)
        return removed

    def get_registration(self, name: str) -> ToolRegistration | None:
        entries = self._registrations.get(name)
        return max(entries, key=self._sort_key) if entries else None

    def get(self, name: str) -> Tool | None:
        registration = self.get_registration(name)
        return registration.tool if registration else None

    def is_enabled(self, name: str) -> bool:
        return self.get_registration(name) is not None and name not in self._disabled

    def enable(self, name: str) -> None:
        self._disabled.discard(name)

    def disable(self, name: str) -> None:
        if self.get_registration(name) is not None:
            self._disabled.add(name)

    def enable_all(self) -> None:
        self._disabled.clear()

    def mark_discovered(self, name: str) -> None:
        if self.get_registration(name) is not None:
            self._discovered.add(name)

    def is_discovered(self, name: str) -> bool:
        return name in self._discovered

    def list_registrations(self, *, active_only: bool = True) -> list[ToolRegistration]:
        if active_only:
            return [
                registration
                for name in self._registrations
                if (registration := self.get_registration(name)) is not None
            ]
        return [entry for entries in self._registrations.values() for entry in entries]

    def list_conflicts(self) -> list[ToolConflict]:
        return list(self._conflicts)

    def list_release_errors(self) -> list[tuple[str, str]]:
        return list(self._release_errors)

    def list_tools(self) -> list[Tool]:
        return [registration.tool for registration in self.list_registrations()]

    def get_deferred_tool_names(self) -> list[str]:
        return [
            registration.name
            for registration in self.list_registrations()
            if registration.tool.should_defer
            and registration.name not in self._discovered
            and registration.name not in self._disabled
        ]

    def search_deferred(
        self, query: str, max_results: int, protocol: str = "anthropic"
    ) -> list[dict[str, Any]]:
        query = query.strip()
        if not query or max_results <= 0:
            return []

        required_name = ""
        ranking_query = query
        if query.startswith("+"):
            parts = query[1:].split(None, 1)
            if not parts or not parts[0]:
                return []
            required_name = self._normalize_search_text(parts[0])
            ranking_query = parts[1] if len(parts) > 1 else parts[0]

        query_text = self._normalize_search_text(ranking_query)
        query_compact = query_text.replace(" ", "")
        query_tokens = set(query_text.split())
        scored: list[tuple[int, float, str, Tool]] = []
        for registration in self.list_registrations():
            name, tool = registration.name, registration.tool
            if (
                not tool.should_defer
                or name in self._disabled
                or name in self._discovered
            ):
                continue

            name_text = self._normalize_search_text(name)
            if required_name and required_name not in name_text:
                continue
            aliases = [
                self._normalize_search_text(term)
                for term in getattr(tool, "search_terms", ())
                if term
            ]
            description = self._normalize_search_text(tool.description or "")
            searchable = " ".join([name_text, description, *aliases]).strip()
            searchable_tokens = set(searchable.split())
            score = 0
            if query_text == name_text:
                score += 200
            if query_compact and query_compact == name_text.replace(" ", ""):
                score += 180
            if query_text and query_text in name_text:
                score += 100
            if query_text and query_text in description:
                score += 60
            if query_text and query_text in aliases:
                score += 120
            score += 18 * len(query_tokens & set(name_text.split()))
            score += 8 * len(query_tokens & searchable_tokens)

            fuzzy = max(
                [
                    SequenceMatcher(
                        None, query_compact, candidate.replace(" ", "")
                    ).ratio()
                    for candidate in [name_text, *aliases]
                    if candidate and query_compact
                ],
                default=0.0,
            )
            if fuzzy >= 0.72:
                score += round(fuzzy * 50)
            if score > 0:
                scored.append((score, fuzzy, name.casefold(), tool))
        scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
        return [
            self._schema(tool, protocol)
            for _, _, _, tool in scored[:min(max_results, 20)]
        ]

    @staticmethod
    def _normalize_search_text(value: str) -> str:
        value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value)
        value = re.sub(r"[^\w]+", " ", value, flags=re.UNICODE)
        return " ".join(value.casefold().split())

    def find_deferred_by_names(
        self, names: list[str], protocol: str = "anthropic"
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for name in names:
            tool = self.get(name)
            if tool is not None and tool.should_defer:
                results.append(self._schema(tool, protocol))
        return results

    def get_all_schemas(self, protocol: str = "anthropic") -> list[dict[str, Any]]:
        return [
            self._schema(registration.tool, protocol)
            for registration in self.list_registrations()
            if registration.name not in self._disabled
            and (
                not registration.tool.should_defer
                or registration.name in self._discovered
            )
        ]

    @staticmethod
    def _schema(tool: Tool, protocol: str) -> dict[str, Any]:
        base = tool.get_schema()
        if protocol in ("openai", "openai-compat"):
            return {
                "type": "function",
                "name": base["name"],
                "description": base["description"],
                "parameters": base["input_schema"],
            }
        return base

    async def execute(self, name: str, params: BaseModel) -> ToolResult:
        registration = self.get_registration(name)
        if registration is None:
            raise KeyError(f"Unknown tool: {name}")
        if name in self._disabled:
            raise PermissionError(f"Tool is disabled: {name}")
        tool = registration.tool
        try:
            await tool.before_execute(params)
            result = await tool.execute(params)
            return await tool.after_execute(params, result)
        except BaseException as exc:
            try:
                await tool.on_error(params, exc)
            except Exception:
                log.exception("Tool error hook failed: %s", registration.tool_id)
            raise

    async def release_scope(self, scope_id: str) -> list[str]:
        active = [
            entry
            for entry in self.list_registrations(active_only=False)
            if entry.scope_id == scope_id
        ]
        retired = [entry for entry in self._retired if entry.scope_id == scope_id]
        self._retired = [
            entry for entry in self._retired if entry.scope_id != scope_id
        ]
        removed = [*active, *retired]
        for entry in reversed(removed):
            is_active = entry in active
            try:
                await self._close_tool(entry.tool)
            except Exception as exc:
                self._release_errors.append((entry.tool_id, str(exc)))
                log.warning("Tool close failed for %s: %s", entry.tool_id, exc)
            finally:
                if is_active:
                    active.remove(entry)
                    self.unregister(
                        entry.name, source=entry.source, scope_id=entry.scope_id
                    )
        return list(dict.fromkeys(entry.tool_id for entry in removed))

    async def release_session(self) -> list[str]:
        return await self.release_scope(self._session_scope_id)

    async def release_source(self, source: ToolSource | str) -> list[str]:
        resolved_source = self._source(source)
        scopes = list(dict.fromkeys(
            entry.scope_id
            for entry in [
                *self.list_registrations(active_only=False),
                *self._retired,
            ]
            if entry.source == resolved_source
        ))
        released: list[str] = []
        for scope_id in scopes:
            released.extend(await self.release_scope(scope_id))
        return list(dict.fromkeys(released))

    @staticmethod
    async def _close_tool(tool: Tool) -> None:
        result = tool.close()
        if inspect.isawaitable(result):
            await result

    def copy_registration_to(
        self, target: ToolRegistry, name: str
    ) -> ToolRegistration | None:
        registration = self.get_registration(name)
        if registration is None:
            return None
        return target.register(
            registration.tool,
            source=registration.source,
            scope_id=registration.scope_id,
        )
