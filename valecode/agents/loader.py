from __future__ import annotations

import inspect
import importlib.resources
import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

from valecode.agents.parser import AgentDef, AgentParseError, parse_agent_file

log = logging.getLogger(__name__)

PROJECT_AGENTS_DIR = ".valecode/agents"
USER_AGENTS_DIR = "~/.valecode/agents"
PLUGIN_ENTRY_POINT_GROUP = "valecode.agents"


@dataclass(frozen=True)
class AgentPluginLoadIssue:
    entry_point: str
    error: str


class AgentLoader:


    def __init__(
        self,
        work_dir: str,
        enable_verification: bool = False,
        *,
        plugin_entry_points: Iterable[Any] | None = None,
    ) -> None:
        self._work_dir = work_dir
        self._enable_verification = enable_verification
        self._agents: dict[str, AgentDef] = {}
        self._plugin_entry_points = (
            None if plugin_entry_points is None else tuple(plugin_entry_points)
        )
        self._plugin_sources: list[Path] = []
        self.plugin_issues: list[AgentPluginLoadIssue] = []


    def _scan_directory(self, path: Path, source: str) -> list[AgentDef]:
        results: list[AgentDef] = []
        if not path.is_dir():
            return results

        for entry in sorted(path.iterdir()):
            if not entry.is_file() or entry.suffix != ".md":
                continue
            try:
                agent_def = parse_agent_file(entry)
                agent_def.source = source
                agent_def.file_path = entry
                results.append(agent_def)
            except AgentParseError as e:
                log.warning("Skipping agent file %s: %s", entry, e)
        return results


    def _load_builtins(self) -> list[AgentDef]:
        results: list[AgentDef] = []
        try:
            builtins_pkg = importlib.resources.files("valecode.agents.builtins")
        except (ModuleNotFoundError, TypeError):
            log.warning("Could not load built-in agents package")
            return results

        for item in builtins_pkg.iterdir():
            if not item.name.endswith(".md"):
                continue
            try:
                raw = item.read_text(encoding="utf-8")
                from valecode.agents.parser import parse_frontmatter, _validate_agent_meta

                meta, body = parse_frontmatter(raw)
                _validate_agent_meta(meta, item.name)

                agent_def = AgentDef(
                    agent_type=meta["name"],
                    when_to_use=meta["description"],
                    system_prompt=body,
                    tools=meta.get("tools", []),
                    disallowed_tools=meta.get("disallowedTools", []),
                    model=str(meta.get("model", "inherit")),
                    max_turns=meta.get("maxTurns") or 200,  # 对齐 Go：未指定时默认 200
                    permission_mode=str(meta.get("permissionMode", "default")),
                    background=bool(meta.get("background", False)),
                    file_path=None,
                    source="builtin",
                )

                if (
                    agent_def.agent_type == "Verification"
                    and not self._enable_verification
                ):
                    continue

                results.append(agent_def)
            except (AgentParseError, Exception) as e:
                log.warning("Skipping built-in agent %s: %s", item.name, e)

        return results

    @staticmethod
    def _entry_label(entry_point: Any) -> str:
        return f"{entry_point.name} ({getattr(entry_point, 'value', '')})"

    @staticmethod
    def _entry_source(entry_point: Any) -> str:
        distribution = getattr(getattr(entry_point, "dist", None), "name", "")
        owner = distribution or getattr(entry_point, "module", "") or "unknown"
        return f"plugin:{owner}:{entry_point.name}"

    @staticmethod
    def _materialize_plugin_paths(loaded: Any) -> list[Path]:
        value = loaded() if callable(loaded) else loaded
        if inspect.isawaitable(value):
            raise TypeError("async agent plugin factories are not supported")

        if isinstance(value, (str, os.PathLike)):
            raw_paths = [value]
        elif isinstance(value, Iterable) and not isinstance(value, (bytes, dict)):
            raw_paths = list(value)
        else:
            raise TypeError(
                "entry point must provide an agent directory, or a synchronous "
                "factory returning one or more directories"
            )

        if not raw_paths or any(
            not isinstance(path, (str, os.PathLike)) for path in raw_paths
        ):
            raise TypeError("plugin factory returned an empty or invalid directory collection")

        paths = [Path(path).expanduser().resolve() for path in raw_paths]
        if any(not path.is_dir() for path in paths):
            missing = next(path for path in paths if not path.is_dir())
            raise ValueError(f"agent plugin directory does not exist: {missing}")
        return paths

    def _load_plugin_sources(self) -> list[tuple[Path, str]]:
        sources = [(path, f"plugin:{path.name}") for path in self._plugin_sources]
        entry_points = self._plugin_entry_points
        if entry_points is None:
            entry_points = metadata.entry_points(group=PLUGIN_ENTRY_POINT_GROUP)

        self.plugin_issues = []
        ordered = sorted(
            entry_points,
            key=lambda item: (str(item.name), str(getattr(item, "value", ""))),
        )
        for entry_point in ordered:
            label = self._entry_label(entry_point)
            try:
                paths = self._materialize_plugin_paths(entry_point.load())
                source = self._entry_source(entry_point)
                sources.extend((path, source) for path in paths)
                log.info(
                    "Loaded ValeCode agent plugin %s with %d source(s)",
                    label,
                    len(paths),
                )
            except Exception as exc:
                issue = AgentPluginLoadIssue(
                    label,
                    f"{type(exc).__name__}: {exc}",
                )
                self.plugin_issues.append(issue)
                log.warning("Unable to load ValeCode agent plugin %s: %s", label, exc)

        unique: list[tuple[Path, str]] = []
        seen_paths: set[Path] = set()
        for path, source in sources:
            if path in seen_paths:
                continue
            seen_paths.add(path)
            unique.append((path, source))
        return unique

    def load_all(self) -> dict[str, AgentDef]:
        seen: dict[str, AgentDef] = {}

        # 优先级 1：项目级（最高）
        project_path = Path(self._work_dir) / PROJECT_AGENTS_DIR
        for agent_def in self._scan_directory(project_path, "project"):
            if agent_def.agent_type not in seen:
                seen[agent_def.agent_type] = agent_def

        # 优先级 2：用户级
        user_path = Path(USER_AGENTS_DIR).expanduser()
        for agent_def in self._scan_directory(user_path, "user"):
            if agent_def.agent_type not in seen:
                seen[agent_def.agent_type] = agent_def

        # 优先级 3：内置
        for agent_def in self._load_builtins():
            if agent_def.agent_type not in seen:
                seen[agent_def.agent_type] = agent_def

        # 优先级 4：插件（最低）
        for path, source in self._load_plugin_sources():
            for agent_def in self._scan_directory(path, source):
                if agent_def.agent_type not in seen:
                    seen[agent_def.agent_type] = agent_def

        self._agents = seen
        return seen


    def get(self, agent_type: str) -> AgentDef | None:
        cached = self._agents.get(agent_type)
        if cached is None:
            return None

        # 从文件热重载
        if cached.file_path is not None and cached.file_path.exists():
            try:
                reloaded = parse_agent_file(cached.file_path)
                reloaded.source = cached.source
                self._agents[agent_type] = reloaded
                return reloaded
            except AgentParseError as e:
                log.warning(
                    "Hot reload failed for %s, using cached: %s",
                    agent_type,
                    e,
                )
        return cached


    def list_agents(self) -> list[tuple[str, str]]:
        return [
            (ad.agent_type, ad.when_to_use) for ad in self._agents.values()
        ]

    def register_plugin_source(self, path: Path) -> None:
        resolved = path.expanduser().resolve()
        if not resolved.is_dir():
            raise ValueError(f"agent plugin directory does not exist: {resolved}")
        if resolved not in self._plugin_sources:
            self._plugin_sources.append(resolved)
