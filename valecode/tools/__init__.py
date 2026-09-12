from __future__ import annotations

from typing import TYPE_CHECKING, Any

from valecode.tools.registry import (
    ToolConflict,
    ToolConflictError,
    ToolRegistration,
    ToolRegistry,
    ToolSource,
)

if TYPE_CHECKING:
    from valecode.cache import FileCache


def create_default_registry(
    file_cache: FileCache | None = None, file_history: Any = None
) -> ToolRegistry:
    from valecode.tools.bash import Bash
    from valecode.tools.edit_file import EditFile
    from valecode.tools.file_state_cache import FileStateCache
    from valecode.tools.glob import Glob
    from valecode.tools.grep import Grep
    from valecode.tools.read_file import ReadFile
    from valecode.tools.write_file import WriteFile

    file_state_cache = FileStateCache()
    registry = ToolRegistry()
    for tool in (
        ReadFile(file_cache=file_cache, file_state_cache=file_state_cache),
        WriteFile(
            file_cache=file_cache,
            file_history=file_history,
            file_state_cache=file_state_cache,
        ),
        EditFile(
            file_cache=file_cache,
            file_history=file_history,
            file_state_cache=file_state_cache,
        ),
        Bash(),
        Glob(),
        Grep(),
    ):
        registry.register(tool, source=ToolSource.BUILTIN, scope_id="builtin")
    return registry


__all__ = [
    "ToolConflict",
    "ToolConflictError",
    "ToolRegistration",
    "ToolRegistry",
    "ToolSource",
    "create_default_registry",
]
