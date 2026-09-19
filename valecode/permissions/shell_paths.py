"""Conservative path extraction for shell permission checks.

This is intentionally smaller than a shell AST. It recognizes explicit paths
and common file commands; uncertain dynamic paths are reported for approval.
The OS sandbox remains the execution boundary when configured.
"""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass

from valecode.permissions.sandbox import PathSandbox


_FILE_COMMANDS = frozenset({
    "cd", "chdir", "pushd", "popd", "set-location", "push-location",
    "rm", "cp", "mv", "mkdir", "touch", "chmod", "chown", "cat",
    "head", "tail", "ls", "dir", "type", "del", "erase", "rd",
    "ren", "rename", "rmdir", "copy", "move", "md",
    "get-content", "set-content", "add-content", "copy-item", "move-item",
    "remove-item", "new-item", "rename-item",
})
_MODE_PREFIX_COMMANDS = frozenset({"chmod", "chown"})
_COMMAND_SPLIT_RE = re.compile(r"&&|\|\||[;&|\r\n]")
_REDIRECT_RE = re.compile(r"(?:^|\s)(?:>{1,2}|<)\s*(\"[^\"]+\"|'[^']+'|\S+)")
_ENV_RE = re.compile(r"\$(?:\{|[A-Za-z_])|%[A-Za-z_][A-Za-z0-9_]*%|`|\$\(")
_WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]|^\\\\")


@dataclass(frozen=True)
class ShellPathIssue:
    value: str
    reason: str


def _tokens(segment: str) -> list[str]:
    try:
        # posix=False preserves Windows backslashes. Quotes are stripped below.
        values = shlex.split(segment, posix=False)
    except ValueError:
        return []
    return [value[1:-1] if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'" else value for value in values]


def _looks_explicit_path(value: str) -> bool:
    return (
        value.startswith(("/", "~/", "~\\", "./", ".\\", "../", "..\\"))
        or _WINDOWS_ABSOLUTE_RE.match(value) is not None
    )


def _path_arguments(tokens: list[str]) -> list[str]:
    if not tokens:
        return []
    command = os.path.basename(tokens[0]).casefold()
    args = tokens[1:]
    if command not in _FILE_COMMANDS:
        return [value for value in args if _looks_explicit_path(value)]

    values: list[str] = []
    skip_mode = command in _MODE_PREFIX_COMMANDS
    for value in args:
        if value == "--":
            continue
        if value.startswith("-") and not _looks_explicit_path(value):
            # Support PowerShell's -Path:C:\x and GNU --output=/x forms.
            if ":" in value:
                candidate = value.split(":", 1)[1]
                if candidate:
                    values.append(candidate)
            elif "=" in value:
                candidate = value.split("=", 1)[1]
                if candidate:
                    values.append(candidate)
            continue
        if skip_mode:
            skip_mode = False
            continue
        values.append(value)
    return values


def find_shell_path_issues(command: str, sandbox: PathSandbox) -> list[ShellPathIssue]:
    """Return explicit paths outside the project/temp roots or dynamic paths."""
    candidates: list[str] = []
    for segment in _COMMAND_SPLIT_RE.split(command):
        tokens = _tokens(segment.strip())
        candidates.extend(_path_arguments(tokens))
        candidates.extend(match.group(1).strip("\"'") for match in _REDIRECT_RE.finditer(segment))

    issues: list[ShellPathIssue] = []
    seen: set[str] = set()
    for value in candidates:
        value = value.rstrip(",")
        if not value or value in seen:
            continue
        seen.add(value)
        if value.startswith(("http://", "https://")):
            continue
        if _ENV_RE.search(value):
            issues.append(ShellPathIssue(value, "动态路径无法在执行前验证"))
            continue
        ok, reason = sandbox.check(value)
        if not ok:
            issues.append(ShellPathIssue(value, reason))
    return issues
