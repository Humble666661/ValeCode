
from __future__ import annotations

import posixpath
import re
import shlex

_DANGEROUS_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"rm\s+-[a-z]*r[a-z]*f[a-z]*\s+/\s*$"), "递归强制删除根目录"),
    (re.compile(r"mkfs\."), "格式化磁盘"),
    (re.compile(r"dd\s+if=.*of=/dev/"), "直接写磁盘设备"),
    (re.compile(r"chmod\s+-R\s+777\s+/"), "递归修改根目录权限"),
    (re.compile(r":\(\)\{\s*:\|:&\s*\};:"), "fork bomb"),
    (re.compile(r"curl\s+.*\|\s*(ba)?sh"), "管道执行远程脚本"),
    (re.compile(r"wget\s+.*\|\s*(ba)?sh"), "管道执行远程脚本"),
    (re.compile(r">\s*/dev/sd"), "覆盖磁盘设备"),
]


_SAFE_EXACT_COMMANDS = frozenset({
    "pwd", "whoami", "hostname", "uname", "date", "uptime",
    "true", "false", "git status", "git log", "git diff",
    "git show", "git stash list", "go version", "node -v",
    "npm -v", "python --version", "cargo --version",
    "rustc --version", "java -version", "java --version",
})

_SAFE_LIST_FLAGS = frozenset({"-a", "-l", "-la", "-al", "-h", "-lh", "-lah", "-alh"})
_SAFE_GIT_STATUS_FLAGS = frozenset({"--short", "--branch", "--porcelain", "-sb"})


def is_safe_command(command: str) -> bool:
    trimmed = command.strip()
    if not trimmed:
        return False
    # This is an auto-allow list, not a command parser. Anything with shell
    # composition, expansion or redirection must go through normal approval.
    if any(ch in trimmed for ch in "|;&><`$\r\n\\"):
        return False
    normalized = " ".join(trimmed.split())
    if normalized in _SAFE_EXACT_COMMANDS:
        return True
    parts = normalized.split(" ")
    if parts[0] in ("ls", "dir"):
        return all(part in _SAFE_LIST_FLAGS for part in parts[1:])
    if parts[:2] == ["git", "status"]:
        return all(part in _SAFE_GIT_STATUS_FLAGS for part in parts[2:])
    return False


class DangerousCommandDetector:


    def __init__(self, extra_patterns: list[tuple[str, str]] | None = None) -> None:
        self._patterns = list(_DANGEROUS_PATTERNS)
        if extra_patterns:
            for regex_str, reason in extra_patterns:
                self._patterns.append((re.compile(regex_str), reason))


    def detect(self, command: str) -> tuple[bool, str]:
        if _deletes_posix_root(command):
            return True, "递归删除根目录"
        for pattern, reason in self._patterns:
            if pattern.search(command):
                return True, reason
        return False, ""


def _deletes_posix_root(command: str) -> bool:
    """Catch common rm flag/order/quoting variants targeting / or /*.

    This is a catastrophic-operation guard, not a general shell parser. Other
    commands still go through the normal permission decision pipeline.
    """
    for segment in re.split(r"&&|\|\||[;|\r\n]", command):
        try:
            parts = shlex.split(segment)
        except ValueError:
            continue
        if not parts:
            continue
        if parts[0] in ("sudo", "command"):
            parts = parts[1:]
        if not parts or parts[0] != "rm":
            continue
        recursive = False
        targets: list[str] = []
        options_done = False
        for part in parts[1:]:
            if part == "--" and not options_done:
                options_done = True
            elif not options_done and part.startswith("--"):
                recursive |= part == "--recursive"
            elif not options_done and part.startswith("-"):
                recursive |= "r" in part[1:] or "R" in part[1:]
            else:
                targets.append(part)
        if recursive and any(
            posixpath.normpath(target) in ("/", "/*") for target in targets
        ):
            return True
    return False
