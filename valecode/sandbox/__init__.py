
"""OS 级沙箱：限制 Bash 命令的文件写入和网络访问。

macOS 使用 sandbox-exec（Seatbelt），Linux 使用 bubblewrap（bwrap）。
与 permissions/sandbox.py 中的 PathSandbox（路径级权限检查）不同，
这里是操作系统层面的强制隔离——即使命令绕过了路径检查，
内核也会阻止越权的写操作。
"""

from __future__ import annotations

import platform
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SandboxConfig:
    """沙箱配置：控制可写路径、禁写路径和网络访问。"""

    # 允许写入的路径列表（白名单）
    allow_write: list[str] = field(default_factory=list)
    # 强制只读的路径列表（优先级高于 allow_write）
    deny_write: list[str] = field(default_factory=list)
    # 是否允许网络访问
    network_enabled: bool = False


class Sandbox(ABC):
    """沙箱抽象基类，各平台实现 wrap() 和 available()。"""

    @abstractmethod
    def wrap(self, command: str, config: SandboxConfig) -> str:
        """将原始命令包装为沙箱内执行的命令字符串。"""
        ...

    @abstractmethod
    def available(self) -> bool:
        """检测当前环境是否支持该沙箱。"""
        ...


def create_sandbox() -> Sandbox | None:
    """根据操作系统自动选择沙箱实现。

    macOS -> SeatbeltSandbox（基于 sandbox-exec）
    Linux -> BwrapSandbox（基于 bubblewrap）
    Windows -> WslBwrapSandbox（基于 WSL2 + bubblewrap）
    """
    system = platform.system()
    if system == "Darwin":
        from .seatbelt import SeatbeltSandbox
        return SeatbeltSandbox()
    elif system == "Linux":
        from .bwrap import BwrapSandbox
        return BwrapSandbox()
    elif system == "Windows":
        from .wsl import WslBwrapSandbox
        return WslBwrapSandbox()
    return None


def attach_sandbox(
    registry: object,
    checker: object,
    work_dir: str,
    *,
    network_enabled: bool = False,
    auto_allow: bool = False,
) -> tuple[bool, str]:
    """Attach a verified backend and enable auto-allow only after verification."""
    sandbox = create_sandbox()
    if sandbox is None:
        return False, "no sandbox backend exists for this platform"
    bash_tool = getattr(registry, "get")("Bash")
    if bash_tool is None:
        return False, "Bash tool is not registered"
    bash_tool.sandbox = sandbox
    bash_tool.sandbox_config = SandboxConfig(
        allow_write=[work_dir, tempfile.gettempdir()],
        deny_write=[
            str(Path(work_dir) / ".valecode" / "config.yaml"),
            str(Path(work_dir) / ".valecode" / "permissions.local.yaml"),
        ],
        network_enabled=network_enabled,
    )
    if not sandbox.available():
        # Keep the configured backend attached so Bash fails closed instead of
        # silently executing outside the sandbox requested by the user.
        if checker is not None:
            checker.sandbox_enabled = False
        return False, f"sandbox backend {type(sandbox).__name__} is unavailable"
    if checker is not None:
        checker.sandbox_enabled = auto_allow
    return True, type(sandbox).__name__
