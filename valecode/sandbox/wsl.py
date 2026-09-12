"""Windows OS sandbox backed by WSL2 and bubblewrap.

The Windows launcher delegates Bash commands to the default WSL distribution,
where bubblewrap enforces read-only mounts, explicit writable paths, process
isolation, and an optional network namespace.  If either WSL or bubblewrap is
unavailable the backend reports unavailable and callers must fail closed.
"""

from __future__ import annotations

import platform
import shlex
import shutil
import subprocess
from pathlib import Path

from valecode.sandbox import Sandbox, SandboxConfig


class WslBwrapSandbox(Sandbox):
    def __init__(self, executable: str | None = None) -> None:
        self.executable = executable or shutil.which("wsl.exe") or "wsl.exe"
        self._availability: bool | None = None

    def _run(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [self.executable, "--exec", *args],
            capture_output=True,
            text=True,
            timeout=10,
            stdin=subprocess.DEVNULL,
        )

    def available(self) -> bool:
        if self._availability is not None:
            return self._availability
        if platform.system() != "Windows" or shutil.which("wsl.exe") is None:
            self._availability = False
            return False
        try:
            probe = self._run(
                [
                    "sh",
                    "-lc",
                    "command -v bwrap >/dev/null 2>&1 && "
                    "bwrap --unshare-user --ro-bind / / -- true",
                ]
            )
            self._availability = probe.returncode == 0
        except (OSError, subprocess.SubprocessError):
            self._availability = False
        return self._availability

    def _to_wsl_path(self, path: str) -> str:
        # Callers may intentionally supply an in-guest path such as /tmp.
        if path.startswith("/") and not path.startswith("//"):
            return path
        resolved = str(Path(path).expanduser().resolve(strict=False))
        result = self._run(["wslpath", "-a", "-u", resolved])
        if result.returncode != 0 or not result.stdout.strip():
            raise RuntimeError(
                f"WSL could not translate sandbox path {resolved!r}: "
                f"{result.stderr.strip()}"
            )
        return result.stdout.strip()

    def wrap(self, command: str, config: SandboxConfig) -> str:
        writable = [self._to_wsl_path(path) for path in config.allow_write]
        protected: list[str] = []
        for configured_path in config.deny_write:
            host_path = Path(configured_path).resolve(strict=False)
            # bwrap requires bind sources to exist.  Protect the nearest
            # existing ancestor so a missing sensitive file cannot simply be
            # created through its writable parent.
            while not host_path.exists() and host_path.parent != host_path:
                host_path = host_path.parent
            translated = self._to_wsl_path(str(host_path))
            if translated not in protected:
                protected.append(translated)
        args: list[str] = [
            "bwrap",
            "--die-with-parent",
            "--new-session",
            "--unshare-user",
            "--unshare-pid",
            "--ro-bind",
            "/",
            "/",
            "--tmpfs",
            "/tmp",
        ]
        for path in writable:
            args.extend(["--bind", path, path])
        for path in protected:
            args.extend(["--ro-bind", path, path])
        if not config.network_enabled:
            args.append("--unshare-net")
        args.extend(["--proc", "/proc", "--dev", "/dev"])
        if writable:
            args.extend(["--chdir", writable[0]])
        args.extend(["--", "bash", "-c", command])

        script = shlex.join(args)
        return subprocess.list2cmdline(
            [self.executable, "--exec", "bash", "-lc", script]
        )
