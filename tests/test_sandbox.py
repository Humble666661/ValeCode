from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from valecode.sandbox import Sandbox, SandboxConfig, attach_sandbox, create_sandbox
from valecode.sandbox.wsl import WslBwrapSandbox
from valecode.tools import create_default_registry
from valecode.tools.bash import Bash, Params


class _UnavailableSandbox(Sandbox):
    def wrap(self, command: str, config: SandboxConfig) -> str:
        raise AssertionError("wrap must not run for an unavailable backend")

    def available(self) -> bool:
        return False


class _AvailableSandbox(Sandbox):
    def wrap(self, command: str, config: SandboxConfig) -> str:
        return command

    def available(self) -> bool:
        return True


def test_create_sandbox_selects_windows_wsl_backend(monkeypatch) -> None:
    monkeypatch.setattr("valecode.sandbox.platform.system", lambda: "Windows")
    assert isinstance(create_sandbox(), WslBwrapSandbox)


def test_wsl_backend_probes_bwrap_enforcement(monkeypatch) -> None:
    monkeypatch.setattr("valecode.sandbox.wsl.platform.system", lambda: "Windows")
    monkeypatch.setattr(
        "valecode.sandbox.wsl.shutil.which", lambda name: "C:/Windows/wsl.exe"
    )
    sandbox = WslBwrapSandbox("C:/Windows/wsl.exe")
    calls: list[list[str]] = []

    def run(args: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, "/usr/bin/bwrap\n", "")

    monkeypatch.setattr(sandbox, "_run", run)
    assert sandbox.available() is True
    assert "bwrap --unshare-user --ro-bind / / -- true" in calls[0][-1]


def test_wsl_wrapper_uses_mount_process_and_network_namespaces(
    monkeypatch, tmp_path
) -> None:
    sandbox = WslBwrapSandbox("C:/Windows/wsl.exe")
    translations = {
        str(tmp_path): "/mnt/c/project",
        str(tmp_path / "protected"): "/mnt/c/project/protected",
    }
    (tmp_path / "protected").mkdir()
    monkeypatch.setattr(sandbox, "_to_wsl_path", translations.__getitem__)

    wrapped = sandbox.wrap(
        "printf 'hello world'",
        SandboxConfig(
            allow_write=[str(tmp_path)],
            deny_write=[str(tmp_path / "protected")],
            network_enabled=False,
        ),
    )

    assert "wsl.exe" in wrapped
    assert "bwrap" in wrapped
    assert "--ro-bind / /" in wrapped
    assert "--bind /mnt/c/project /mnt/c/project" in wrapped
    assert "--ro-bind /mnt/c/project/protected" in wrapped
    assert "--unshare-pid" in wrapped
    assert "--unshare-net" in wrapped
    assert "--chdir /mnt/c/project" in wrapped


def test_attach_sandbox_never_auto_allows_unavailable_backend(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        "valecode.sandbox.create_sandbox", lambda: _UnavailableSandbox()
    )
    registry = create_default_registry()
    checker = SimpleNamespace(sandbox_enabled=False)

    attached, _ = attach_sandbox(
        registry, checker, str(tmp_path), auto_allow=True
    )

    assert attached is False
    assert checker.sandbox_enabled is False
    assert isinstance(registry.get("Bash").sandbox, _UnavailableSandbox)
    assert registry.get("Bash").sandbox_config is not None


def test_attach_sandbox_enables_auto_allow_only_after_attach(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        "valecode.sandbox.create_sandbox", lambda: _AvailableSandbox()
    )
    registry = create_default_registry()
    checker = SimpleNamespace(sandbox_enabled=False)

    attached, backend = attach_sandbox(
        registry, checker, str(tmp_path), auto_allow=True
    )

    assert attached is True
    assert backend == "_AvailableSandbox"
    assert checker.sandbox_enabled is True
    assert registry.get("Bash").sandbox_config is not None


@pytest.mark.asyncio
async def test_bash_fails_closed_when_configured_backend_is_unavailable() -> None:
    bash = Bash()
    bash.sandbox = _UnavailableSandbox()
    bash.sandbox_config = SandboxConfig(allow_write=["."])

    result = await bash.execute(Params(command="echo must-not-run"))

    assert result.is_error is True
    assert "command was not executed" in result.output
