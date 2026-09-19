"""The 'don't ask again' choice is durable only within its session."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from valecode.agent import Agent, PermissionRequest, PermissionResponse
from valecode.app import ValeCodeApp
from valecode.conversation import ConversationManager
from valecode.memory.session import SessionManager
from valecode.permissions import (
    DangerousCommandDetector,
    PathSandbox,
    PermissionChecker,
    PermissionMode,
    RuleEngine,
)
from valecode.permissions.session_store import SessionAllowStore
from valecode.tools import create_default_registry
from valecode.tools.base import StreamEnd, TextDelta, ToolCallComplete
from valecode.tools.bash import Bash


def _checker(root: Path, *, project_rules: Path | None = None) -> PermissionChecker:
    return PermissionChecker(
        detector=DangerousCommandDetector(),
        sandbox=PathSandbox(str(root)),
        rule_engine=RuleEngine(
            project_rules_path=project_rules,
            local_rules_path=root / ".valecode" / "permissions.local.yaml",
        ),
        mode=PermissionMode.DEFAULT,
    )


def test_session_grant_is_exact_isolated_and_restored(tmp_path: Path) -> None:
    checker = _checker(tmp_path)
    tool = Bash()
    first = {"command": "echo hello"}
    different = {"command": "echo goodbye"}
    checker.bind_session("session-one")
    assert checker.check(tool, first).effect == "ask"
    checker.add_session_allow("Bash", first)
    assert checker.check(tool, first).effect == "allow"
    assert checker.check(tool, different).effect == "ask"
    assert not (tmp_path / ".valecode" / "permissions.local.yaml").exists()

    checker.bind_session("session-two")
    assert checker.check(tool, first).effect == "ask"
    checker.bind_session("session-one")
    assert checker.check(tool, first).effect == "allow"
    assert _checker(tmp_path).check(tool, first).effect == "ask"

    restarted = _checker(tmp_path)
    restarted.bind_session("session-one")
    assert restarted.check(tool, first).effect == "allow"
    assert "echo hello" not in SessionAllowStore(tmp_path, "session-one").path.read_text(
        encoding="utf-8"
    )


def test_explicit_deny_and_catastrophic_guard_override_session_grant(tmp_path: Path) -> None:
    rules_path = tmp_path / "rules.yaml"
    rules_path.write_text(
        yaml.safe_dump([{"rule": "Bash(echo hello)", "effect": "deny"}]),
        encoding="utf-8",
    )
    checker = _checker(tmp_path, project_rules=rules_path)
    checker.bind_session("session-one")
    checker.add_session_allow("Bash", {"command": "echo hello"})
    checker.add_session_allow("Bash", {"command": "rm -rf /"})
    assert checker.check(Bash(), {"command": "echo hello"}).effect == "deny"
    assert checker.check(Bash(), {"command": "rm -rf /"}).effect == "deny"


def test_explicit_ask_rule_overrides_session_grant(tmp_path: Path) -> None:
    rules_path = tmp_path / "rules.yaml"
    rules_path.write_text(
        yaml.safe_dump([{"rule": "Bash(echo hello)", "effect": "ask"}]),
        encoding="utf-8",
    )
    checker = _checker(tmp_path, project_rules=rules_path)
    checker.bind_session("session-one")
    checker.add_session_allow("Bash", {"command": "echo hello"})
    assert checker.check(Bash(), {"command": "echo hello"}).effect == "ask"


def test_explicit_rule_overrides_safe_command_auto_allow(tmp_path: Path) -> None:
    rules_path = tmp_path / "rules.yaml"
    rules_path.write_text(
        yaml.safe_dump([{"rule": "Bash(git status)", "effect": "deny"}]),
        encoding="utf-8",
    )
    checker = _checker(tmp_path, project_rules=rules_path)
    assert checker.check(Bash(), {"command": "git status"}).effect == "deny"


def test_corrupt_state_fails_closed_and_store_is_protected(tmp_path: Path) -> None:
    store = SessionAllowStore(tmp_path, "session-one")
    store.path.parent.mkdir(parents=True)
    store.path.write_text("invalid", encoding="utf-8")
    checker = _checker(tmp_path)
    checker.bind_session("session-one")
    assert checker.check(Bash(), {"command": "echo hello"}).effect == "ask"
    assert checker.sandbox.check(str(store.path))[0] is False
    with pytest.raises(ValueError, match="session ID"):
        SessionAllowStore(tmp_path, "../escape")


def test_session_delete_removes_permission_sidecar(tmp_path: Path) -> None:
    manager = SessionManager(str(tmp_path))
    session = manager.create()
    store = SessionAllowStore(tmp_path, session.session_id)
    checker = _checker(tmp_path)
    checker.bind_session(session.session_id)
    checker.add_session_allow("Bash", {"command": "echo hello"})
    assert store.path.exists()
    session.close()
    assert manager.delete(session.session_id)
    assert not store.path.exists()


def test_app_session_switch_rebinds_checker(tmp_path: Path) -> None:
    checker = _checker(tmp_path)
    app = ValeCodeApp([])
    app.agent = SimpleNamespace(permission_checker=checker, session_id="")
    app._set_session(SimpleNamespace(session_id="session-one"))
    checker.add_session_allow("Bash", {"command": "echo hello"})
    app._set_session(SimpleNamespace(session_id="session-two"))
    assert checker.check(Bash(), {"command": "echo hello"}).effect == "ask"
    app._set_session(SimpleNamespace(session_id="session-one"))
    assert checker.check(Bash(), {"command": "echo hello"}).effect == "allow"


@pytest.mark.asyncio
@pytest.mark.parametrize("approval", [
    PermissionResponse.ALLOW_SESSION, PermissionResponse.ALLOW_ALWAYS,
])
async def test_session_choice_does_not_create_project_rule(
    tmp_path: Path, approval: PermissionResponse
) -> None:
    from tests.test_agent import MockLLMClient

    command = "echo hello"
    client = MockLLMClient([
        [ToolCallComplete("t1", "Bash", {"command": command}), StreamEnd("end_turn")],
        [ToolCallComplete("t2", "Bash", {"command": command}), StreamEnd("end_turn")],
        [TextDelta("done"), StreamEnd("end_turn")],
    ])
    checker = _checker(tmp_path)
    checker.bind_session("session-one")
    agent = Agent(
        client, create_default_registry(), "anthropic",
        work_dir=str(tmp_path), permission_checker=checker,
    )
    agent.session_id = "session-one"
    conversation = ConversationManager()
    conversation.add_user_message("run twice")
    requests = 0
    async for event in agent.run(conversation):
        if isinstance(event, PermissionRequest):
            requests += 1
            event.future.set_result(approval)
    assert requests == 1
    assert checker.check(Bash(), {"command": command}).effect == "allow"
    assert not (tmp_path / ".valecode" / "permissions.local.yaml").exists()
