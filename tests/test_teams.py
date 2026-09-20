
"""Agent Team（智能体团队）系统的测试（第 14 章）。"""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
import os
import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from valecode.teams.models import (
    AgentTeam,
    BackendType,
    TeammateInfo,
    resolve_team_dir,
    unique_team_name,
)
from valecode.teams.shared_task import (
    SharedTask,
    SharedTaskClaimError,
    SharedTaskStore,
)
from valecode.teams.mailbox import (
    Mailbox,
    MailboxDataError,
    MailboxLockTimeout,
    MailboxMessage,
    create_message,
)
from valecode.teams.registry import AgentNameRegistry
from valecode.teams.backend_detect import BackendDetectionError, detect_backend, detect_pane_backend
from valecode.teams.coordinator import (
    get_coordinator_system_prompt,
    get_coordinator_user_context,
    is_coordinator_mode,
    match_session_mode,
)
from valecode.tools.task_update import TaskUpdateParams, TaskUpdateTool
from valecode.tools.lead_tasks import (
    LeadTaskCreateParams,
    LeadTaskCreateTool,
    LeadTaskListParams,
    LeadTaskListTool,
    LeadTaskUpdateParams,
    LeadTaskUpdateTool,
)
from valecode.agents.tool_filter import (
    COORDINATOR_MODE_ALLOWED_TOOLS,
    IN_PROCESS_TEAMMATE_ALLOWED_TOOLS,
    TEAMMATE_COORDINATION_TOOLS,
    build_teammate_tools,
    apply_coordinator_filter,
)
from valecode.tools import ToolRegistry
from valecode.tools.base import Tool, ToolResult
from valecode.tools.send_message import SendMessageParams, SendMessageTool
from valecode.persistence import Database, TaskStore
from valecode.teams.manager import TeamManager

# =====================================================================
# 辅助工具
# =====================================================================

class DummyTool(Tool):
    params_model = MagicMock

    def __init__(self, name: str, category: str = "read"):
        self.name = name
        self.description = f"Dummy {name}"
        self.category = category
        self.is_concurrency_safe = True
        self.is_system_tool = False

    def get_schema(self):
        return {"name": self.name, "description": self.description, "input_schema": {}}

    async def execute(self, params):
        return ToolResult(output=f"{self.name} executed")

def make_registry(*tool_names: str) -> ToolRegistry:
    reg = ToolRegistry()
    for name in tool_names:
        reg.register(DummyTool(name))
    return reg

@pytest.fixture(autouse=True)
def _reset_registry():
    AgentNameRegistry.reset()
    yield
    AgentNameRegistry.reset()

@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)

# =====================================================================
# 1. AgentTeam / TeammateInfo
# =====================================================================

class TestModels:
    def test_teammate_info_roundtrip(self):
        info = TeammateInfo(
            name="alice",
            agent_id="abc123",
            agent_type="worker",
            model="sonnet",
            worktree_path="/tmp/wt",
            backend_type="tmux",
            is_active=True,
        )
        d = info.to_dict()
        restored = TeammateInfo.from_dict(d)
        assert restored.name == "alice"
        assert restored.agent_id == "abc123"
        assert restored.is_active is True

    def test_agent_team_save_load(self, tmp_dir):
        config_path = str(Path(tmp_dir) / "team" / "config.json")
        team = AgentTeam(
            name="test-team",
            lead_agent_id="lead-001",
            config_path=config_path,
            description="Test team",
        )
        team.add_member(TeammateInfo(
            name="alice", agent_id="a1", agent_type="worker",
            model="sonnet", worktree_path="/tmp/wt1", backend_type="tmux",
        ))
        team.save()

        loaded = AgentTeam.load(config_path)
        assert loaded.name == "test-team"
        assert loaded.lead_agent_id == "lead-001"
        assert len(loaded.members) == 1
        assert loaded.members[0].name == "alice"

    def test_get_member(self):
        team = AgentTeam(name="t", lead_agent_id="l")
        team.add_member(TeammateInfo(
            name="bob", agent_id="b1", agent_type="w",
            model="", worktree_path="", backend_type="in-process",
        ))
        assert team.get_member("bob") is not None
        assert team.get_member("b1") is not None
        assert team.get_member("nonexistent") is None

    def test_remove_member(self):
        team = AgentTeam(name="t", lead_agent_id="l")
        team.add_member(TeammateInfo(
            name="bob", agent_id="b1", agent_type="w",
            model="", worktree_path="", backend_type="in-process",
        ))
        assert team.remove_member("bob") is True
        assert len(team.members) == 0
        assert team.remove_member("bob") is False

    def test_set_member_active(self):
        team = AgentTeam(name="t", lead_agent_id="l")
        team.add_member(TeammateInfo(
            name="alice", agent_id="a1", agent_type="w",
            model="", worktree_path="", backend_type="in-process",
            is_active=True,
        ))
        team.set_member_active("alice", False)
        assert team.members[0].is_active is False
        assert team.all_idle() is True

    def test_all_idle(self):
        team = AgentTeam(name="t", lead_agent_id="l")
        team.add_member(TeammateInfo(
            name="alice", agent_id="a1", agent_type="w",
            model="", worktree_path="", backend_type="in-process",
            is_active=False,
        ))
        team.add_member(TeammateInfo(
            name="bob", agent_id="b1", agent_type="w",
            model="", worktree_path="", backend_type="in-process",
            is_active=True,
        ))
        assert team.all_idle() is False

    def test_unique_team_name(self, tmp_dir):
        with patch("valecode.teams.models.Path.home", return_value=Path(tmp_dir)):
            name1 = unique_team_name("my-team")
            assert name1 == "my-team"
            (Path(tmp_dir) / ".valecode" / "teams" / "my-team").mkdir(parents=True)
            name2 = unique_team_name("my-team")
            assert name2 == "my-team-2"

# =====================================================================
# 2. Durable Team state
# =====================================================================

def test_team_manager_persists_and_recovers_member_state(tmp_path):
    database = Database(tmp_path / "control.db")
    database.initialize()
    task_store = TaskStore(database)
    state_home = tmp_path / "state-home"
    with patch("valecode.teams.models.Path.home", return_value=state_home):
        manager = TeamManager(task_store=task_store)
        team = manager.create_team(
            "durable-team", "lead", teammate_mode="in-process"
        )
        member = TeammateInfo(
            name="worker",
            agent_id="agent-1",
            agent_type="general",
            model="test",
            worktree_path="",
            backend_type="in-process",
            is_active=True,
        )
        manager.register_member(team.name, member)
        manager.set_member_idle(team.name, member.name)
        Path(team.config_path).unlink()

        restored_manager = TeamManager(task_store=task_store)
        restored = restored_manager.get_team(team.name)
        assert restored is not None
        assert restored.members[0].is_active is False
        restored_manager.delete_team(team.name)
        assert restored_manager._team_store.get_team(team.name).status == "deleted"


# =====================================================================
# 3. SharedTaskStore
# =====================================================================

class TestSharedTaskStore:
    def test_create_and_get(self, tmp_dir):
        store = SharedTaskStore(Path(tmp_dir) / "tasks.json")
        store.init_empty()
        task = store.create(title="Do something", description="Details", assignee="alice")
        assert task.id == "1"
        assert task.title == "Do something"

        fetched = store.get("1")
        assert fetched is not None
        assert fetched.assignee == "alice"

    def test_auto_increment_id(self, tmp_dir):
        store = SharedTaskStore(Path(tmp_dir) / "tasks.json")
        store.init_empty()
        t1 = store.create(title="First")
        t2 = store.create(title="Second")
        assert t1.id == "1"
        assert t2.id == "2"

    def test_list_with_filters(self, tmp_dir):
        store = SharedTaskStore(Path(tmp_dir) / "tasks.json")
        store.init_empty()
        store.create(title="A", assignee="alice")
        t2 = store.create(title="B", assignee="bob")
        store.update(t2.id, status="in_progress")

        all_tasks = store.list_tasks()
        assert len(all_tasks) == 2

        pending = store.list_tasks(status="pending")
        assert len(pending) == 1
        assert pending[0].title == "A"

        bobs = store.list_tasks(assignee="bob")
        assert len(bobs) == 1

    def test_update_with_dependencies(self, tmp_dir):
        store = SharedTaskStore(Path(tmp_dir) / "tasks.json")
        store.init_empty()
        store.create(title="Task A")
        store.create(title="Task B")

        updated = store.update("2", add_blocked_by=["1"])
        assert updated is not None
        assert "1" in updated.blocked_by

        updated = store.update("1", add_blocks=["2"])
        assert "2" in updated.blocks

    def test_update_nonexistent_returns_none(self, tmp_dir):
        store = SharedTaskStore(Path(tmp_dir) / "tasks.json")
        store.init_empty()
        assert store.update("999") is None

    def test_persistence(self, tmp_dir):
        path = Path(tmp_dir) / "tasks.json"
        store1 = SharedTaskStore(path)
        store1.init_empty()
        store1.create(title="Persisted task")

        store2 = SharedTaskStore(path)
        tasks = store2.list_tasks()
        assert len(tasks) == 1
        assert tasks[0].title == "Persisted task"

    def test_claim_requires_completed_dependencies(self, tmp_dir):
        store = SharedTaskStore(Path(tmp_dir) / "tasks.json")
        store.init_empty()
        prerequisite = store.create(title="Design")
        implementation = store.create(
            title="Implement", blocked_by=[prerequisite.id]
        )

        with pytest.raises(SharedTaskClaimError, match="blocked by incomplete"):
            store.claim(implementation.id, "alice")

        store.update(prerequisite.id, status="completed")
        claimed = store.claim(implementation.id, "alice")
        assert claimed.status == "in_progress"
        assert claimed.assignee == "alice"
        assert store.claim(implementation.id, "alice").assignee == "alice"

    def test_concurrent_claim_has_exactly_one_owner(self, tmp_dir):
        path = Path(tmp_dir) / "tasks.json"
        store = SharedTaskStore(path)
        store.init_empty()
        task = store.create(title="Only once")

        def attempt(index: int) -> str | None:
            try:
                return SharedTaskStore(path).claim(task.id, f"worker-{index}").assignee
            except SharedTaskClaimError:
                return None

        with ThreadPoolExecutor(max_workers=12) as pool:
            winners = [result for result in pool.map(attempt, range(12)) if result]

        assert len(winners) == 1
        persisted = SharedTaskStore(path).get(task.id)
        assert persisted is not None
        assert persisted.assignee == winners[0]

    def test_dependency_relations_are_bidirectional_and_acyclic(self, tmp_dir):
        store = SharedTaskStore(Path(tmp_dir) / "tasks.json")
        store.init_empty()
        first = store.create(title="First")
        second = store.create(title="Second", blocked_by=[first.id])

        assert store.get(first.id).blocks == [second.id]
        with pytest.raises(ValueError, match="cannot contain a cycle"):
            store.update(first.id, add_blocked_by=[second.id])
        with pytest.raises(ValueError, match="Unknown shared task dependencies"):
            store.create(title="Broken", blocked_by=["999"])
        assert [task.id for task in store.list_tasks()] == [first.id, second.id]

    def test_concurrent_create_preserves_every_task(self, tmp_dir):
        path = Path(tmp_dir) / "tasks.json"
        SharedTaskStore(path).init_empty()

        def create(index: int) -> str:
            return SharedTaskStore(path).create(title=f"Task {index}").id

        with ThreadPoolExecutor(max_workers=12) as pool:
            ids = list(pool.map(create, range(20)))

        assert len(set(ids)) == 20
        assert len(SharedTaskStore(path).list_tasks()) == 20

    @pytest.mark.asyncio
    async def test_task_update_claims_only_for_current_teammate(self, tmp_dir):
        store = SharedTaskStore(Path(tmp_dir) / "tasks.json")
        store.init_empty()
        task = store.create(title="Claim me")
        manager = MagicMock()
        manager.get_task_store.return_value = store
        tool = TaskUpdateTool(manager, "alpha", "alice")

        denied = await tool.execute(
            TaskUpdateParams(
                task_id=task.id, status="in_progress", assignee="bob"
            )
        )
        assert denied.is_error is True
        assert "only claim a task for itself" in denied.output

        claimed = await tool.execute(
            TaskUpdateParams(task_id=task.id, status="in_progress")
        )
        assert claimed.is_error is False
        assert store.get(task.id).assignee == "alice"

    @pytest.mark.asyncio
    async def test_lead_can_manage_but_not_claim_team_tasks(self, tmp_dir):
        store = SharedTaskStore(Path(tmp_dir) / "tasks.json")
        store.init_empty()
        manager = MagicMock()
        manager.get_team.return_value = AgentTeam(
            name="alpha", lead_agent_id="lead-1"
        )
        manager.get_task_store.return_value = store

        created = await LeadTaskCreateTool(manager, "lead-1").execute(
            LeadTaskCreateParams(team_name="alpha", title="Implement")
        )
        assert created.is_error is False

        assigned = await LeadTaskUpdateTool(manager, "lead-1").execute(
            LeadTaskUpdateParams(
                team_name="alpha", task_id="1", assignee="worker"
            )
        )
        assert assigned.is_error is False
        assert store.get("1").assignee == "worker"

        listed = await LeadTaskListTool(manager, "lead-1").execute(
            LeadTaskListParams(team_name="alpha")
        )
        assert "Implement" in listed.output

        denied = await LeadTaskUpdateTool(manager, "lead-1").execute(
            LeadTaskUpdateParams(
                team_name="alpha", task_id="1", status="in_progress"
            )
        )
        assert denied.is_error is True
        assert "only the teammate" in denied.output

    @pytest.mark.asyncio
    async def test_lead_task_tools_reject_other_teams(self, tmp_dir):
        store = SharedTaskStore(Path(tmp_dir) / "tasks.json")
        store.init_empty()
        manager = MagicMock()
        manager.get_team.return_value = AgentTeam(
            name="alpha", lead_agent_id="someone-else"
        )
        manager.get_task_store.return_value = store

        result = await LeadTaskListTool(manager, "lead-1").execute(
            LeadTaskListParams(team_name="alpha")
        )
        assert result.is_error is True
        assert "not the lead" in result.output

# =====================================================================
# 3. Mailbox
# =====================================================================

class TestMailbox:
    def test_write_and_consume(self, tmp_dir):
        mailbox = Mailbox(tmp_dir)
        msg = create_message("alice", "bob", "Hello bob", summary="greeting")
        mailbox.write("bob-agent-id", msg)

        messages = mailbox.consume("bob-agent-id")
        assert len(messages) == 1
        assert messages[0].content == "Hello bob"
        assert messages[0].from_agent == "alice"

        # 已被消费 —— 此时应该为空
        messages2 = mailbox.consume("bob-agent-id")
        assert len(messages2) == 0

    def test_read_without_consume(self, tmp_dir):
        mailbox = Mailbox(tmp_dir)
        msg = create_message("alice", "bob", "Peek")
        mailbox.write("bob-id", msg)

        messages = mailbox.read("bob-id")
        assert len(messages) == 1

        # 仍然存在
        messages2 = mailbox.read("bob-id")
        assert len(messages2) == 1

    def test_broadcast(self, tmp_dir):
        mailbox = Mailbox(tmp_dir)
        msg = create_message("alice", "*", "Team update", summary="update")
        mailbox.broadcast(["bob-id", "charlie-id", "alice-id"], msg, exclude="alice-id")

        bob_msgs = mailbox.consume("bob-id")
        charlie_msgs = mailbox.consume("charlie-id")
        alice_msgs = mailbox.consume("alice-id")

        assert len(bob_msgs) == 1
        assert len(charlie_msgs) == 1
        assert len(alice_msgs) == 0

    def test_cleanup(self, tmp_dir):
        mailbox = Mailbox(tmp_dir)
        msg = create_message("a", "b", "test")
        mailbox.write("agent-1", msg)
        mailbox.cleanup("agent-1")
        assert len(mailbox.read("agent-1")) == 0

    def test_empty_mailbox(self, tmp_dir):
        mailbox = Mailbox(tmp_dir)
        assert mailbox.consume("nonexistent") == []
        assert mailbox.read("nonexistent") == []

    def test_lock_timeout_fails_closed_without_deleting_owner_lock(
        self, tmp_dir, monkeypatch
    ):
        mailbox = Mailbox(tmp_dir)
        mailbox.write("agent-1", create_message("lead", "agent-1", "existing"))
        lock_path = mailbox._lock_path("agent-1")
        lock_path.write_text("other-owner", encoding="ascii")
        monkeypatch.setattr("valecode.teams.mailbox.time.sleep", lambda _delay: None)

        with pytest.raises(MailboxLockTimeout):
            mailbox.write(
                "agent-1", create_message("lead", "agent-1", "must not append")
            )

        assert lock_path.read_text(encoding="ascii") == "other-owner"
        lock_path.unlink()
        assert [message.content for message in mailbox.read("agent-1")] == [
            "existing"
        ]

    def test_read_holds_the_same_lock_as_writers(self, tmp_dir, monkeypatch):
        mailbox = Mailbox(tmp_dir)
        mailbox.write("agent-1", create_message("lead", "agent-1", "message"))
        original = mailbox._read_inbox

        def checked(agent_id):
            assert mailbox._lock_path(agent_id).exists()
            return original(agent_id)

        monkeypatch.setattr(mailbox, "_read_inbox", checked)
        assert mailbox.read("agent-1")[0].content == "message"

    def test_corrupt_inbox_is_not_silently_overwritten(self, tmp_dir):
        mailbox = Mailbox(tmp_dir)
        inbox = mailbox._inbox_path("agent-1")
        inbox.write_text("{not-json", encoding="utf-8")

        with pytest.raises(MailboxDataError):
            mailbox.write("agent-1", create_message("lead", "agent-1", "new"))

        assert inbox.read_text(encoding="utf-8") == "{not-json"
        assert mailbox._lock_path("agent-1").exists() is False

    def test_concurrent_writes_do_not_lose_messages(self, tmp_dir):
        mailbox = Mailbox(tmp_dir)

        def send(index: int) -> None:
            mailbox.write(
                "agent-1",
                create_message("lead", "agent-1", f"message-{index}"),
            )

        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(send, range(20)))

        assert {message.content for message in mailbox.read("agent-1")} == {
            f"message-{index}" for index in range(20)
        }

    @pytest.mark.parametrize("agent_id", ["../escape", "a/b", "a\\b", ""])
    def test_agent_id_cannot_escape_mailbox_directory(self, tmp_dir, agent_id):
        mailbox = Mailbox(tmp_dir)

        with pytest.raises(ValueError, match="agent ID"):
            mailbox.read(agent_id)


class TestSendMessageRouting:
    @staticmethod
    def _setup(tmp_dir):
        team = AgentTeam(name="alpha", lead_agent_id="lead-1")
        team.add_member(TeammateInfo(
            name="alice",
            agent_id="agent-a",
            agent_type="worker",
            model="test",
            worktree_path="",
            backend_type="in-process",
        ))
        team.add_member(TeammateInfo(
            name="bob",
            agent_id="agent-b",
            agent_type="worker",
            model="test",
            worktree_path="",
            backend_type="in-process",
        ))
        mailbox = Mailbox(tmp_dir)
        manager = MagicMock()
        manager.get_team.return_value = team
        manager.get_mailbox.return_value = mailbox
        manager.get_pane_id.return_value = None
        registry = AgentNameRegistry.instance()
        registry.register("alice", "agent-a")
        registry.register("bob", "agent-b")
        return team, mailbox, manager

    @pytest.mark.asyncio
    async def test_lead_alias_routes_to_real_lead_mailbox(self, tmp_dir):
        _team, mailbox, manager = self._setup(tmp_dir)
        tool = SendMessageTool(manager, "alpha", "agent-a", "alice")

        result = await tool.execute(SendMessageParams(
            to="lead",
            message="Work is complete",
            summary="work completed",
        ))

        assert result.is_error is False
        messages = mailbox.consume("lead-1")
        assert [message.content for message in messages] == ["Work is complete"]

    @pytest.mark.asyncio
    async def test_global_registry_cannot_route_to_another_team(self, tmp_dir):
        _team, mailbox, manager = self._setup(tmp_dir)
        AgentNameRegistry.instance().register("outsider", "agent-other")
        tool = SendMessageTool(manager, "alpha", "agent-a", "alice")

        result = await tool.execute(SendMessageParams(
            to="outsider",
            message="secret",
            summary="cross team message",
        ))

        assert result.is_error is True
        assert "not a member" in result.output
        assert mailbox.read("agent-other") == []

    @pytest.mark.asyncio
    async def test_team_local_name_wins_over_global_registry_collision(self, tmp_dir):
        _team, mailbox, manager = self._setup(tmp_dir)
        AgentNameRegistry.instance().register("bob", "agent-other")
        tool = SendMessageTool(manager, "alpha", "agent-a", "alice")

        result = await tool.execute(SendMessageParams(
            to="bob",
            message="for local bob",
            summary="local teammate message",
        ))

        assert result.is_error is False
        assert mailbox.consume("agent-b")[0].content == "for local bob"
        assert mailbox.read("agent-other") == []

    @pytest.mark.asyncio
    async def test_stale_sender_is_rejected(self, tmp_dir):
        _team, mailbox, manager = self._setup(tmp_dir)
        tool = SendMessageTool(manager, "alpha", "agent-old", "old")

        result = await tool.execute(SendMessageParams(
            to="bob",
            message="secret",
            summary="stale sender message",
        ))

        assert result.is_error is True
        assert "no longer a member" in result.output
        assert mailbox.read("agent-b") == []

# =====================================================================
# 4. AgentNameRegistry
# =====================================================================

class TestAgentNameRegistry:

    def test_register_and_resolve(self):
        reg = AgentNameRegistry.instance()
        reg.register("alice", "agent-abc")
        assert reg.resolve("alice") == "agent-abc"
        assert reg.resolve("agent-abc") == "agent-abc"  # 直接按 ID 查找
        assert reg.resolve("nonexistent") is None

    def test_unregister(self):
        reg = AgentNameRegistry.instance()
        reg.register("bob", "agent-xyz")
        reg.unregister("bob")
        assert reg.resolve("bob") is None

    def test_list_all(self):
        reg = AgentNameRegistry.instance()
        reg.register("alice", "a1")
        reg.register("bob", "b1")
        all_names = reg.list_all()
        assert all_names == {"alice": "a1", "bob": "b1"}

    def test_singleton(self):
        r1 = AgentNameRegistry.instance()
        r2 = AgentNameRegistry.instance()
        assert r1 is r2

# =====================================================================
# 5. Backend Detection（后端探测）
# =====================================================================

class TestBackendDetect:
    def test_in_process_mode(self):
        result = detect_backend(teammate_mode="in-process")
        assert result == BackendType.IN_PROCESS

    def test_non_interactive(self):
        result = detect_backend(is_interactive=False)
        assert result == BackendType.IN_PROCESS

    def test_detect_backend_always_in_process(self):
        # detect_backend 统一返回 IN_PROCESS，pane 检测由 detect_pane_backend 负责
        with patch.dict(os.environ, {"TMUX": "/tmp/tmux-1234/default,12345,0"}):
            result = detect_backend()
            assert result == BackendType.IN_PROCESS

    def test_pane_tmux_session(self):
        with patch.dict(os.environ, {"TMUX": "/tmp/tmux-1234/default,12345,0"}):
            result = detect_pane_backend()
            assert result == BackendType.TMUX

    def test_pane_iterm2_with_it2(self):
        env = {"TERM_PROGRAM": "iTerm.app"}
        with patch.dict(os.environ, env, clear=False):
            with patch("valecode.teams.backend_detect.shutil.which") as mock_which:
                def which_side_effect(cmd):
                    if cmd == "it2":
                        return "/usr/local/bin/it2"
                    if cmd == "tmux":
                        return None
                    return None
                mock_which.side_effect = which_side_effect
                with patch.dict(os.environ, {"TMUX": ""}, clear=False):
                    os.environ.pop("TMUX", None)
                    result = detect_pane_backend()
                    assert result == BackendType.ITERM2

    def test_pane_tmux_installed_not_in_session(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TMUX", None)
            os.environ.pop("TERM_PROGRAM", None)
            with patch("valecode.teams.backend_detect.shutil.which") as mock_which:
                mock_which.return_value = "/usr/bin/tmux"
                result = detect_pane_backend()
                assert result == BackendType.TMUX

    def test_pane_no_backend_falls_back_to_in_process(self):
        # 没有外部终端时，detect_pane_backend 回退到 IN_PROCESS 而非抛异常
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TMUX", None)
            os.environ.pop("TERM_PROGRAM", None)
            with patch("valecode.teams.backend_detect.shutil.which", return_value=None):
                result = detect_pane_backend()
                assert result == BackendType.IN_PROCESS

# =====================================================================
# 6. Tool Filtering（工具过滤）
# =====================================================================

class TestToolFilter:
    def test_teammate_coordination_tools_in_allowed(self):
        for tool_name in TEAMMATE_COORDINATION_TOOLS:
            assert tool_name in IN_PROCESS_TEAMMATE_ALLOWED_TOOLS

    def test_coordinator_mode_tools(self):
        assert "Agent" in COORDINATOR_MODE_ALLOWED_TOOLS
        assert "SendMessage" in COORDINATOR_MODE_ALLOWED_TOOLS
        assert "TaskStop" in COORDINATOR_MODE_ALLOWED_TOOLS
        assert "SyntheticOutput" in COORDINATOR_MODE_ALLOWED_TOOLS
        assert "ReadFile" in COORDINATOR_MODE_ALLOWED_TOOLS
        assert "Bash" in COORDINATOR_MODE_ALLOWED_TOOLS
        assert "Glob" in COORDINATOR_MODE_ALLOWED_TOOLS
        assert "Grep" in COORDINATOR_MODE_ALLOWED_TOOLS
        assert "WriteFile" not in COORDINATOR_MODE_ALLOWED_TOOLS
        assert "EditFile" not in COORDINATOR_MODE_ALLOWED_TOOLS

    def test_apply_coordinator_filter(self):
        reg = make_registry(
            "Agent", "ReadFile", "WriteFile", "Bash", "SendMessage",
            "TaskStop", "SyntheticOutput", "TeamCreate", "TeamDelete",
        )
        filtered = apply_coordinator_filter(reg)
        names = {t.name for t in filtered.list_tools()}
        assert "Agent" in names
        assert "SendMessage" in names
        assert "SyntheticOutput" in names
        assert "ReadFile" in names
        assert "Bash" in names
        assert "WriteFile" not in names

# =====================================================================
# 7. Coordinator Mode（协调者模式）
# =====================================================================

class TestCoordinatorMode:
    def test_disabled_by_default(self):
        assert is_coordinator_mode(enable_flag=False) is False

    def test_enabled_with_flag(self):
        assert is_coordinator_mode(enable_flag=True) is True

    def test_system_prompt_contains_phases(self):
        prompt = get_coordinator_system_prompt()
        assert "Research" in prompt
        assert "Synthesis" in prompt
        assert "Implementation" in prompt
        assert "Verification" in prompt

    def test_system_prompt_anti_pattern(self):
        prompt = get_coordinator_system_prompt()
        assert "based on your findings" in prompt.lower()
        assert "Anti-pattern" in prompt or "BAD" in prompt

    def test_system_prompt_continue_vs_spawn(self):
        prompt = get_coordinator_system_prompt()
        assert "Continue" in prompt
        assert "Spawn fresh" in prompt

    def test_system_prompt_task_notification(self):
        prompt = get_coordinator_system_prompt()
        assert "<task-notification>" in prompt
        assert "<task-id>" in prompt

    def test_match_session_mode_no_switch(self):
        result = match_session_mode("coordinator", enable_flag=True)
        assert result is None

    def test_match_session_mode_switch(self):
        result = match_session_mode("coordinator", enable_flag=False)
        assert result is not None
        assert "Entered" in result

    def test_match_session_mode_none(self):
        result = match_session_mode(None)
        assert result is None

    def test_coordinator_user_context(self):
        ctx = get_coordinator_user_context()
        assert "workerToolsContext" in ctx
        assert "Workers" in ctx["workerToolsContext"]

# =====================================================================
# 8. Config Extensions（配置项扩展）
# =====================================================================

class TestConfigExtensions:
    def test_teammate_mode_defaults(self):
        from valecode.config import AppConfig
        cfg = AppConfig(providers=[])
        assert cfg.teammate_mode == ""
        assert cfg.enable_coordinator_mode is False

    def test_load_config_with_team_fields(self, tmp_dir):
        from valecode.config import load_config
        config_path = Path(tmp_dir) / "config.yaml"
        config_path.write_text(
            "providers:\n"
            "  - name: test\n"
            "    protocol: anthropic\n"
            "    base_url: http://localhost\n"
            "    model: test-model\n"
            "teammate_mode: 'in-process'\n"
            "enable_coordinator_mode: true\n"
        )
        cfg = load_config(config_path)
        assert cfg.teammate_mode == "in-process"
        assert cfg.enable_coordinator_mode is True

    def test_invalid_teammate_mode(self, tmp_dir):
        from valecode.config import ConfigError, load_config
        config_path = Path(tmp_dir) / "config.yaml"
        config_path.write_text(
            "providers:\n"
            "  - name: test\n"
            "    protocol: anthropic\n"
            "    base_url: http://localhost\n"
            "    model: test-model\n"
            "teammate_mode: 'invalid'\n"
        )
        with pytest.raises(ConfigError):
            load_config(config_path)

# =====================================================================
# 9. Transcript Persistence（会话记录持久化）
# =====================================================================

class TestTranscript:

    def test_save_and_load(self, tmp_dir):
        from valecode.conversation import ConversationManager
        from valecode.teams.transcript import load_transcript, save_transcript

        conv = ConversationManager()
        conv.add_user_message("Hello agent")
        conv.add_assistant_message("Hello user")

        with patch("valecode.teams.models.Path.home", return_value=Path(tmp_dir)):
            save_transcript("test-team", "agent-001", conv)
            restored = load_transcript("test-team", "agent-001")

        assert restored is not None
        assert len(restored.history) == 2
        assert restored.history[0].role == "user"
        assert restored.history[0].content == "Hello agent"
        assert restored.history[1].role == "assistant"

    def test_load_nonexistent(self, tmp_dir):
        from valecode.teams.transcript import load_transcript
        with patch("valecode.teams.models.Path.home", return_value=Path(tmp_dir)):
            result = load_transcript("no-team", "no-agent")
        assert result is None

# =====================================================================
# 10. Agent build_system_prompt 集成测试
# =====================================================================

class TestAgentCoordinatorIntegration:
    def test_normal_prompt(self):
        from valecode.prompts import build_system_prompt, IDENTITY_SECTION
        prompt = build_system_prompt()
        # 验证 identity section 内容包含在 prompt 中
        assert "ValeCode" in prompt
        assert IDENTITY_SECTION.content[:30] in prompt

    def test_coordinator_prompt(self):
        from valecode.prompts import build_system_prompt
        prompt = build_system_prompt(coordinator_mode=True)
        assert "coordinator" in prompt.lower()

    def test_coordinator_mode_overrides_normal(self):
        from valecode.prompts import build_system_prompt
        # coordinator 模式走独立的 prompt 生成路径，不包含普通 identity 段
        prompt = build_system_prompt(coordinator_mode=True)
        assert "coordinator" in prompt.lower()
