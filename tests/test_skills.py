
"""Skill 系统的测试 —— 包括 parser、loader、executor 以及 LoadSkill 工具。"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from valecode.skills.parser import (
    SkillDef,
    SkillParseError,
    parse_frontmatter,
    parse_skill_file,
    substitute_arguments,
)
from valecode.skills.loader import SkillLoader
from valecode.skills.executor import SkillExecutor
from valecode.tools import ToolRegistry

# ---------------------------------------------------------------------------
# 辅助工具
# ---------------------------------------------------------------------------

VALID_SKILL_MD = textwrap.dedent("""\
    ---
    name: test-skill
    description: A test skill
    mode: inline
    ---

    # Task

    Do something.

    $ARGUMENTS
""")

FORK_SKILL_MD = textwrap.dedent("""\
    ---
    name: review
    description: Review code
    mode: fork
    context: none
    ---

    Review the code changes.

    $ARGUMENTS
""")


# ---------------------------------------------------------------------------
# Parser 测试
# ---------------------------------------------------------------------------

class TestParseFrontmatter:
    def test_valid(self) -> None:
        meta, body = parse_frontmatter(VALID_SKILL_MD)
        assert meta["name"] == "test-skill"
        assert meta["description"] == "A test skill"
        assert "Do something" in body

    def test_missing_opening(self) -> None:
        with pytest.raises(SkillParseError, match="Missing YAML frontmatter"):
            parse_frontmatter("no frontmatter here")

    def test_unclosed(self) -> None:
        with pytest.raises(SkillParseError, match="Unclosed YAML"):
            parse_frontmatter("---\nname: foo\n")

    def test_invalid_yaml(self) -> None:
        with pytest.raises(SkillParseError, match="Invalid YAML"):
            parse_frontmatter("---\n: :\n  bad: [yaml\n---\nbody")

    def test_non_dict_frontmatter(self) -> None:
        with pytest.raises(SkillParseError, match="must be a YAML mapping"):
            parse_frontmatter("---\n- list\n- item\n---\nbody")

class TestParseSkillFile:
    def test_valid_file(self, tmp_path: Path) -> None:
        f = tmp_path / "test.md"
        f.write_text(VALID_SKILL_MD)
        skill = parse_skill_file(f)
        assert skill.name == "test-skill"
        assert skill.description == "A test skill"
        assert skill.mode == "inline"
        assert "$ARGUMENTS" in skill.prompt_body

    def test_missing_name(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.md"
        f.write_text("---\ndescription: oops\n---\nbody")
        with pytest.raises(SkillParseError, match="Missing required field 'name'"):
            parse_skill_file(f)

    def test_missing_description(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.md"
        f.write_text("---\nname: foo\n---\nbody")
        with pytest.raises(SkillParseError, match="Missing required field 'description'"):
            parse_skill_file(f)

    def test_invalid_name_format(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.md"
        f.write_text("---\nname: UPPER\ndescription: x\n---\nbody")
        with pytest.raises(SkillParseError, match="Invalid skill name"):
            parse_skill_file(f)

    def test_invalid_mode(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.md"
        f.write_text("---\nname: foo\ndescription: x\nmode: bad\n---\nbody")
        with pytest.raises(SkillParseError, match="Invalid mode"):
            parse_skill_file(f)

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        with pytest.raises(SkillParseError, match="Cannot read"):
            parse_skill_file(tmp_path / "nope.md")

    def test_fork_mode_with_context(self, tmp_path: Path) -> None:
        f = tmp_path / "fork.md"
        f.write_text(FORK_SKILL_MD)
        skill = parse_skill_file(f)
        assert skill.mode == "fork"
        assert skill.context == "none"

    def test_model_must_be_a_nonempty_string(self, tmp_path: Path) -> None:
        f = tmp_path / "bad-model.md"
        f.write_text(
            "---\nname: bad-model\ndescription: bad\nmodel: ''\n---\nbody"
        )

        with pytest.raises(SkillParseError, match="Invalid model"):
            parse_skill_file(f)

    def test_scoped_permission_rules(self, tmp_path: Path) -> None:
        f = tmp_path / "guarded.md"
        f.write_text(textwrap.dedent("""\
            ---
            name: guarded
            description: Scoped access
            permissions:
              allow:
                - WriteFile(src/**)
              deny:
                - Bash(rm *)
            allowed-tools:
              - ReadFile
            ---
            Follow the scoped policy.
        """))

        skill = parse_skill_file(f)
        assert skill.permission_rules == {
            "allow": ["WriteFile(src/**)", "ReadFile(*)"],
            "deny": ["Bash(rm *)"],
        }

    def test_invalid_scoped_permission_rule(self, tmp_path: Path) -> None:
        f = tmp_path / "bad-permission.md"
        f.write_text(textwrap.dedent("""\
            ---
            name: guarded
            description: Scoped access
            permissions:
              allow:
                - not-a-rule
            ---
            Body
        """))

        with pytest.raises(SkillParseError, match="expected ToolName"):
            parse_skill_file(f)

class TestSubstituteArguments:
    def test_with_args(self) -> None:
        result = substitute_arguments("Do $ARGUMENTS now", "something cool")
        assert result == "Do something cool now"

    def test_without_args(self) -> None:
        result = substitute_arguments("Do $ARGUMENTS now", "")
        assert result == "Do  now"

    def test_no_placeholder(self) -> None:
        # 无占位符但 args 非空时，append fallback 将用户请求追加到末尾
        result = substitute_arguments("No placeholder here", "args")
        assert result == "No placeholder here\n\n## User Request\n\nargs"

    def test_multiple_placeholders(self) -> None:
        result = substitute_arguments("$ARGUMENTS and $ARGUMENTS", "x")
        assert result == "x and x"

# ---------------------------------------------------------------------------
# Loader 测试
# ---------------------------------------------------------------------------

class TestSkillLoader:
    def test_load_builtins_empty(self) -> None:
        """内置 skill 已移除，load_all 不再返回内置 skill。"""
        loader = SkillLoader("/nonexistent")
        skills = loader.load_all()
        assert "commit" not in skills
        assert "review" not in skills
        assert "test" not in skills
        assert "backend-interview" not in skills

    def test_project_overrides_builtin(self, tmp_path: Path) -> None:
        skills_dir = tmp_path / ".valecode" / "skills"
        skills_dir.mkdir(parents=True)
        custom = skills_dir / "commit.md"
        custom.write_text(textwrap.dedent("""\
            ---
            name: commit
            description: Custom commit
            mode: inline
            ---
            Custom prompt
        """))
        loader = SkillLoader(str(tmp_path))
        skills = loader.load_all()
        assert skills["commit"].description == "Custom commit"
        assert "Custom prompt" in skills["commit"].prompt_body

    def test_catalog_empty_without_project_skills(self) -> None:
        """无项目/用户 skill 且内置已移除时，catalog 为空。"""
        loader = SkillLoader("/nonexistent")
        loader.load_all()
        catalog = loader.get_catalog()
        assert catalog == []

    def test_get_returns_none_for_removed_builtins(self) -> None:
        """内置 skill 已移除，get 返回 None。"""
        loader = SkillLoader("/nonexistent")
        loader.load_all()
        assert loader.get("commit") is None
        assert loader.get("review") is None
        assert loader.get("test") is None

    def test_get_unknown(self) -> None:
        loader = SkillLoader("/nonexistent")
        loader.load_all()
        assert loader.get("nonexistent") is None

    def test_hot_reload(self, tmp_path: Path) -> None:
        skills_dir = tmp_path / ".valecode" / "skills"
        skills_dir.mkdir(parents=True)
        f = skills_dir / "custom.md"
        f.write_text(textwrap.dedent("""\
            ---
            name: custom
            description: v1
            ---
            Prompt v1
        """))
        loader = SkillLoader(str(tmp_path))
        loader.load_all()
        assert loader.get("custom").description == "v1"

        f.write_text(textwrap.dedent("""\
            ---
            name: custom
            description: v2
            ---
            Prompt v2
        """))
        skill = loader.get("custom")
        assert skill.description == "v2"
        assert "v2" in skill.prompt_body

    def test_hot_reload_fallback_on_error(self, tmp_path: Path) -> None:
        skills_dir = tmp_path / ".valecode" / "skills"
        skills_dir.mkdir(parents=True)
        f = skills_dir / "custom.md"
        f.write_text(textwrap.dedent("""\
            ---
            name: custom
            description: good
            ---
            Good prompt
        """))
        loader = SkillLoader(str(tmp_path))
        loader.load_all()

        f.write_text("broken content no frontmatter")
        skill = loader.get("custom")
        assert skill is not None
        assert skill.description == "good"

    def test_directory_skill_detected(self, tmp_path: Path) -> None:
        skills_dir = tmp_path / ".valecode" / "skills"
        skill_dir = skills_dir / "my-skill"
        skill_dir.mkdir(parents=True)
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text(textwrap.dedent("""\
            ---
            name: my-skill
            description: A directory skill
            ---
            SOP here
        """))
        loader = SkillLoader(str(tmp_path))
        skills = loader.load_all()
        assert "my-skill" in skills
        assert skills["my-skill"].is_directory is True

    def test_source_label(self, tmp_path: Path) -> None:
        loader = SkillLoader(str(tmp_path))
        loader.load_all()
        assert loader.get_source_label("nonexistent") == "unknown"

    def test_malformed_file_skipped(self, tmp_path: Path) -> None:
        skills_dir = tmp_path / ".valecode" / "skills"
        skills_dir.mkdir(parents=True)
        bad = skills_dir / "broken.md"
        bad.write_text("not valid frontmatter")
        good = skills_dir / "valid.md"
        good.write_text(textwrap.dedent("""\
            ---
            name: valid
            description: Works fine
            ---
            Prompt
        """))
        loader = SkillLoader(str(tmp_path))
        skills = loader.load_all()
        assert "valid" in skills
        assert "broken" not in skills

    def test_reload(self, tmp_path: Path) -> None:
        """reload 重新扫描目录，内置已移除时结果为空。"""
        loader = SkillLoader(str(tmp_path))
        loader.load_all()
        skills = loader.reload()
        assert isinstance(skills, dict)


# ---------------------------------------------------------------------------
# LoadSkill 工具
# ---------------------------------------------------------------------------

class TestLoadSkillTool:
    @pytest.mark.asyncio
    async def test_load_existing_skill(self) -> None:
        from valecode.tools.load_skill import LoadSkill, LoadSkillParams

        tool = LoadSkill()
        loader = MagicMock()
        agent = MagicMock()
        agent.registry = ToolRegistry()

        skill = SkillDef(
            name="commit",
            description="Commit",
            prompt_body="Do commit",
            is_directory=False,
        )
        loader.get.return_value = skill

        tool.set_loader(loader)
        tool.set_agent(agent)

        result = await tool.execute(LoadSkillParams(name="commit"))
        assert not result.is_error
        assert "# Skill: commit" in result.output and "Do commit" in result.output
        agent.activate_skill.assert_called_once_with("commit", "Do commit", {})

    @pytest.mark.asyncio
    async def test_load_unknown_skill(self) -> None:
        from valecode.tools.load_skill import LoadSkill, LoadSkillParams

        tool = LoadSkill()
        loader = MagicMock()
        loader.get.return_value = None
        loader.get_catalog.return_value = [("commit", "x")]
        agent = MagicMock()

        tool.set_loader(loader)
        tool.set_agent(agent)

        result = await tool.execute(LoadSkillParams(name="nonexistent"))
        assert result.is_error
        assert "unknown skill" in result.output

    @pytest.mark.asyncio
    async def test_not_initialized(self) -> None:
        from valecode.tools.load_skill import LoadSkill, LoadSkillParams

        tool = LoadSkill()
        result = await tool.execute(LoadSkillParams(name="test"))
        assert result.is_error
        assert "not properly initialized" in result.output

    def test_is_system_tool(self) -> None:
        from valecode.tools.load_skill import LoadSkill

        tool = LoadSkill()
        assert tool.is_system_tool is True
        assert tool.category == "read"


class TestSkillExecutor:
    def test_inline_skill_is_injected_into_main_conversation(self) -> None:
        from valecode.conversation import ConversationManager

        agent = MagicMock()
        agent.recovery_state = None
        conversation = ConversationManager()
        skill = SkillDef(
            name="review",
            description="Review code",
            prompt_body="Review $ARGUMENTS carefully",
        )
        executor = SkillExecutor(agent, MagicMock(), "anthropic")

        prompt = executor.execute_inline(skill, "src/app.py", conversation)

        assert prompt == "Review src/app.py carefully"
        agent.activate_skill.assert_called_once_with(
            "review", "Review src/app.py carefully", {}
        )
        assert len(conversation.history) == 1
        assert "# Skill: review" in conversation.history[0].content
        assert "Review src/app.py carefully" in conversation.history[0].content

    def test_fork_skill_model_uses_current_provider_connection(self) -> None:
        from valecode.config import ProviderConfig

        agent = MagicMock()
        agent.context_window = 100
        agent.provider_name = "default"
        agent.model = "parent-model"
        switched_client = MagicMock()
        provider = ProviderConfig(
            name="default",
            protocol="openai-compat",
            base_url="https://example.invalid/v1",
            model="parent-model",
            api_key="secret",
            context_window=64_000,
            max_output_tokens=4_096,
        )
        executor = SkillExecutor(
            agent,
            MagicMock(),
            "openai-compat",
            provider_config=provider,
        )
        skill = SkillDef(
            name="review",
            description="Review code",
            mode="fork",
            model="review-model",
        )

        with patch("valecode.client.create_client", return_value=switched_client) as create:
            runtime = executor._resolve_fork_runtime(skill)

        assert runtime == (
            switched_client,
            "openai-compat",
            64_000,
            "skill-review",
            "review-model",
        )
        created_config = create.call_args.args[0]
        assert created_config.base_url == provider.base_url
        assert created_config.api_key == "secret"
        assert created_config.model == "review-model"

    def test_fork_skill_model_without_provider_fails_loudly(self) -> None:
        agent = MagicMock()
        agent.context_window = 100
        agent.provider_name = "default"
        agent.model = "parent-model"
        executor = SkillExecutor(agent, MagicMock(), "anthropic")
        skill = SkillDef(
            name="review",
            description="Review code",
            mode="fork",
            model="other-model",
        )

        with pytest.raises(ValueError, match="no Provider configuration"):
            executor._resolve_fork_runtime(skill)

    @pytest.mark.asyncio
    async def test_execute_fork_passes_declared_model_to_agent(self) -> None:
        from valecode.agent import LoopComplete
        from valecode.config import ProviderConfig

        parent = MagicMock()
        parent.registry = ToolRegistry()
        parent.work_dir = "."
        parent.max_iterations = 3
        parent.permission_checker = None
        parent.context_window = 100
        parent.execution_controller = MagicMock()
        parent.cancellation_token = MagicMock()
        parent.provider_name = "default"
        parent.model = "parent-model"
        parent.recovery_state = None
        provider = ProviderConfig(
            name="default",
            protocol="openai-compat",
            base_url="https://example.invalid/v1",
            model="parent-model",
            api_key="secret",
            context_window=32_000,
        )
        created: dict = {}

        class ForkAgent:
            def __init__(self, **kwargs):
                created.update(kwargs)

            def activate_skill(self, *args):
                created["activated"] = args

            async def run(self, conversation):
                created["conversation"] = conversation
                yield LoopComplete(total_turns=1)

        skill = SkillDef(
            name="review",
            description="Review code",
            prompt_body="Review now",
            mode="fork",
            context="none",
            model="review-model",
        )
        executor = SkillExecutor(
            parent,
            MagicMock(),
            "openai-compat",
            provider_config=provider,
        )

        with (
            patch("valecode.client.create_client", return_value=MagicMock()),
            patch("valecode.agent.Agent", ForkAgent),
        ):
            result = await executor.execute_fork(skill, "")

        assert result == ""
        assert created["model"] == "review-model"
        assert created["provider_name"] == "skill-review"
        assert created["protocol"] == "openai-compat"
        assert created["context_window"] == 32_000
        assert created["activated"] == ("review", "Review now", {})

# ---------------------------------------------------------------------------
# Agent 集成
# ---------------------------------------------------------------------------

class TestAgentSkillIntegration:
    def test_env_context_does_not_include_active_skills(self) -> None:
        from valecode.prompts import build_environment_context

        env = build_environment_context(
            "/test",
            active_skills={"commit": "Do commit stuff"},
            skill_catalog="Available: commit",
        )
        assert "Active Skills" not in env
        assert "Do commit stuff" not in env
        assert "Available: commit" in env

    def test_agent_activate_and_clear(self) -> None:
        agent = MagicMock()
        agent.active_skills = {}

        from valecode.agent import Agent

        real_agent = MagicMock(spec=Agent)
        real_agent.active_skills = {}
        real_agent.activate_skill = Agent.activate_skill.__get__(real_agent)
        real_agent.clear_active_skills = Agent.clear_active_skills.__get__(real_agent)

        real_agent.activate_skill("test", "SOP")
        assert "test" in real_agent.active_skills

        real_agent.clear_active_skills()
        assert len(real_agent.active_skills) == 0

    def test_skill_permission_scope_activates_and_releases(self, tmp_path: Path) -> None:
        from pydantic import BaseModel

        from valecode.agent import Agent
        from valecode.permissions import (
            DangerousCommandDetector,
            PathSandbox,
            PermissionChecker,
            PermissionMode,
            RuleEngine,
        )
        from valecode.tools.base import Tool, ToolResult

        class Params(BaseModel):
            file_path: str

        class ScopedWrite(Tool):
            name = "WriteFile"
            description = "write"
            params_model = Params
            category = "write"

            async def execute(self, params):
                return ToolResult("ok")

        checker = PermissionChecker(
            detector=DangerousCommandDetector(),
            sandbox=PathSandbox(str(tmp_path)),
            rule_engine=RuleEngine(),
            mode=PermissionMode.DEFAULT,
        )
        agent = Agent.__new__(Agent)
        agent.active_skills = {}
        agent.permission_checker = checker
        tool = ScopedWrite()
        arguments = {"file_path": str(tmp_path / "allowed.txt")}

        assert checker.check(tool, arguments).effect == "ask"
        agent.activate_skill(
            "writer",
            "write files",
            {"allow": ["WriteFile(*)"]},
        )
        assert checker.check(tool, arguments).effect == "allow"
        assert "Skill 权限作用域" in checker.check(tool, arguments).reason

        agent.clear_active_skills()
        assert checker.check(tool, arguments).effect == "ask"

    def test_deny_wins_across_active_skill_scopes(self, tmp_path: Path) -> None:
        from pydantic import BaseModel

        from valecode.agent import Agent
        from valecode.permissions import (
            DangerousCommandDetector,
            PathSandbox,
            PermissionChecker,
            PermissionMode,
            RuleEngine,
        )
        from valecode.tools.base import Tool, ToolResult

        class Params(BaseModel):
            file_path: str

        class ScopedWrite(Tool):
            name = "WriteFile"
            description = "write"
            params_model = Params
            category = "write"

            async def execute(self, params):
                return ToolResult("ok")

        checker = PermissionChecker(
            detector=DangerousCommandDetector(),
            sandbox=PathSandbox(str(tmp_path)),
            rule_engine=RuleEngine(),
            mode=PermissionMode.DEFAULT,
        )
        agent = Agent.__new__(Agent)
        agent.active_skills = {}
        agent.permission_checker = checker
        arguments = {"file_path": str(tmp_path / "file.txt")}

        agent.activate_skill("allowing", "", {"allow": ["WriteFile(*)"]})
        agent.activate_skill("blocking", "", {"deny": ["WriteFile(*)"]})

        assert checker.check(ScopedWrite(), arguments).effect == "deny"
