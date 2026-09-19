from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any

from valecode.permissions.dangerous import DangerousCommandDetector, is_safe_command
from valecode.permissions.modes import DecisionEffect, PermissionMode, mode_decide
from valecode.permissions.rules import RuleEngine, extract_content, parse_rule
from valecode.permissions.sandbox import PathSandbox
from valecode.permissions.session_store import SessionAllowStore
from valecode.tools.base import Tool
from valecode.tools.todo_write import TodoWrite

_PLAN_MODE_ALLOWED_TOOLS = frozenset({"Agent", "ToolSearch", "AskUserQuestion", "ExitPlanMode"})


@dataclass
class Decision:
    effect: DecisionEffect
    reason: str


class PermissionChecker:


    def __init__(
        self,
        detector: DangerousCommandDetector,
        sandbox: PathSandbox,
        rule_engine: RuleEngine,
        mode: PermissionMode = PermissionMode.DEFAULT,
        sandbox_enabled: bool = False,
    ) -> None:
        self.detector = detector
        self.sandbox = sandbox
        self.rule_engine = rule_engine
        self.mode = mode
        self.plan_file_path: str = ""
        # OS 级沙箱是否启用（开启后命令类工具可自动放行，因为内核会兜底）
        self.sandbox_enabled = sandbox_enabled
        # Layer 4b: exact grants bound to the current conversation session.
        self._session_allowed: set[str] = set()
        self._session_allow_store: SessionAllowStore | None = None


    def bind_session(self, session_id: str) -> None:
        """Switch grants to one session; a new session never inherits them."""
        store = SessionAllowStore(self.sandbox.project_root, session_id)
        try:
            grants = store.load()
        except (OSError, ValueError):
            grants = set()  # Corrupt state must never grant permission.
        self._session_allow_store = store
        self._session_allowed = grants

    @staticmethod
    def _session_fingerprint(tool_name: str, arguments: dict[str, Any]) -> str:
        content = extract_content(tool_name, arguments)
        # Known tools use a stable action field (e.g. Bash command or file
        # path). Unknown tools use their full argument object, never a broad
        # empty-string grant for every invocation of that tool.
        action: Any = content if content else arguments
        payload = json.dumps(
            [tool_name, action], ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def add_session_allow(self, tool_name: str, arguments: dict[str, Any]) -> None:
        """Remember one exact action for this session, including after resume."""
        fingerprint = self._session_fingerprint(tool_name, arguments)
        if self._session_allow_store is not None:
            try:
                # Cloned checkers may have learned grants since this instance
                # was created; do not overwrite them with a stale snapshot.
                self._session_allowed.update(self._session_allow_store.load())
            except (OSError, ValueError):
                pass
        self._session_allowed.add(fingerprint)
        if self._session_allow_store is not None:
            try:
                self._session_allow_store.save(self._session_allowed)
            except (OSError, ValueError):
                pass  # In-process grant remains valid; no broader rule is made.

    def _check_session_allowed(self, tool_name: str, arguments: dict[str, Any]) -> bool:
        return self._session_fingerprint(tool_name, arguments) in self._session_allowed

    def bind_skill_scope(
        self, skill_name: str, permission_rules: dict[str, list[str]]
    ) -> None:
        rules = [
            parse_rule(raw, effect)  # type: ignore[arg-type]
            for effect, entries in permission_rules.items()
            for raw in entries
        ]
        self.rule_engine.bind_scope(f"skill:{skill_name}", rules)

    def release_skill_scope(self, skill_name: str) -> None:
        self.rule_engine.release_scope(f"skill:{skill_name}")

    def clone(self) -> PermissionChecker:
        cloned = PermissionChecker(
            detector=self.detector,
            sandbox=self.sandbox,
            rule_engine=self.rule_engine.clone(),
            mode=self.mode,
            sandbox_enabled=self.sandbox_enabled,
        )
        cloned.plan_file_path = self.plan_file_path
        cloned._session_allowed = set(self._session_allowed)
        cloned._session_allow_store = self._session_allow_store
        return cloned

    @staticmethod
    def describe_tool_action(tool_name: str, arguments: dict[str, Any]) -> str:
        """为 HITL 确认生成人类可读的操作描述（对齐 Go 版 ExtractContent + formatToolArgs）。"""
        content = extract_content(tool_name, arguments)
        if content:
            return content
        # 无法从标准字段提取时，拼接参数摘要
        parts = []
        for k, v in arguments.items():
            sv = str(v)
            if len(sv) > 80:
                sv = sv[:77] + "..."
            parts.append(f"{k}={sv}")
        return ", ".join(parts) if parts else tool_name


    def check(self, tool: Tool, arguments: dict[str, Any]) -> Decision:
        permission_name = tool.permission_name
        content = extract_content(permission_name, arguments)

        # Layer 0: Plan 模式例外放行
        if self.mode == PermissionMode.PLAN:
            if permission_name in _PLAN_MODE_ALLOWED_TOOLS:
                return Decision(effect="allow", reason="Plan mode: allowed tool")
            if permission_name in ("WriteFile", "EditFile") and content:
                if self._is_plan_file(content):
                    return Decision(effect="allow", reason="Plan mode: plan file write")

        # Layer 1: 安全的只读命令（自动放行）
        if tool.category == "command" and is_safe_command(content or ""):
            safe_rule = self.rule_engine.evaluate(permission_name, content)
            if safe_rule == "deny":
                return Decision(effect="deny", reason="权限规则拒绝")
            if safe_rule == "ask":
                return Decision(effect="ask", reason="权限规则要求确认")
            return Decision(effect="allow", reason="Safe read-only command")

        # Layer 1b: 危险命令黑名单（仅 Bash）
        if tool.category == "command":
            hit, reason = self.detector.detect(content)
            if hit:
                return Decision(effect="deny", reason=f"危险命令拦截: {reason}")

        # Layer 1c: OS 沙箱自动放行
        # 沙箱开启时，命令类工具通过了危险命令检查后直接放行——
        # 内核级隔离会阻止越权写入，无需再弹确认。
        # 对齐 Claude Code checkSandboxAutoAllow：拆分复合命令逐条检查，
        # deny 规则和 ask 规则不受沙箱影响。
        # Only Bash is executed through the attached OS backend.  Other
        # command-category tools (Agent, Team, Worktree, mailbox, tasks) must
        # continue through their normal permission rules.
        if (
            self.sandbox_enabled
            and tool.category == "command"
            and permission_name == "Bash"
        ):
            import re
            subcommands = [s.strip() for s in re.split(r'\s*(?:&&|\|\||[;|])\s*', content) if s.strip()]
            if not subcommands:
                subcommands = [content]
            has_ask = False
            for sub in subcommands:
                rule_result = self.rule_engine.evaluate(permission_name, sub)
                if rule_result == "deny":
                    return Decision(effect="deny", reason="权限规则拒绝")
                if rule_result == "ask":
                    has_ask = True
            if has_ask:
                return Decision(effect="ask", reason="权限规则要求确认")
            return Decision(effect="allow", reason="OS 沙箱自动放行")

        # Layer 2: 路径沙箱（仅文件类工具）
        if tool.category in ("read", "write") and content:
            ok, reason = self.sandbox.check(content)
            if not ok and self.mode != PermissionMode.BYPASS:
                return Decision(effect="ask", reason=f"路径沙箱拦截: {reason}")

        # Layer 3: 规则引擎匹配
        rule_result = self.rule_engine.evaluate(permission_name, content)
        if rule_result == "allow":
            return Decision(effect="allow", reason="权限规则放行")
        if rule_result == "deny":
            return Decision(effect="deny", reason="权限规则拒绝")
        if rule_result == "ask":
            return Decision(effect="ask", reason="权限规则要求确认")

        # Layer 3b: active Skill scopes. Persistent user/project rules above
        # retain authority; dangerous-command and path checks can never be
        # bypassed by a Skill declaration.
        scoped_result = self.rule_engine.evaluate_scoped(permission_name, content)
        if scoped_result is not None:
            return Decision(
                effect=scoped_result,
                reason=f"Skill 权限作用域 {scoped_result}",
            )

        # The built-in TodoWrite only updates this session's local progress
        # metadata. Keep explicit user/project deny/ask rules above authoritative
        # while avoiding a permission prompt for every checklist update.
        if isinstance(tool, TodoWrite):
            return Decision(effect="allow", reason="内置会话任务进度更新")

        # Layer 4b: session-only exact grant, after explicit persistent rules.
        if self._check_session_allowed(permission_name, arguments):
            return Decision(effect="allow", reason="当前会话已允许此操作")

        # Layer 4: 权限模式兜底判定
        effect = mode_decide(self.mode, tool.category)
        if effect == "allow":
            return Decision(effect="allow", reason=f"权限模式 {self.mode.value} 放行")
        if effect == "deny":
            return Decision(effect="deny", reason=f"权限模式 {self.mode.value} 拒绝")

        # Layer 5: 触发人工确认（HITL）
        return Decision(effect="ask", reason="需要用户确认")


    def _is_plan_file(self, target_path: str) -> bool:
        if not self.plan_file_path or not target_path:
            return ".valecode/plans/" in target_path
        try:
            abs_target = os.path.abspath(target_path)
            abs_plan = os.path.abspath(self.plan_file_path)
            if abs_target == abs_plan:
                return True
        except Exception:
            pass
        if os.path.basename(target_path) == os.path.basename(self.plan_file_path):
            return True
        return ".valecode/plans/" in target_path
