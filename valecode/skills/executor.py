from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from valecode.conversation import ConversationManager, Message
from valecode.skills.parser import SkillDef, substitute_arguments

if TYPE_CHECKING:
    from valecode.agent import Agent, AgentEvent
    from valecode.client import LLMClient
    from valecode.config import ProviderConfig

log = logging.getLogger(__name__)

FORK_RECENT_COUNT = 5


class SkillExecutor:


    def __init__(
        self,
        agent: Agent,
        client: LLMClient,
        protocol: str,
        provider_config: ProviderConfig | None = None,
    ) -> None:
        self.agent = agent
        self.client = client
        self.protocol = protocol
        self.provider_config = provider_config


    def execute_inline(
        self,
        skill: SkillDef,
        args: str,
        conversation: ConversationManager | None = None,
    ) -> str:
        prompt = substitute_arguments(skill.prompt_body, args)
        self.agent.activate_skill(skill.name, prompt, skill.permission_rules)
        target = conversation or getattr(self.agent, "_current_conversation", None)
        if target is not None:
            target.add_system_reminder(f"# Skill: {skill.name}\n\n{prompt}")
        if getattr(self.agent, "recovery_state", None) is not None:
            self.agent.recovery_state.record_skill_invocation(skill.name, prompt)
        return prompt

    def _resolve_fork_runtime(
        self, skill: SkillDef,
    ) -> tuple[LLMClient, str, int, str | None, str | None]:
        inherited = (
            self.client,
            self.protocol,
            self.agent.context_window,
            self.agent.provider_name,
            self.agent.model,
        )
        if not skill.model or skill.model == "inherit":
            return inherited
        if self.provider_config is None:
            raise ValueError(
                f"Skill '{skill.name}' declares model '{skill.model}', but the "
                "current runtime has no Provider configuration for model switching"
            )

        from valecode.client import create_client
        from valecode.config import ProviderConfig

        model_map = {
            "haiku": "claude-haiku-4-5-20251001",
            "sonnet": "claude-sonnet-4-6-20250514",
            "opus": "claude-opus-4-6-20250514",
        }
        model_id = model_map.get(skill.model, skill.model)
        base = self.provider_config
        config = ProviderConfig(
            name=f"skill-{skill.name}",
            protocol=base.protocol,
            base_url=base.base_url,
            model=model_id,
            api_key=base.api_key,
            thinking=base.thinking,
            context_window=base.context_window,
            max_output_tokens=base.max_output_tokens,
        )
        try:
            client = create_client(config)
        except Exception as exc:
            raise ValueError(
                f"Unable to initialize model '{skill.model}' for Skill "
                f"'{skill.name}': {exc}"
            ) from exc
        return (
            client,
            config.protocol,
            config.get_context_window(),
            config.name,
            model_id,
        )


    async def execute_fork(
        self, skill: SkillDef, args: str
    ) -> str:
        prompt = substitute_arguments(skill.prompt_body, args)
        if getattr(self.agent, "recovery_state", None) is not None:
            self.agent.recovery_state.record_skill_invocation(
                skill.name, skill.prompt_body
            )

        fork_conv = ConversationManager()

        context_messages = self._build_fork_context(skill.context)
        for msg in context_messages:
            if msg.role == "user":
                fork_conv.add_user_message(msg.content)
            else:
                fork_conv.add_assistant_message(msg.content)

        fork_conv.add_user_message(prompt)

        from valecode.agent import Agent as AgentClass, StreamText, LoopComplete, ErrorEvent

        client, protocol, context_window, provider_name, model = (
            self._resolve_fork_runtime(skill)
        )

        fork_agent = AgentClass(
            client=client,
            registry=self.agent.registry,
            protocol=protocol,
            work_dir=self.agent.work_dir,
            max_iterations=self.agent.max_iterations,
            permission_checker=(
                self.agent.permission_checker.clone()
                if self.agent.permission_checker is not None
                else None
            ),
            context_window=context_window,
            execution_controller=self.agent.execution_controller,
            cancellation_token=self.agent.cancellation_token,
            provider_name=provider_name,
            model=model,
        )
        fork_agent.activate_skill(
            skill.name, prompt, skill.permission_rules
        )

        result_parts: list[str] = []
        async for event in fork_agent.run(fork_conv):
            if isinstance(event, StreamText):
                result_parts.append(event.text)
            elif isinstance(event, ErrorEvent):
                result_parts.append(f"\n[Error: {event.message}]")
            elif isinstance(event, LoopComplete):
                break

        return "".join(result_parts)


    def _build_fork_context(self, mode: str) -> list[Message]:
        if mode == "none":
            return []

        history = self.agent._conversation.history if hasattr(self.agent, '_conversation') else []
        if not history:
            main_history = []
        else:
            main_history = history

        if mode == "recent":
            content_messages = [
                m for m in main_history
                if m.content and not m.tool_results
            ]
            return content_messages[-FORK_RECENT_COUNT:]

        if mode == "full":
            content_messages = [
                m for m in main_history
                if m.content and not m.tool_results
            ]
            if not content_messages:
                return []
            summary_parts = []
            for m in content_messages:
                prefix = "User" if m.role == "user" else "Assistant"
                text = m.content[:200]
                if len(m.content) > 200:
                    text += "..."
                summary_parts.append(f"{prefix}: {text}")
            summary = "## Previous conversation summary\n\n" + "\n\n".join(summary_parts)
            return [Message(role="user", content=summary)]

        return []
