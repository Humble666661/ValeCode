---
name: customize-valecode
description: Use only when creating or editing ValeCode configuration, agents, skills, commands, hooks, MCP servers, permissions, sandbox, worktree, or project instruction files.
mode: inline
---
# Customize ValeCode

Use this Skill only for ValeCode's own configuration and extension files. Do not use it for ordinary application code merely because ValeCode is the coding agent.

Before changing configuration:

1. Inspect the repository's current `README.md`, `.env.example`, and relevant files under `.valecode/`; do not guess keys or formats.
2. Preserve the documented precedence between environment files and YAML configuration.
3. Keep API keys, remote tokens, and other secrets in `.env.local` or system environment variables. Never put real secrets in tracked files.
4. Treat installed Python plugins and Skill scripts as trusted code. Prefer MCP when an external integration needs process isolation.
5. Keep permission rules narrow. Skill permissions must not be used to bypass dangerous-command checks, external-path confirmation, or the OS sandbox.
6. After editing, run the smallest relevant test set and report any configuration that still requires user-supplied credentials or external services.

Common locations:

- Project instructions: `VALECODE.md`, `AGENTS.md`, `.valecode/INSTRUCTIONS.md`
- Project configuration: `.valecode/config.yaml`, `.valecode/config.local.yaml`
- Agents: `.valecode/agents/*.md`
- Skills: `.valecode/skills/<name>/SKILL.md` or `skill.yaml + prompt.md`
- Commands: `.valecode/commands/*.md`
- Permission rules: `.valecode/permissions.yaml`, `.valecode/permissions.local.yaml`
