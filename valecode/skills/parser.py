from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

log = logging.getLogger(__name__)

VALID_NAME_RE = re.compile(r"^[a-z][a-z0-9\-]*$")
VALID_MODES = {"inline", "fork"}
VALID_CONTEXTS = {"full", "recent", "none"}
VALID_PERMISSION_EFFECTS = {"allow", "deny", "ask"}
_PERMISSION_RULE_RE = re.compile(r"^\w+\(.+\)$")


class SkillParseError(Exception):
    pass


@dataclass
class SkillDef:
    name: str
    description: str
    prompt_body: str = ""
    mode: Literal["inline", "fork"] = "inline"
    model: str | None = None
    context: Literal["full", "recent", "none"] = "full"
    source_path: Path | None = None
    is_directory: bool = False
    permission_rules: dict[str, list[str]] = field(default_factory=dict)


def parse_skill_permissions(
    meta: dict, source: str = ""
) -> dict[str, list[str]]:
    """Parse scoped tool rules declared by a skill.

    Preferred syntax is ``permissions: {allow|deny|ask: [Tool(pattern)]}``.
    ``allowed-tools`` and ``disallowed-tools`` accept bare tool names as a
    compatibility shorthand, which expands to ``Tool(*)``.
    """

    ctx = f" in {source}" if source else ""
    collected: dict[str, list[str]] = {effect: [] for effect in VALID_PERMISSION_EFFECTS}
    raw = meta.get("permissions", {})
    if raw is not None and not isinstance(raw, dict):
        raise SkillParseError(f"permissions must be a mapping{ctx}")
    for effect, entries in (raw or {}).items():
        if effect not in VALID_PERMISSION_EFFECTS:
            raise SkillParseError(f"Invalid permission effect '{effect}'{ctx}")
        if isinstance(entries, str):
            entries = [entries]
        if not isinstance(entries, list) or not all(
            isinstance(entry, str) for entry in entries
        ):
            raise SkillParseError(f"permissions.{effect} must be a string list{ctx}")
        collected[effect].extend(entries)

    for key, effect in (("allowed-tools", "allow"), ("disallowed-tools", "deny")):
        entries = meta.get(key, [])
        if isinstance(entries, str):
            entries = [entries]
        if not isinstance(entries, list) or not all(
            isinstance(entry, str) for entry in entries
        ):
            raise SkillParseError(f"{key} must be a string list{ctx}")
        collected[effect].extend(
            entry if "(" in entry else f"{entry}(*)" for entry in entries
        )

    for effect, entries in collected.items():
        for entry in entries:
            if not _PERMISSION_RULE_RE.fullmatch(entry.strip()):
                raise SkillParseError(
                    f"Invalid permissions.{effect} rule '{entry}'{ctx}: "
                    "expected ToolName(pattern)"
                )
    return {effect: entries for effect, entries in collected.items() if entries}


def parse_frontmatter(raw: str) -> tuple[dict, str]:
    stripped = raw.lstrip()
    if not stripped.startswith("---"):
        raise SkillParseError("Missing YAML frontmatter (must start with ---)")

    end = stripped.find("---", 3)
    if end == -1:
        raise SkillParseError("Unclosed YAML frontmatter (missing closing ---)")

    yaml_block = stripped[3:end]
    body = stripped[end + 3:].lstrip("\n")

    try:
        meta = yaml.safe_load(yaml_block)
    except yaml.YAMLError as e:
        raise SkillParseError(f"Invalid YAML in frontmatter: {e}") from e

    if not isinstance(meta, dict):
        raise SkillParseError("Frontmatter must be a YAML mapping")

    return meta, body


def _validate_meta(meta: dict, source: str = "") -> None:
    ctx = f" in {source}" if source else ""

    if "name" not in meta:
        raise SkillParseError(f"Missing required field 'name'{ctx}")
    if "description" not in meta:
        raise SkillParseError(f"Missing required field 'description'{ctx}")

    name = meta["name"]
    if not isinstance(name, str) or not VALID_NAME_RE.match(name):
        raise SkillParseError(
            f"Invalid skill name '{name}'{ctx}: "
            "must be lowercase letters, digits, and hyphens, starting with a letter"
        )

    mode = meta.get("mode", "inline")
    if mode not in VALID_MODES:
        raise SkillParseError(f"Invalid mode '{mode}'{ctx}: must be one of {VALID_MODES}")

    context = meta.get("context", "full")
    if context not in VALID_CONTEXTS:
        raise SkillParseError(f"Invalid context '{context}'{ctx}: must be one of {VALID_CONTEXTS}")

    model = meta.get("model")
    if model is not None and (not isinstance(model, str) or not model.strip()):
        raise SkillParseError(f"Invalid model '{model}'{ctx}: must be a non-empty string")

    parse_skill_permissions(meta, source)


def parse_skill_file(path: Path) -> SkillDef:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as e:
        raise SkillParseError(f"Cannot read skill file {path}: {e}") from e

    meta, body = parse_frontmatter(raw)
    _validate_meta(meta, str(path))

    return SkillDef(
        name=meta["name"],
        description=meta["description"],
        prompt_body=body,
        mode=meta.get("mode", "inline"),
        model=meta["model"].strip() if meta.get("model") else None,
        context=meta.get("context", "full"),
        permission_rules=parse_skill_permissions(meta, str(path)),
        source_path=path,
        is_directory=False,
    )


def substitute_arguments(prompt_body: str, args: str) -> str:
    """将 $ARGUMENTS 占位符替换为用户请求（对齐 Go 版 promptHandler 逻辑）。

    若 prompt_body 中不含 $ARGUMENTS 占位符且 args 非空，
    则将用户请求追加到末尾（append fallback）。
    """
    if "$ARGUMENTS" in prompt_body:
        return prompt_body.replace("$ARGUMENTS", args)
    # 无占位符时的 append fallback
    if args.strip():
        return prompt_body + "\n\n## User Request\n\n" + args
    return prompt_body
