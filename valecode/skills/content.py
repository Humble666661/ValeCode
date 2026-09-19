from __future__ import annotations

import os
from pathlib import Path

from valecode.skills.parser import SkillDef

MAX_SKILL_FILES = 10
MAX_SCAN_ENTRIES = 256
MAX_SCAN_DEPTH = 4
_ENTRY_FILES = {"SKILL.md", "prompt.md", "skill.yaml"}
_SKIP_DIRS = {".git", "__pycache__"}


def _skill_base_dir(skill: SkillDef) -> Path | None:
    if not skill.is_directory or skill.source_path is None:
        return None
    try:
        return skill.source_path.parent.resolve()
    except OSError:
        return None


def list_skill_support_files(
    skill: SkillDef, *, limit: int = MAX_SKILL_FILES,
) -> list[Path]:
    """Return a bounded, deterministic sample of regular package files."""
    root = _skill_base_dir(skill)
    if root is None or not root.is_dir() or limit <= 0:
        return []

    files: list[Path] = []
    scanned = 0
    for current, dirs, names in os.walk(root, followlinks=False):
        current_path = Path(current)
        try:
            depth = len(current_path.relative_to(root).parts)
        except ValueError:
            continue
        dirs[:] = sorted(
            name
            for name in dirs
            if name not in _SKIP_DIRS and not name.startswith(".")
        )
        if depth >= MAX_SCAN_DEPTH:
            dirs[:] = []
        for name in sorted(names):
            scanned += 1
            if scanned > MAX_SCAN_ENTRIES:
                return files
            if name in _ENTRY_FILES or name.startswith("."):
                continue
            path = current_path / name
            try:
                resolved = path.resolve()
                resolved.relative_to(root)
            except (OSError, ValueError):
                continue
            if path.is_symlink() or not resolved.is_file():
                continue
            files.append(resolved)
            if len(files) >= min(limit, MAX_SKILL_FILES):
                return files
    return files


def render_skill_content(skill: SkillDef, prompt: str | None = None) -> str:
    """Render an activated Skill with safe package-location context."""
    body = skill.prompt_body if prompt is None else prompt
    lines = [
        f'<skill_content name="{skill.name}">',
        f"# Skill: {skill.name}",
        "",
        body.strip(),
    ]
    base_dir = _skill_base_dir(skill)
    if base_dir is not None:
        files = list_skill_support_files(skill)
        lines.extend([
            "",
            f"Base directory for this Skill: {base_dir}",
            (
                "Relative paths such as scripts/ and references/ are relative "
                "to this directory. Use the normal file and shell tools to access "
                "them; all ordinary permissions and sandbox rules still apply."
            ),
            "The following support-file list is a bounded sample:",
            "",
            "<skill_files>",
            *(f"<file>{path}</file>" for path in files),
            "</skill_files>",
        ])
    lines.append("</skill_content>")
    return "\n".join(lines)
