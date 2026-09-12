from __future__ import annotations

import os
from pathlib import Path


def canonical_path(path: str | Path) -> str:
    """Canonical comparison form with Windows case/drive normalization."""
    return os.path.normcase(str(Path(path).resolve(strict=False)))


def is_path_within(path: str | Path, root: str | Path) -> bool:
    candidate = canonical_path(path)
    boundary = canonical_path(root)
    try:
        return os.path.commonpath((candidate, boundary)) == boundary
    except ValueError:
        # Different Windows drives or incompatible UNC/local path forms.
        return False


def require_path_within(
    path: str | Path, root: str | Path, *, label: str = "path"
) -> Path:
    resolved = Path(path).resolve(strict=False)
    if not is_path_within(resolved, root) or canonical_path(resolved) == canonical_path(root):
        raise ValueError(f"{label} escapes its managed directory: {resolved}")
    return resolved
