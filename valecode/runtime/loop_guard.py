from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from valecode.runtime.retry import _layered_env
from valecode.tools.base import ToolCallComplete


@dataclass(frozen=True)
class LoopDecision:
    blocked: bool
    signature: str
    repeat_count: int
    tool_name: str
    reason: str = ""


class LoopGuard:
    def __init__(self, repeat_limit: int = 3) -> None:
        self.repeat_limit = 0 if repeat_limit <= 0 else max(2, repeat_limit)
        self._last_signature = ""
        self._repeat_count = 0

    @classmethod
    def from_environment(cls, work_dir: str | Path = ".") -> LoopGuard:
        values = _layered_env(work_dir)
        try:
            limit = int(values.get("VALECODE_LOOP_REPEAT_LIMIT", 3))
        except (TypeError, ValueError):
            limit = 3
        return cls(limit)

    def observe(self, call: ToolCallComplete) -> LoopDecision:
        canonical = json.dumps(
            {"tool": call.tool_name, "arguments": call.arguments},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=repr,
        )
        signature = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if signature == self._last_signature:
            self._repeat_count += 1
        else:
            self._last_signature = signature
            self._repeat_count = 1
        blocked = self.repeat_limit > 0 and self._repeat_count >= self.repeat_limit
        return LoopDecision(
            blocked=blocked,
            signature=signature,
            repeat_count=self._repeat_count,
            tool_name=call.tool_name,
            reason=(
                f"Repeated identical {call.tool_name} call "
                f"{self._repeat_count} times"
                if blocked
                else ""
            ),
        )

    def reset(self) -> None:
        self._last_signature = ""
        self._repeat_count = 0
