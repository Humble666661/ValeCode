from __future__ import annotations

import email.utils
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Mapping

from dotenv import dotenv_values


class ErrorCategory(StrEnum):
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    OVERLOADED = "overloaded"
    SERVER = "server"
    NETWORK = "network"
    INVALID_REQUEST = "invalid_request"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class RetryDecision:
    retry: bool
    category: ErrorCategory
    delay: float = 0.0
    attempt: int = 0
    reason: str = ""


def parse_retry_after(value: str | None, *, now: datetime | None = None) -> float | None:
    if not value:
        return None
    stripped = value.strip()
    try:
        return max(0.0, float(stripped))
    except ValueError:
        pass
    try:
        parsed = email.utils.parsedate_to_datetime(stripped)
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    current = now or datetime.now(UTC)
    return max(0.0, (parsed - current).total_seconds())


def classify_provider_error(error: BaseException) -> tuple[ErrorCategory, bool, float | None]:
    # Imported lazily to avoid a client -> runtime.retry -> client cycle.
    from valecode.client import (
        AuthenticationError,
        InvalidRequestError,
        NetworkError,
        OverloadedError,
        RateLimitError,
        ServerError,
    )

    if isinstance(error, AuthenticationError):
        return ErrorCategory.AUTHENTICATION, False, None
    if isinstance(error, RateLimitError):
        return ErrorCategory.RATE_LIMIT, True, error.retry_after
    if isinstance(error, OverloadedError):
        return ErrorCategory.OVERLOADED, True, error.retry_after
    if isinstance(error, ServerError):
        return ErrorCategory.SERVER, True, error.retry_after
    if isinstance(error, NetworkError):
        return ErrorCategory.NETWORK, True, None
    if isinstance(error, InvalidRequestError):
        return ErrorCategory.INVALID_REQUEST, False, None
    return ErrorCategory.UNKNOWN, False, None


@dataclass
class RetryPolicy:
    max_retries: int = 3
    base_delay: float = 0.5
    max_delay: float = 30.0
    max_total_wait: float = 60.0
    jitter_ratio: float = 0.2

    def decide(
        self,
        error: BaseException,
        *,
        retries_used: int,
        total_wait: float,
        random_value: float | None = None,
    ) -> RetryDecision:
        category, retryable, retry_after = classify_provider_error(error)
        attempt = retries_used + 1
        if not retryable:
            return RetryDecision(False, category, attempt=attempt, reason="not_retryable")
        if retries_used >= self.max_retries:
            return RetryDecision(False, category, attempt=attempt, reason="max_retries")

        exponential = min(self.max_delay, self.base_delay * (2**retries_used))
        if retry_after is not None:
            delay = min(self.max_delay, retry_after)
        else:
            sample = random.random() if random_value is None else random_value
            jitter = exponential * self.jitter_ratio * ((sample * 2.0) - 1.0)
            delay = max(0.0, min(self.max_delay, exponential + jitter))
        if total_wait + delay > self.max_total_wait:
            return RetryDecision(False, category, attempt=attempt, reason="max_total_wait")
        return RetryDecision(True, category, delay, attempt, "retryable")

    @classmethod
    def from_environment(cls, work_dir: str | Path = ".") -> RetryPolicy:
        values = _layered_env(work_dir)
        return cls(
            max_retries=_int_value(values, "VALECODE_LLM_MAX_RETRIES", 3),
            base_delay=_float_value(values, "VALECODE_LLM_RETRY_BASE_DELAY", 0.5),
            max_delay=_float_value(values, "VALECODE_LLM_RETRY_MAX_DELAY", 30.0),
            max_total_wait=_float_value(values, "VALECODE_LLM_RETRY_MAX_WAIT", 60.0),
            jitter_ratio=_float_value(values, "VALECODE_LLM_RETRY_JITTER", 0.2),
        )


def _layered_env(work_dir: str | Path) -> dict[str, str]:
    import os

    values: dict[str, str] = {}
    for path in (
        Path.home() / ".valecode" / ".env",
        Path(work_dir) / ".env",
        Path(work_dir) / ".env.local",
    ):
        try:
            if path.is_file():
                values.update(
                    {
                        key: value
                        for key, value in dotenv_values(path).items()
                        if value is not None
                    }
                )
        except OSError:
            continue
    values.update(os.environ)
    return values


def _int_value(values: Mapping[str, str], key: str, default: int) -> int:
    try:
        return max(0, int(values.get(key, default)))
    except (TypeError, ValueError):
        return default


def _float_value(values: Mapping[str, str], key: str, default: float) -> float:
    try:
        return max(0.0, float(values.get(key, default)))
    except (TypeError, ValueError):
        return default
