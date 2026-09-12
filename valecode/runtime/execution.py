from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from valecode.runtime.retry import _float_value, _int_value, _layered_env


T = TypeVar("T")


class ExecutionTimeoutError(TimeoutError):
    def __init__(self, operation: str, timeout: float) -> None:
        self.operation = operation
        self.timeout = timeout
        super().__init__(f"{operation} timed out after {timeout:g}s")


class CancellationToken:
    """Cooperative cancellation signal shared by a run and its descendants."""

    def __init__(self) -> None:
        self._event = asyncio.Event()
        self._reason = "Execution cancelled"

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str:
        return self._reason

    def cancel(self, reason: str = "Execution cancelled") -> None:
        if not self._event.is_set():
            self._reason = reason
            self._event.set()

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise asyncio.CancelledError(self._reason)

    async def wait(self) -> None:
        await self._event.wait()


@dataclass(frozen=True)
class ExecutionLimits:
    max_concurrent_tools: int = 8
    llm_timeout: float = 300.0
    tool_timeout: float = 120.0
    permission_timeout: float = 0.0

    @classmethod
    def from_environment(cls, work_dir: str | Path = ".") -> ExecutionLimits:
        values = _layered_env(work_dir)
        return cls(
            max_concurrent_tools=max(
                1, _int_value(values, "VALECODE_MAX_CONCURRENT_TOOLS", 8)
            ),
            llm_timeout=_float_value(values, "VALECODE_LLM_TIMEOUT", 300.0),
            tool_timeout=_float_value(values, "VALECODE_TOOL_TIMEOUT", 120.0),
            permission_timeout=_float_value(
                values, "VALECODE_PERMISSION_TIMEOUT", 0.0
            ),
        )


class ExecutionController:
    """Shared timeout, cancellation and tool-capacity control plane."""

    def __init__(self, limits: ExecutionLimits | None = None) -> None:
        self.limits = limits or ExecutionLimits()
        self._tool_capacity = asyncio.Semaphore(self.limits.max_concurrent_tools)

    @classmethod
    def from_environment(cls, work_dir: str | Path = ".") -> ExecutionController:
        return cls(ExecutionLimits.from_environment(work_dir))

    async def wait(
        self,
        awaitable: Awaitable[T],
        *,
        token: CancellationToken,
        timeout: float = 0.0,
        operation: str = "operation",
    ) -> T:
        token.raise_if_cancelled()
        operation_task = asyncio.ensure_future(awaitable)
        cancellation_task = asyncio.create_task(token.wait())
        try:
            done, _ = await asyncio.wait(
                {operation_task, cancellation_task},
                timeout=timeout if timeout > 0 else None,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if cancellation_task in done:
                operation_task.cancel()
                await asyncio.gather(operation_task, return_exceptions=True)
                raise asyncio.CancelledError(token.reason)
            if operation_task in done:
                return await operation_task
            operation_task.cancel()
            await asyncio.gather(operation_task, return_exceptions=True)
            raise ExecutionTimeoutError(operation, timeout)
        except asyncio.CancelledError:
            operation_task.cancel()
            await asyncio.gather(operation_task, return_exceptions=True)
            raise
        finally:
            cancellation_task.cancel()
            await asyncio.gather(cancellation_task, return_exceptions=True)

    async def execute_tool(
        self,
        factory: Callable[[], Awaitable[T]],
        *,
        token: CancellationToken,
        tool_name: str,
        use_capacity: bool = True,
    ) -> T:
        if not use_capacity:
            return await self.wait(
                factory(),
                token=token,
                timeout=self.limits.tool_timeout,
                operation=f"tool {tool_name}",
            )
        await self.wait(
            self._tool_capacity.acquire(),
            token=token,
            operation=f"tool capacity for {tool_name}",
        )
        try:
            return await self.wait(
                factory(),
                token=token,
                timeout=self.limits.tool_timeout,
                operation=f"tool {tool_name}",
            )
        finally:
            self._tool_capacity.release()

    async def stream(
        self,
        stream: AsyncIterator[T],
        *,
        token: CancellationToken,
        operation: str = "LLM stream",
    ) -> AsyncIterator[T]:
        loop = asyncio.get_running_loop()
        timeout = self.limits.llm_timeout
        deadline = loop.time() + timeout if timeout > 0 else None
        iterator = stream.__aiter__()
        while True:
            remaining = max(0.0, deadline - loop.time()) if deadline else 0.0
            if deadline is not None and remaining <= 0:
                raise ExecutionTimeoutError(operation, timeout)
            try:
                item = await self.wait(
                    iterator.__anext__(),
                    token=token,
                    timeout=remaining,
                    operation=operation,
                )
            except StopAsyncIteration:
                return
            yield item
