from __future__ import annotations

import asyncio
import logging
import os
import re
import subprocess
import sys
from urllib.request import Request, urlopen
from urllib.error import URLError

from valecode.hooks.models import Action, ActionResult, HookContext

log = logging.getLogger(__name__)


async def execute_command(action: Action, ctx: HookContext) -> ActionResult:
    command = ctx.expand(action.command)
    executable_command = command
    if os.name == "nt":
        sleep_match = re.fullmatch(r"\s*sleep\s+(\d+(?:\.\d+)?)\s*", command)
        if sleep_match:
            executable_command = subprocess.list2cmdline(
                [
                    sys.executable,
                    "-c",
                    f"import time; time.sleep({float(sleep_match.group(1))})",
                ]
            )
    try:
        proc = await asyncio.create_subprocess_shell(
            executable_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            stdout, _ = await asyncio.wait_for(
                proc.communicate(), timeout=action.timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return ActionResult(
                output=f"Command timed out after {action.timeout}s: {command}",
                success=False,
            )
        output = stdout.decode(errors="replace").strip() if stdout else ""
        return ActionResult(output=output, success=proc.returncode == 0)
    except Exception as e:
        return ActionResult(output=f"Command execution error: {e}", success=False)


async def execute_prompt(action: Action, ctx: HookContext) -> ActionResult:
    message = ctx.expand(action.message)
    return ActionResult(output=message, success=True)


async def execute_http(action: Action, ctx: HookContext) -> ActionResult:
    url = ctx.expand(action.url)
    body = ctx.expand(action.body) if action.body else None
    method = action.method or "POST"

    headers = dict(action.headers)
    for k, v in headers.items():
        headers[k] = ctx.expand(v)
    if body and "Content-Type" not in headers:
        headers["Content-Type"] = "application/json"


    def _do_request() -> ActionResult:
        try:
            data = body.encode() if body else None
            req = Request(url, data=data, headers=headers, method=method)
            with urlopen(req, timeout=30) as resp:
                resp_body = resp.read().decode(errors="replace")[:500]
                return ActionResult(
                    output=f"HTTP {resp.status}: {resp_body}",
                    success=200 <= resp.status < 300,
                )
        except URLError as e:
            return ActionResult(output=f"HTTP error: {e}", success=False)
        except Exception as e:
            return ActionResult(output=f"HTTP error: {e}", success=False)

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _do_request)


async def execute_agent(action: Action, ctx: HookContext) -> ActionResult:
    prompt = ctx.expand(action.prompt)
    if ctx.agent_runner is None:
        return ActionResult(
            output="Agent hook is unavailable outside an Agent runtime",
            success=False,
        )
    output = await ctx.agent_runner(prompt)
    return ActionResult(output=output, success=True)


_EXECUTOR_MAP = {
    "command": execute_command,
    "prompt": execute_prompt,
    "http": execute_http,
    "agent": execute_agent,
}


async def execute_action(action: Action, ctx: HookContext) -> ActionResult:
    executor = _EXECUTOR_MAP.get(action.type)
    if executor is None:
        return ActionResult(
            output=f"Unknown action type: {action.type}",
            success=False,
        )
    return await executor(action, ctx)
