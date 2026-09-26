"""Offline integration server for transport ownership and request tests."""
from mcp.server.fastmcp import FastMCP
import os

server = FastMCP("valecode-offline-test")


@server.tool()
def echo(text: str) -> str:
    return text


@server.tool()
def disconnect() -> str:
    os._exit(3)


@server.resource("valecode-test://greeting")
def greeting() -> str:
    return "hello from offline MCP"


@server.resource("valecode-test://greeting/{name}")
def named_greeting(name: str) -> str:
    return f"hello {name}"


@server.prompt()
def review(topic: str) -> str:
    return f"Review {topic} carefully."


if __name__ == "__main__":
    server.run(transport="stdio")
