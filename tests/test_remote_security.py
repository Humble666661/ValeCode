from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import websockets

from valecode.remote import RemoteServer, _is_loopback_bind


def _request(path: str, headers: dict[str, str] | None = None) -> SimpleNamespace:
    return SimpleNamespace(path=path, headers=websockets.Headers(headers or {}))


def test_remote_defaults_to_loopback_without_authentication() -> None:
    server = RemoteServer(providers=[])

    assert server.addr == "127.0.0.1"
    assert server.auth_token == ""
    assert server._process_http_request(None, _request("/ws")) is None


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost"])
def test_loopback_bind_detection(host: str) -> None:
    assert _is_loopback_bind(host) is True


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.10", "remote.test"])
def test_non_loopback_bind_requires_token(host: str) -> None:
    with pytest.raises(ValueError, match="token is required"):
        RemoteServer(providers=[], addr=host)


def test_websocket_rejects_missing_or_incorrect_token() -> None:
    server = RemoteServer(providers=[], auth_token="expected-secret")

    missing = server._process_http_request(None, _request("/ws"))
    incorrect = server._process_http_request(None, _request("/ws?token=wrong"))

    assert missing is not None and missing.status_code == 401
    assert incorrect is not None and incorrect.status_code == 401
    assert missing.headers["WWW-Authenticate"] == 'Bearer realm="ValeCode Remote"'


def test_websocket_accepts_query_or_bearer_token() -> None:
    server = RemoteServer(providers=[], auth_token="expected-secret")

    query_result = server._process_http_request(
        None, _request("/ws?token=expected-secret")
    )
    bearer_result = server._process_http_request(
        None,
        _request("/ws", {"Authorization": "Bearer expected-secret"}),
    )

    assert query_result is None
    assert bearer_result is None


def test_remote_page_only_exposes_auth_requirement_not_token() -> None:
    server = RemoteServer(providers=[], auth_token="must-not-leak")

    response = server._process_http_request(None, _request("/"))

    assert response is not None and response.status_code == 200
    html = bytes(response.body).decode("utf-8")
    assert "const remoteAuthRequired = true;" in html
    assert "must-not-leak" not in html


@pytest.mark.asyncio
async def test_live_websocket_handshake_enforces_token() -> None:
    server = RemoteServer(providers=[], auth_token="expected-secret")
    async with websockets.serve(
        server._ws_handler,
        "127.0.0.1",
        0,
        process_request=server._process_http_request,
    ) as listening:
        port = listening.sockets[0].getsockname()[1]
        with pytest.raises(websockets.exceptions.InvalidStatus) as rejected:
            async with websockets.connect(f"ws://127.0.0.1:{port}/ws"):
                pass
        assert rejected.value.response.status_code == 401

        async with websockets.connect(
            f"ws://127.0.0.1:{port}/ws?token=expected-secret"
        ) as websocket:
            message = json.loads(await websocket.recv())
            assert message["type"] == "connected"
