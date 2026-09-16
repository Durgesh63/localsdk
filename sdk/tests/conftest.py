"""Shared fixtures. Every test runs against ``httpx.MockTransport``: no
network, no server, no Ollama.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Iterable

import httpx
import pytest

import localsdk.config as config_module
from localsdk import AsyncClient, Client

BASE_URL = "https://test-tunnel.ngrok-free.app"
API_KEY = "sk-test-001"

Handler = Callable[[httpx.Request], httpx.Response]


@pytest.fixture(autouse=True)
def isolated_config(monkeypatch: pytest.MonkeyPatch, tmp_path) -> Any:
    """Keep the developer's real env and ~/.localsdk/config.toml out of tests."""
    for name in (
        "LOCALSDK_API_KEY",
        "LOCALSDK_BASE_URL",
        "LOCALSDK_MODEL",
        "LOCALSDK_TIMEOUT",
        "LOCALSDK_MAX_RETRIES",
    ):
        monkeypatch.delenv(name, raising=False)
    missing = tmp_path / "absent" / "config.toml"
    monkeypatch.setattr(config_module, "default_config_path", lambda: missing)
    return missing


@pytest.fixture(autouse=True)
def no_real_sleeping(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make retry backoff instantaneous."""
    import asyncio
    import time

    monkeypatch.setattr(time, "sleep", lambda _s: None)

    async def _no_sleep(_s: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", _no_sleep)


# --------------------------------------------------------------------------- #
# response builders
# --------------------------------------------------------------------------- #


def chat_body(
    content: str | None = "Hello there.",
    *,
    finish_reason: str = "stop",
    tool_calls: list[dict[str, Any]] | None = None,
    model: str = "qwen2.5:14b",
) -> dict[str, Any]:
    """A contract-shaped chat.completion body."""
    return {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tool_calls,
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }


def error_body(message: str, *, type: str, code: str) -> dict[str, Any]:
    """The OpenAI error envelope from CONTRACT section 6."""
    return {"error": {"message": message, "type": type, "param": None, "code": code}}


def sse(*events: str) -> str:
    """Join SSE event payloads with the blank-line separator the contract requires."""
    return "".join("data: {0}\n\n".format(event) for event in events)


def sse_chunk(content: str | None = None, *, finish_reason: str | None = None) -> str:
    """One chat.completion.chunk as a JSON string."""
    delta: dict[str, Any] = {}
    if content is not None:
        delta["content"] = content
    return json.dumps(
        {
            "id": "chatcmpl-x",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "qwen2.5:14b",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
    )


# --------------------------------------------------------------------------- #
# handlers / clients
# --------------------------------------------------------------------------- #


def json_handler(
    payload: Any,
    *,
    status_code: int = 200,
    headers: dict[str, str] | None = None,
    record: list[httpx.Request] | None = None,
) -> Handler:
    """A handler that always answers with the same JSON body."""

    def handler(request: httpx.Request) -> httpx.Response:
        if record is not None:
            record.append(request)
        return httpx.Response(status_code, json=payload, headers=headers)

    return handler


def text_handler(
    body: str,
    *,
    status_code: int = 200,
    content_type: str = "text/plain",
) -> Handler:
    """A handler that answers with a raw (non-JSON) body."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code, text=body, headers={"content-type": content_type}
        )

    return handler


def sequence_handler(responses: Iterable[Callable[[httpx.Request], httpx.Response]]) -> Handler:
    """Answer each successive request with the next factory in ``responses``."""
    queue = list(responses)
    index = {"i": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        i = min(index["i"], len(queue) - 1)
        index["i"] += 1
        return queue[i](request)

    return handler


def make_client(handler: Handler, **kwargs: Any) -> Client:
    """A sync client wired to a mock transport."""
    kwargs.setdefault("api_key", API_KEY)
    kwargs.setdefault("base_url", BASE_URL)
    return Client(transport=httpx.MockTransport(handler), **kwargs)


def make_async_client(handler: Handler, **kwargs: Any) -> AsyncClient:
    """An async client wired to a mock transport."""
    kwargs.setdefault("api_key", API_KEY)
    kwargs.setdefault("base_url", BASE_URL)
    return AsyncClient(transport=httpx.MockTransport(handler), **kwargs)
