"""Test harness.

Every test runs against an in-process app whose Ollama transport is an
``httpx.MockTransport``: no live Ollama, no network, no sleeping.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Iterator

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

API_KEY = "sk-test-001"
BAD_KEY = "sk-not-a-key"
MODEL = "qwen2.5:14b"
EMBED_MODEL = "nomic-embed-text"

Handler = Callable[[httpx.Request], httpx.Response]


def make_settings(**overrides: Any) -> Settings:
    """Deterministic settings: init kwargs outrank both env and .env."""
    values: dict[str, Any] = {
        "ollama_base_url": "http://ollama.test",
        "default_model": MODEL,
        "embed_model": EMBED_MODEL,
        "keystore_backend": "static",
        "api_keys": API_KEY,
        "request_timeout_s": 30.0,
        "log_level": "WARNING",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def ndjson_stream(lines: list[dict[str, Any]]) -> Any:
    """An async byte iterator emitting one JSON object per line, like Ollama."""

    async def generator() -> Any:
        for line in lines:
            yield (json.dumps(line) + "\n").encode()

    return generator()


def auth(key: str = API_KEY) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


@pytest.fixture
def build_client() -> Iterator[Callable[..., TestClient]]:
    """Factory: ``build_client(handler, **settings_overrides) -> TestClient``."""
    clients: list[TestClient] = []

    def build(handler: Handler, **overrides: Any) -> TestClient:
        app = create_app(
            make_settings(**overrides), transport=httpx.MockTransport(handler)
        )
        client = TestClient(app)
        client.__enter__()  # run lifespan
        clients.append(client)
        return client

    yield build

    for client in reversed(clients):
        client.__exit__(None, None, None)


def sse_events(raw: str) -> list[str]:
    """Split an SSE body into its ``data:`` payloads (blank-line separated)."""
    return [
        line[len("data: ") :]
        for line in raw.splitlines()
        if line.startswith("data: ")
    ]
