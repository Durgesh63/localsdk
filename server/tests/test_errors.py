"""Every error path emits the CONTRACT section 6 envelope."""

from __future__ import annotations

import httpx
import pytest

from conftest import MODEL, auth

ENVELOPE_KEYS = {"message", "type", "param", "code"}


def ok_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": MODEL,
            "message": {"role": "assistant", "content": "hi"},
            "done": True,
        },
    )


def assert_envelope(body: dict, type_: str, code: str) -> None:
    assert set(body) == {"error"}
    error = body["error"]
    assert set(error) == ENVELOPE_KEYS
    assert error["type"] == type_
    assert error["code"] == code
    assert isinstance(error["message"], str) and error["message"]


def test_validation_error_uses_openai_envelope(build_client):
    """FastAPI's default 422 {"detail": [...]} must never reach the client."""
    client = build_client(ok_handler)
    response = client.post("/v1/chat/completions", headers=auth(), json={})
    assert response.status_code == 400
    assert_envelope(response.json(), "invalid_request_error", "invalid_request")
    assert response.json()["error"]["param"] == "messages"


def test_empty_messages_is_400(build_client):
    client = build_client(ok_handler)
    response = client.post(
        "/v1/chat/completions", headers=auth(), json={"messages": []}
    )
    assert response.status_code == 400
    assert_envelope(response.json(), "invalid_request_error", "invalid_request")


def test_bad_role_is_400(build_client):
    client = build_client(ok_handler)
    response = client.post(
        "/v1/chat/completions",
        headers=auth(),
        json={"messages": [{"role": "wizard", "content": "hi"}]},
    )
    assert response.status_code == 400


def test_tool_message_without_tool_call_id_is_400(build_client):
    client = build_client(ok_handler)
    response = client.post(
        "/v1/chat/completions",
        headers=auth(),
        json={"messages": [{"role": "tool", "content": "31C"}]},
    )
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "messages[0].tool_call_id"


def test_ollama_down_is_502(build_client):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = build_client(handler)
    response = client.post(
        "/v1/chat/completions",
        headers=auth(),
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 502
    assert_envelope(response.json(), "upstream_error", "ollama_unavailable")


def test_ollama_timeout_is_504(build_client):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    client = build_client(handler)
    response = client.post(
        "/v1/chat/completions",
        headers=auth(),
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 504
    assert_envelope(response.json(), "upstream_error", "ollama_timeout")


def test_unknown_model_is_404(build_client):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": 'model "ghost" not found'})

    client = build_client(handler)
    response = client.post(
        "/v1/chat/completions",
        headers=auth(),
        json={"model": "ghost", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 404
    assert_envelope(response.json(), "invalid_request_error", "model_not_found")


def test_unknown_route_is_envelope_too(build_client):
    client = build_client(ok_handler)
    response = client.get("/v1/nope", headers=auth())
    assert response.status_code == 404
    assert set(response.json()) == {"error"}


def test_upstream_500_is_502(build_client):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    client = build_client(handler)
    response = client.post(
        "/v1/chat/completions",
        headers=auth(),
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 502
    assert_envelope(response.json(), "upstream_error", "ollama_unavailable")


@pytest.mark.parametrize(
    "exc_name, status, code",
    [
        ("InvalidRequestError", 400, "invalid_request"),
        ("AuthenticationError", 401, "invalid_api_key"),
        ("KeyRevokedError", 403, "key_revoked"),
        ("ModelNotFoundError", 404, "model_not_found"),
        ("RateLimitError", 429, "rate_limit_exceeded"),
        ("OllamaUnavailableError", 502, "ollama_unavailable"),
        ("OllamaTimeoutError", 504, "ollama_timeout"),
    ],
)
def test_error_table_matches_contract(exc_name, status, code):
    import app.errors as errors

    exc = getattr(errors, exc_name)("boom")
    assert exc.status_code == status
    assert exc.code == code
    body = exc.to_dict()["error"]
    assert set(body) == ENVELOPE_KEYS
    assert exc.to_sse().startswith("data: {")
    assert exc.to_sse().endswith("\n\n")


def test_non_json_upstream_body_is_502(build_client):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json at all")

    client = build_client(handler)
    response = client.post(
        "/v1/chat/completions",
        headers=auth(),
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 502
    assert_envelope(response.json(), "upstream_error", "ollama_unavailable")
