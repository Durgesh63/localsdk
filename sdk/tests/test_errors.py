"""Status mapping, the HTML interstitial, transport failures, and retry policy."""

from __future__ import annotations

import httpx
import pytest

from localsdk import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UpstreamError,
)
from localsdk.errors import APIError

from conftest import chat_body, error_body, json_handler, make_async_client, make_client, text_handler

MESSAGES = [{"role": "user", "content": "hi"}]

CASES = [
    (400, "invalid_request_error", "invalid_request", BadRequestError),
    (401, "authentication_error", "invalid_api_key", AuthenticationError),
    (403, "permission_error", "key_revoked", PermissionDeniedError),
    (404, "invalid_request_error", "model_not_found", NotFoundError),
    (429, "rate_limit_error", "rate_limit_exceeded", RateLimitError),
    (502, "upstream_error", "ollama_unavailable", UpstreamError),
    (504, "upstream_error", "ollama_timeout", UpstreamError),
]


@pytest.mark.parametrize("status,etype,code,expected", CASES)
def test_status_codes_map_to_exceptions(status, etype, code, expected):
    body = error_body("something went wrong", type=etype, code=code)
    client = make_client(json_handler(body, status_code=status))
    try:
        with pytest.raises(expected) as excinfo:
            client.chat(MESSAGES)
    finally:
        client.close()

    error = excinfo.value
    assert isinstance(error, APIError)
    assert error.status_code == status
    assert error.code == code
    assert error.type == etype
    assert error.message == "something went wrong"
    assert "something went wrong" in str(error)


def test_unmapped_status_falls_back_to_api_error():
    client = make_client(json_handler({"error": {"message": "teapot"}}, status_code=418))
    try:
        with pytest.raises(APIError) as excinfo:
            client.chat(MESSAGES)
    finally:
        client.close()
    assert type(excinfo.value) is APIError
    assert excinfo.value.status_code == 418


def test_non_json_error_body_still_produces_a_message():
    client = make_client(text_handler("gateway exploded", status_code=502))
    try:
        with pytest.raises(UpstreamError) as excinfo:
            client.chat(MESSAGES)
    finally:
        client.close()
    assert "gateway exploded" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# the ngrok interstitial -- the single most likely real-world failure
# --------------------------------------------------------------------------- #

INTERSTITIAL = "<!DOCTYPE html><html><head><title>ngrok</title></head><body>You are about to visit...</body></html>"


def test_html_response_is_reported_as_a_connection_problem():
    client = make_client(text_handler(INTERSTITIAL, content_type="text/html; charset=utf-8"))
    try:
        with pytest.raises(APIConnectionError) as excinfo:
            client.chat(MESSAGES)
    finally:
        client.close()

    message = str(excinfo.value)
    assert "HTML" in message
    assert "ngrok" in message
    assert "ngrok-skip-browser-warning" in message
    assert "LOCALSDK_BASE_URL" in message
    assert "base_url" in message


def test_html_wins_over_the_status_code():
    """A dead tunnel answers 404 with HTML; 'not found' would be misleading."""
    client = make_client(
        text_handler(INTERSTITIAL, status_code=404, content_type="text/html")
    )
    try:
        with pytest.raises(APIConnectionError):
            client.chat(MESSAGES)
    finally:
        client.close()


def test_html_body_without_the_content_type_is_still_caught():
    client = make_client(text_handler(INTERSTITIAL, content_type="text/plain"))
    try:
        with pytest.raises(APIConnectionError) as excinfo:
            client.chat(MESSAGES)
    finally:
        client.close()
    assert "ngrok" in str(excinfo.value)


def test_html_during_streaming_is_caught_too():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=INTERSTITIAL, headers={"content-type": "text/html"})

    client = make_client(handler)
    try:
        with pytest.raises(APIConnectionError):
            list(client.stream(MESSAGES))
    finally:
        client.close()


def test_non_json_success_body_is_a_connection_error():
    client = make_client(text_handler("not json at all"))
    try:
        with pytest.raises(APIConnectionError):
            client.chat(MESSAGES)
    finally:
        client.close()


# --------------------------------------------------------------------------- #
# transport failures
# --------------------------------------------------------------------------- #


def test_connect_failure_mentions_the_rotating_ngrok_url():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("nodename nor servname provided", request=request)

    client = make_client(handler)
    try:
        with pytest.raises(APIConnectionError) as excinfo:
            client.chat(MESSAGES)
    finally:
        client.close()
    assert "ngrok" in str(excinfo.value)


def test_timeout_is_its_own_error_with_advice():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    client = make_client(handler, timeout=5)
    try:
        with pytest.raises(APITimeoutError) as excinfo:
            client.chat(MESSAGES)
    finally:
        client.close()
    assert "stream()" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# retries: off unless asked for, and only where it is safe
# --------------------------------------------------------------------------- #


def counting_handler(status_code: int, counter: list[int]):
    def handler(_request: httpx.Request) -> httpx.Response:
        counter.append(1)
        return httpx.Response(status_code, json=error_body("upstream", type="upstream_error", code="x"))

    return handler


def test_retries_are_off_by_default():
    calls: list[int] = []
    client = make_client(counting_handler(502, calls))
    try:
        with pytest.raises(UpstreamError):
            client.chat(MESSAGES)
    finally:
        client.close()
    assert len(calls) == 1, "a metered request must never be re-sent implicitly"


def test_enabled_retries_recover_from_502():
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(502, json=error_body("down", type="upstream_error", code="x"))
        return httpx.Response(200, json=chat_body("recovered"))

    client = make_client(handler, max_retries=2)
    try:
        assert client.chat(MESSAGES).text == "recovered"
    finally:
        client.close()
    assert len(calls) == 3


def test_retries_give_up_after_the_budget():
    calls: list[int] = []
    client = make_client(counting_handler(504, calls), max_retries=2)
    try:
        with pytest.raises(UpstreamError):
            client.chat(MESSAGES)
    finally:
        client.close()
    assert len(calls) == 3


def test_client_errors_are_never_retried():
    calls: list[int] = []
    client = make_client(counting_handler(400, calls), max_retries=3)
    try:
        with pytest.raises(BadRequestError):
            client.chat(MESSAGES)
    finally:
        client.close()
    assert len(calls) == 1


def test_connect_errors_are_retried_when_enabled():
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ConnectError("boom", request=request)
        return httpx.Response(200, json=chat_body("second try"))

    client = make_client(handler, max_retries=1)
    try:
        assert client.chat(MESSAGES).text == "second try"
    finally:
        client.close()
    assert len(calls) == 2


def test_read_timeouts_are_not_retried_even_when_retries_are_enabled():
    """The generation already started; retrying doubles the spend."""
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        raise httpx.ReadTimeout("slow", request=request)

    client = make_client(handler, max_retries=3)
    try:
        with pytest.raises(APITimeoutError):
            client.chat(MESSAGES)
    finally:
        client.close()
    assert len(calls) == 1


async def test_async_errors_map_the_same_way():
    body = error_body("bad key", type="authentication_error", code="invalid_api_key")
    async with make_async_client(json_handler(body, status_code=401)) as client:
        with pytest.raises(AuthenticationError) as excinfo:
            await client.chat(MESSAGES)
    assert excinfo.value.code == "invalid_api_key"


async def test_async_html_interstitial():
    async with make_async_client(
        text_handler(INTERSTITIAL, content_type="text/html")
    ) as client:
        with pytest.raises(APIConnectionError):
            await client.chat(MESSAGES)
