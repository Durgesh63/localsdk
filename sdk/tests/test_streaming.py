"""SSE parsing: multi-line events, comments, [DONE], and mid-stream errors."""

from __future__ import annotations

import json

import httpx
import pytest

from localsdk import APIError, UpstreamError
from localsdk._transport import SSEDecoder, parse_sse_payload, retry_delay

from conftest import make_async_client, make_client, sse, sse_chunk

MESSAGES = [{"role": "user", "content": "hi"}]


def stream_handler(body: str, *, record: list[httpx.Request] | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if record is not None:
            record.append(request)
        return httpx.Response(
            200, text=body, headers={"content-type": "text/event-stream"}
        )

    return handler


# --------------------------------------------------------------------------- #
# the decoder itself
# --------------------------------------------------------------------------- #


def test_decoder_emits_on_blank_line_separator():
    decoder = SSEDecoder()
    assert decoder.feed("data: {\"a\": 1}") is None
    assert decoder.feed("") == '{"a": 1}'


def test_decoder_joins_multi_line_events():
    decoder = SSEDecoder()
    decoder.feed("data: line one")
    decoder.feed("data: line two")
    assert decoder.feed("") == "line one\nline two"


def test_decoder_ignores_comments_and_other_fields():
    decoder = SSEDecoder()
    assert decoder.feed(": keep-alive") is None
    assert decoder.feed("event: message") is None
    assert decoder.feed("id: 7") is None
    decoder.feed("data: payload")
    assert decoder.feed("") == "payload"


def test_decoder_handles_crlf_and_split_chunks():
    decoder = SSEDecoder()
    payloads = decoder.feed_text('data: {"a"')
    assert payloads == []
    payloads = decoder.feed_text(': 1}\r\n\r\ndata: [DO')
    assert payloads == ['{"a": 1}']
    payloads = decoder.feed_text("NE]\n\n")
    assert payloads == ["[DONE]"]


def test_done_payload_parses_to_none():
    assert parse_sse_payload("[DONE]") is None


def test_streaming_is_never_retried():
    assert retry_delay(attempt=0, max_retries=5, status_code=502, stream=True) is None


# --------------------------------------------------------------------------- #
# end to end through the client
# --------------------------------------------------------------------------- #


def test_stream_yields_chunks_and_stops_at_done():
    body = (
        ": ngrok keep-alive\n\n"
        + sse(
            json.dumps(
                {
                    "id": "chatcmpl-x",
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                }
            ),
            sse_chunk("Hel"),
            sse_chunk("lo"),
            sse_chunk(finish_reason="stop"),
            "[DONE]",
        )
        + sse(sse_chunk("never delivered"))
    )
    client = make_client(stream_handler(body))
    try:
        chunks = list(client.stream(MESSAGES))
    finally:
        client.close()

    assert "".join(chunk.text for chunk in chunks) == "Hello"
    assert chunks[-1].finish_reason == "stop"
    assert chunks[0].raw["object"] == "chat.completion.chunk"


def test_stream_sets_stream_true_in_the_payload():
    seen: list[httpx.Request] = []
    client = make_client(stream_handler(sse(sse_chunk("x"), "[DONE]"), record=seen))
    try:
        list(client.stream(MESSAGES, temperature=0.1))
    finally:
        client.close()

    payload = json.loads(seen[0].content)
    assert payload["stream"] is True
    assert payload["temperature"] == 0.1
    assert seen[0].headers["accept"] == "text/event-stream"


def test_mid_stream_error_event_raises_the_mapped_exception():
    body = sse(
        sse_chunk("partial"),
        json.dumps(
            {
                "error": {
                    "message": "ollama went away",
                    "type": "upstream_error",
                    "param": None,
                    "code": "ollama_unavailable",
                }
            }
        ),
        "[DONE]",
    )
    client = make_client(stream_handler(body))
    collected = []
    try:
        with pytest.raises(UpstreamError) as excinfo:
            for chunk in client.stream(MESSAGES):
                collected.append(chunk.text)
    finally:
        client.close()

    assert collected == ["partial"]
    assert "ollama went away" in str(excinfo.value)
    assert excinfo.value.code == "ollama_unavailable"
    assert excinfo.value.status_code == 502


def test_error_status_before_the_stream_starts_is_mapped():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404,
            json={
                "error": {
                    "message": "model not found",
                    "type": "invalid_request_error",
                    "code": "model_not_found",
                }
            },
        )

    client = make_client(handler)
    try:
        with pytest.raises(APIError) as excinfo:
            list(client.stream(MESSAGES))
    finally:
        client.close()

    assert excinfo.value.status_code == 404
    assert excinfo.value.code == "model_not_found"


def test_stream_without_trailing_done_still_yields_everything():
    body = sse(sse_chunk("a"), sse_chunk("b"))
    client = make_client(stream_handler(body))
    try:
        assert "".join(c.text for c in client.stream(MESSAGES)) == "ab"
    finally:
        client.close()


async def test_async_stream_yields_chunks():
    body = sse(sse_chunk("as"), sse_chunk("ync"), "[DONE]")
    async with make_async_client(stream_handler(body)) as client:
        out = [chunk.text async for chunk in client.stream(MESSAGES)]
    assert "".join(out) == "async"


async def test_async_stream_raises_mid_stream_error():
    body = sse(
        sse_chunk("x"),
        json.dumps({"error": {"message": "boom", "type": "upstream_error", "code": "ollama_timeout"}}),
        "[DONE]",
    )
    async with make_async_client(stream_handler(body)) as client:
        with pytest.raises(APIError) as excinfo:
            async for _chunk in client.stream(MESSAGES):
                pass
    assert excinfo.value.status_code == 504
