"""Streaming /v1/chat/completions: SSE chunk sequence, headers, mid-stream errors."""

from __future__ import annotations

import json

import httpx

from conftest import MODEL, auth, ndjson_stream, sse_events


def stream_handler(lines: list[dict], captured: dict | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if captured is not None:
            captured["request"] = json.loads(request.content)
        return httpx.Response(
            200,
            headers={"Content-Type": "application/x-ndjson"},
            content=ndjson_stream(lines),
        )

    return handler


TEXT_LINES = [
    {"model": MODEL, "message": {"role": "assistant", "content": "Hel"}, "done": False},
    {"model": MODEL, "message": {"role": "assistant", "content": "lo"}, "done": False},
    {
        "model": MODEL,
        "message": {"role": "assistant", "content": ""},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 5,
        "eval_count": 2,
    },
]


def collect(client, body: dict) -> tuple[httpx.Response, list[str]]:
    with client.stream(
        "POST", "/v1/chat/completions", headers=auth(), json=body
    ) as response:
        raw = "".join(response.iter_text())
        return response, sse_events(raw)


def test_stream_chunk_sequence_ends_in_done(build_client):
    captured: dict = {}
    client = build_client(stream_handler(TEXT_LINES, captured))
    response, events = collect(
        client, {"messages": [{"role": "user", "content": "hi"}], "stream": True}
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"
    assert response.headers["connection"] == "keep-alive"

    assert captured["request"]["stream"] is True
    assert events[-1] == "[DONE]"

    chunks = [json.loads(event) for event in events[:-1]]
    ids = {chunk["id"] for chunk in chunks}
    assert len(ids) == 1
    assert next(iter(ids)).startswith("chatcmpl-")
    assert all(chunk["object"] == "chat.completion.chunk" for chunk in chunks)
    assert all(chunk["model"] == MODEL for chunk in chunks)

    # first chunk: role only
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert chunks[0]["choices"][0]["finish_reason"] is None

    # middle chunks: content deltas in order
    assert [chunk["choices"][0]["delta"].get("content") for chunk in chunks[1:-1]] == [
        "Hel",
        "lo",
    ]

    # last chunk: empty delta plus finish_reason
    last = chunks[-1]["choices"][0]
    assert last["delta"] == {}
    assert last["finish_reason"] == "stop"


def test_stream_raw_body_uses_blank_line_separated_events(build_client):
    client = build_client(stream_handler(TEXT_LINES))
    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers=auth(),
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
    ) as response:
        raw = "".join(response.iter_text())
    assert raw.endswith("data: [DONE]\n\n")
    assert "\n\ndata: " in raw


def test_stream_tool_calls(build_client):
    lines = [
        {
            "model": MODEL,
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "get_weather", "arguments": {"city": "Pune"}}}
                ],
            },
            "done": False,
        },
        {"model": MODEL, "message": {"role": "assistant", "content": ""}, "done": True},
    ]
    client = build_client(stream_handler(lines))
    _, events = collect(
        client,
        {
            "messages": [{"role": "user", "content": "weather?"}],
            "stream": True,
            "tools": [
                {"type": "function", "function": {"name": "get_weather"}},
            ],
        },
    )
    chunks = [json.loads(event) for event in events[:-1]]
    tool_delta = chunks[1]["choices"][0]["delta"]["tool_calls"][0]

    assert tool_delta["index"] == 0
    assert tool_delta["id"].startswith("call_")
    assert tool_delta["function"]["name"] == "get_weather"
    assert isinstance(tool_delta["function"]["arguments"], str)
    assert json.loads(tool_delta["function"]["arguments"]) == {"city": "Pune"}

    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_stream_ollama_down_is_502_before_any_bytes(build_client):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = build_client(handler)
    response = client.post(
        "/v1/chat/completions",
        headers=auth(),
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "ollama_unavailable"


def test_mid_stream_failure_emits_error_event_then_done(build_client):
    async def broken_body():
        payload = {
            "model": MODEL,
            "message": {"role": "assistant", "content": "Hel"},
            "done": False,
        }
        yield (json.dumps(payload) + "\n").encode()
        raise httpx.ReadTimeout("ollama went away")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=broken_body())

    client = build_client(handler)
    _, events = collect(
        client, {"messages": [{"role": "user", "content": "hi"}], "stream": True}
    )

    assert events[-1] == "[DONE]"
    error = json.loads(events[-2])["error"]
    assert error["type"] == "upstream_error"
    assert error["code"] == "ollama_timeout"


def test_stream_upstream_http_error_is_mapped(build_client):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": 'model "ghost" not found'})

    client = build_client(handler)
    response = client.post(
        "/v1/chat/completions",
        headers=auth(),
        json={
            "model": "ghost",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_found"
