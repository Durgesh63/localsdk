"""Non-streaming /v1/chat/completions: shape, translation, tool calls."""

from __future__ import annotations

import json

import httpx

from conftest import MODEL, auth

CAPTURED: dict[str, dict] = {}


def chat_handler(payload: dict) -> httpx.Response:
    def handler(request: httpx.Request) -> httpx.Response:
        CAPTURED["request"] = json.loads(request.content)
        CAPTURED["url"] = str(request.url)
        return httpx.Response(200, json=payload)

    return handler


PLAIN_REPLY = {
    "model": MODEL,
    "created_at": "2024-01-01T00:00:00Z",
    "message": {"role": "assistant", "content": "Hello there."},
    "done": True,
    "done_reason": "stop",
    "prompt_eval_count": 11,
    "eval_count": 3,
}

TOOL_REPLY = {
    "model": MODEL,
    "message": {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"function": {"name": "get_weather", "arguments": {"city": "Pune"}}}
        ],
    },
    "done": True,
    "done_reason": "stop",
}


def test_non_stream_response_shape(build_client):
    client = build_client(chat_handler(PLAIN_REPLY))
    response = client.post(
        "/v1/chat/completions",
        headers=auth(),
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    body = response.json()

    assert body["id"].startswith("chatcmpl-")
    assert body["object"] == "chat.completion"
    assert body["model"] == MODEL
    assert isinstance(body["created"], int)
    assert body["usage"] == {
        "prompt_tokens": 11,
        "completion_tokens": 3,
        "total_tokens": 14,
    }

    choice = body["choices"][0]
    assert choice["index"] == 0
    assert choice["finish_reason"] == "stop"
    assert choice["message"] == {
        "role": "assistant",
        "content": "Hello there.",
        "tool_calls": None,
    }

    # default model applied, and it went to /api/chat with stream disabled
    assert CAPTURED["request"]["model"] == MODEL
    assert CAPTURED["request"]["stream"] is False
    assert CAPTURED["url"].endswith("/api/chat")


def test_unknown_fields_are_ignored_not_rejected(build_client):
    client = build_client(chat_handler(PLAIN_REPLY))
    response = client.post(
        "/v1/chat/completions",
        headers=auth(),
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "frequency_penalty": 0.4,
            "user": "abc",
            "logit_bias": {"1": 2},
        },
    )
    assert response.status_code == 200


def test_sampling_params_map_to_ollama_options(build_client):
    client = build_client(chat_handler(PLAIN_REPLY))
    client.post(
        "/v1/chat/completions",
        headers=auth(),
        json={
            "model": "custom:7b",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.3,
            "top_p": 0.8,
            "max_tokens": 64,
            "seed": 7,
            "stop": "END",
        },
    )
    sent = CAPTURED["request"]
    assert sent["model"] == "custom:7b"
    assert sent["options"] == {
        "temperature": 0.3,
        "top_p": 0.8,
        "num_predict": 64,
        "seed": 7,
        "stop": ["END"],
    }


def test_tool_calls_arguments_is_a_json_string(build_client):
    client = build_client(chat_handler(TOOL_REPLY))
    response = client.post(
        "/v1/chat/completions",
        headers=auth(),
        json={
            "messages": [{"role": "user", "content": "weather in Pune?"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            "tool_choice": "auto",
        },
    )
    assert response.status_code == 200
    choice = response.json()["choices"][0]

    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None

    call = choice["message"]["tool_calls"][0]
    assert call["id"].startswith("call_")
    assert call["type"] == "function"
    assert call["function"]["name"] == "get_weather"
    # The critical detail: a JSON-encoded *string*, not an object.
    assert isinstance(call["function"]["arguments"], str)
    assert json.loads(call["function"]["arguments"]) == {"city": "Pune"}

    # tools were forwarded to ollama
    assert CAPTURED["request"]["tools"][0]["function"]["name"] == "get_weather"


def test_tool_result_message_round_trips_to_ollama(build_client):
    client = build_client(chat_handler(PLAIN_REPLY))
    response = client.post(
        "/v1/chat/completions",
        headers=auth(),
        json={
            "messages": [
                {"role": "user", "content": "weather in Pune?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "Pune"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_abc", "content": "31C"},
            ],
        },
    )
    assert response.status_code == 200
    messages = CAPTURED["request"]["messages"]

    # assistant tool_calls arguments become an object again for ollama
    assert messages[1]["tool_calls"][0]["function"]["arguments"] == {"city": "Pune"}
    # tool result keeps its id and gains the resolved tool name
    assert messages[2]["role"] == "tool"
    assert messages[2]["content"] == "31C"
    assert messages[2]["tool_call_id"] == "call_abc"
    assert messages[2]["tool_name"] == "get_weather"


def test_response_format_json_object_maps_to_format_json(build_client):
    client = build_client(chat_handler(PLAIN_REPLY))
    client.post(
        "/v1/chat/completions",
        headers=auth(),
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {"type": "json_object"},
        },
    )
    assert CAPTURED["request"]["format"] == "json"


def test_response_format_json_schema_maps_to_raw_schema(build_client):
    schema = {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    }
    client = build_client(chat_handler(PLAIN_REPLY))
    client.post(
        "/v1/chat/completions",
        headers=auth(),
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "City", "schema": schema, "strict": True},
            },
        },
    )
    assert CAPTURED["request"]["format"] == schema


def test_length_finish_reason(build_client):
    reply = {**PLAIN_REPLY, "done_reason": "length"}
    client = build_client(chat_handler(reply))
    response = client.post(
        "/v1/chat/completions",
        headers=auth(),
        json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
    )
    assert response.json()["choices"][0]["finish_reason"] == "length"
