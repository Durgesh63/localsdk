"""Chat happy path, headers, payload shaping, and the other simple endpoints."""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic import BaseModel

from localsdk import Message, StructuredOutputError
from localsdk._transport import NGROK_HEADER
from localsdk._version import USER_AGENT

from conftest import (
    API_KEY,
    BASE_URL,
    chat_body,
    json_handler,
    make_async_client,
    make_client,
)

MESSAGES = [{"role": "user", "content": "hi"}]


def test_chat_returns_conveniences():
    client = make_client(json_handler(chat_body("Hello there.")))
    try:
        response = client.chat(MESSAGES)
    finally:
        client.close()

    assert response.text == "Hello there."
    assert response.finish_reason == "stop"
    assert response.tool_calls == []
    assert response.usage.total_tokens == 18
    assert response.model == "qwen2.5:14b"
    assert response.raw["choices"][0]["message"]["content"] == "Hello there."


def test_every_request_carries_the_contract_headers():
    seen: list[httpx.Request] = []
    client = make_client(json_handler(chat_body(), record=seen))
    try:
        client.chat(MESSAGES)
    finally:
        client.close()

    request = seen[0]
    assert request.headers[NGROK_HEADER] == "true"
    assert request.headers["authorization"] == "Bearer {0}".format(API_KEY)
    assert request.headers["user-agent"] == USER_AGENT
    assert USER_AGENT.startswith("localsdk-python/")
    assert str(request.url) == BASE_URL + "/v1/chat/completions"


def test_health_is_unauthenticated_but_still_skips_the_ngrok_warning():
    seen: list[httpx.Request] = []
    body = {"status": "ok", "ollama": "up", "model": "qwen2.5:14b", "version": "0.1.0"}
    client = make_client(json_handler(body, record=seen))
    try:
        health = client.health()
    finally:
        client.close()

    assert health.ok is True
    assert str(seen[0].url) == BASE_URL + "/healthz"
    assert "authorization" not in seen[0].headers
    assert seen[0].headers[NGROK_HEADER] == "true"


def test_health_reports_ollama_down_without_raising():
    client = make_client(json_handler({"status": "ok", "ollama": "down"}))
    try:
        health = client.health()
    finally:
        client.close()
    assert health.ok is False
    assert health.ollama == "down"


def test_payload_omits_unset_options_and_passes_extras_through():
    seen: list[httpx.Request] = []
    client = make_client(json_handler(chat_body(), record=seen))
    try:
        client.chat(MESSAGES, temperature=0.2, top_p=0.9, stop=["\n\n"], seed=7)
    finally:
        client.close()

    payload = json.loads(seen[0].content)
    assert payload["model"] == "qwen2.5:14b"
    assert payload["stream"] is False
    assert payload["temperature"] == 0.2
    assert payload["top_p"] == 0.9
    assert payload["stop"] == ["\n\n"]
    assert payload["seed"] == 7
    assert "max_tokens" not in payload
    assert "tools" not in payload
    assert "response_format" not in payload


def test_per_call_model_overrides_the_client_default():
    seen: list[httpx.Request] = []
    client = make_client(json_handler(chat_body(), record=seen), model="default-model")
    try:
        client.chat(MESSAGES)
        client.chat(MESSAGES, model="other-model")
    finally:
        client.close()

    assert json.loads(seen[0].content)["model"] == "default-model"
    assert json.loads(seen[1].content)["model"] == "other-model"


def test_message_objects_are_accepted_alongside_dicts():
    seen: list[httpx.Request] = []
    client = make_client(json_handler(chat_body(), record=seen))
    try:
        client.chat([Message(role="system", content="be terse"), {"role": "user", "content": "hi"}])
    finally:
        client.close()

    payload = json.loads(seen[0].content)
    assert payload["messages"] == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hi"},
    ]


def test_bare_string_messages_is_a_type_error():
    client = make_client(json_handler(chat_body()))
    try:
        with pytest.raises(TypeError):
            client.chat("hello")
    finally:
        client.close()


def test_models_and_embeddings():
    models_body = {
        "object": "list",
        "data": [
            {"id": "qwen2.5:14b", "object": "model", "created": 1, "owned_by": "ollama"}
        ],
    }
    embed_body = {
        "object": "list",
        "data": [
            {"object": "embedding", "index": 1, "embedding": [0.3, 0.4]},
            {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]},
        ],
        "model": "nomic-embed-text",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json=models_body)
        return httpx.Response(200, json=embed_body)

    client = make_client(handler)
    try:
        models = client.models()
        vectors = client.embed(["a", "b"])
    finally:
        client.close()

    assert [m.id for m in models] == ["qwen2.5:14b"]
    assert models[0].owned_by == "ollama"
    # sorted back into request order by "index"
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]


def test_embed_accepts_a_single_string():
    seen: list[httpx.Request] = []
    body = {"data": [{"object": "embedding", "index": 0, "embedding": [0.5]}]}
    client = make_client(json_handler(body, record=seen))
    try:
        vectors = client.embed("just one")
    finally:
        client.close()

    assert json.loads(seen[0].content)["input"] == ["just one"]
    assert vectors == [[0.5]]


def test_context_manager_closes_the_pool():
    with make_client(json_handler(chat_body())) as client:
        assert client.chat(MESSAGES).text == "Hello there."
    assert client._http.is_closed


async def test_async_chat_matches_sync():
    async with make_async_client(json_handler(chat_body("async hi"))) as client:
        response = await client.chat(MESSAGES)
    assert response.text == "async hi"


# --------------------------------------------------------------------------- #
# structured output
# --------------------------------------------------------------------------- #


class Person(BaseModel):
    """A person."""

    name: str
    age: int


def test_structured_sends_the_json_schema_response_format():
    seen: list[httpx.Request] = []
    body = chat_body('{"name": "Ada", "age": 36}')
    client = make_client(json_handler(body, record=seen))
    try:
        person = client.structured(MESSAGES, Person)
    finally:
        client.close()

    assert isinstance(person, Person)
    assert person.name == "Ada" and person.age == 36

    response_format = json.loads(seen[0].content)["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "Person"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"] == Person.model_json_schema()


def test_structured_tolerates_a_markdown_code_fence():
    body = chat_body('```json\n{"name": "Ada", "age": 36}\n```')
    client = make_client(json_handler(body))
    try:
        assert client.structured(MESSAGES, Person).name == "Ada"
    finally:
        client.close()


def test_structured_raises_on_unparseable_json():
    client = make_client(json_handler(chat_body("sorry, I cannot do that")))
    try:
        with pytest.raises(StructuredOutputError) as excinfo:
            client.structured(MESSAGES, Person)
    finally:
        client.close()
    assert "valid JSON" in str(excinfo.value)
    assert excinfo.value.text == "sorry, I cannot do that"


def test_structured_raises_when_the_json_does_not_fit_the_schema():
    client = make_client(json_handler(chat_body('{"name": "Ada"}')))
    try:
        with pytest.raises(StructuredOutputError) as excinfo:
            client.structured(MESSAGES, Person)
    finally:
        client.close()
    assert "Person" in str(excinfo.value)


def test_structured_rejects_a_non_model_schema():
    client = make_client(json_handler(chat_body("{}")))
    try:
        with pytest.raises(TypeError):
            client.structured(MESSAGES, dict)
    finally:
        client.close()


async def test_async_structured():
    async with make_async_client(json_handler(chat_body('{"name": "Bo", "age": 9}'))) as client:
        person = await client.structured(MESSAGES, Person)
    assert person.age == 9


async def test_async_models_embed_health():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/healthz":
            return httpx.Response(200, json={"status": "ok", "ollama": "up"})
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "m"}]})
        return httpx.Response(
            200, json={"data": [{"index": 0, "embedding": [1.0]}]}
        )

    client = make_async_client(handler)
    try:
        assert (await client.health()).ok
        assert [m.id for m in await client.models()] == ["m"]
        assert await client.embed("x") == [[1.0]]
    finally:
        await client.close()
