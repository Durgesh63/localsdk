"""/v1/embeddings shape and input handling."""

from __future__ import annotations

import json

import httpx

from conftest import EMBED_MODEL, auth

CAPTURED: dict = {}


def embed_handler(payload: dict, status: int = 200):
    def handler(request: httpx.Request) -> httpx.Response:
        CAPTURED["request"] = json.loads(request.content)
        CAPTURED["url"] = str(request.url)
        return httpx.Response(status, json=payload)

    return handler


def test_embeddings_single_input(build_client):
    client = build_client(
        embed_handler(
            {
                "model": EMBED_MODEL,
                "embeddings": [[0.1, 0.2]],
                "prompt_eval_count": 4,
            }
        )
    )
    response = client.post(
        "/v1/embeddings", headers=auth(), json={"input": "hello world"}
    )
    assert response.status_code == 200
    assert response.json() == {
        "object": "list",
        "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
        "model": EMBED_MODEL,
        "usage": {"prompt_tokens": 4, "total_tokens": 4},
    }
    assert CAPTURED["url"].endswith("/api/embed")
    assert CAPTURED["request"] == {"model": EMBED_MODEL, "input": "hello world"}


def test_embeddings_batch_input_is_indexed(build_client):
    client = build_client(
        embed_handler({"embeddings": [[0.1], [0.2], [0.3]]})
    )
    response = client.post(
        "/v1/embeddings", headers=auth(), json={"input": ["a", "b", "c"]}
    )
    data = response.json()["data"]
    assert [item["index"] for item in data] == [0, 1, 2]
    assert [item["embedding"] for item in data] == [[0.1], [0.2], [0.3]]


def test_embeddings_legacy_embedding_key(build_client):
    client = build_client(embed_handler({"embedding": [0.5, 0.6]}))
    response = client.post("/v1/embeddings", headers=auth(), json={"input": "x"})
    assert response.json()["data"] == [
        {"object": "embedding", "index": 0, "embedding": [0.5, 0.6]}
    ]


def test_embeddings_requires_auth(build_client):
    client = build_client(embed_handler({"embeddings": [[0.1]]}))
    assert client.post("/v1/embeddings", json={"input": "x"}).status_code == 401


def test_embeddings_empty_input_is_400(build_client):
    client = build_client(embed_handler({"embeddings": []}))
    response = client.post("/v1/embeddings", headers=auth(), json={"input": []})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"
