"""Bearer auth on /v1/*, and the absence of auth on /healthz."""

from __future__ import annotations

import httpx
import pytest

from app.auth.keystore import PostgresKeyStore, StaticKeyStore, build_keystore
from conftest import API_KEY, BAD_KEY, MODEL, auth, make_settings


def tags_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"models": [{"model": MODEL}]})


def test_missing_key_is_401(build_client):
    client = build_client(tags_handler)
    response = client.get("/v1/models")
    assert response.status_code == 401
    error = response.json()["error"]
    assert error["type"] == "authentication_error"
    assert error["code"] == "invalid_api_key"
    assert error["param"] is None


def test_bad_key_is_401(build_client):
    client = build_client(tags_handler)
    response = client.get("/v1/models", headers=auth(BAD_KEY))
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "invalid_api_key"


def test_malformed_authorization_header_is_401(build_client):
    client = build_client(tags_handler)
    response = client.get("/v1/models", headers={"Authorization": API_KEY})
    assert response.status_code == 401


def test_good_key_is_200(build_client):
    client = build_client(tags_handler)
    response = client.get("/v1/models", headers=auth())
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert body["data"][0] == {
        "id": MODEL,
        "object": "model",
        "created": body["data"][0]["created"],
        "owned_by": "ollama",
    }


def test_healthz_needs_no_auth_and_reports_ollama_up(build_client):
    client = build_client(tags_handler)
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "ollama": "up",
        "model": MODEL,
        "version": "0.1.0",
    }


def test_healthz_is_200_even_when_ollama_is_down(build_client):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = build_client(handler)
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["ollama"] == "down"


def test_revoked_key_is_403(build_client):
    from app.main import create_app
    from fastapi.testclient import TestClient

    store = StaticKeyStore(keys=[API_KEY], revoked_keys=["sk-revoked"])
    app = create_app(
        make_settings(), transport=httpx.MockTransport(tags_handler), keystore=store
    )
    with TestClient(app) as client:
        response = client.get("/v1/models", headers=auth("sk-revoked"))
    assert response.status_code == 403
    error = response.json()["error"]
    assert error["type"] == "permission_error"
    assert error["code"] == "key_revoked"


def test_static_keystore_round_trip():
    store = StaticKeyStore(keys=["sk-a", " sk-b "])
    assert store.validate("sk-a").key == "sk-a"
    assert store.validate("sk-b").revoked is False
    assert store.validate("sk-missing") is None
    assert store.revoked("sk-a") is False


def test_postgres_keystore_is_a_loud_stub():
    store = PostgresKeyStore("postgresql://x")
    with pytest.raises(NotImplementedError, match="stub"):
        store.validate("sk-a")
    with pytest.raises(NotImplementedError):
        store.revoked("sk-a")


def test_build_keystore_selects_backend():
    assert isinstance(build_keystore(make_settings()), StaticKeyStore)
    assert isinstance(
        build_keystore(make_settings(keystore_backend="postgres")), PostgresKeyStore
    )
    with pytest.raises(ValueError):
        build_keystore(make_settings(keystore_backend="redis"))
