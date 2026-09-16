"""FastAPI dependencies: settings, key store, Ollama client, bearer auth."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from app.auth.keystore import KeyRecord, KeyStore
from app.config import Settings
from app.errors import AuthenticationError, KeyRevokedError
from app.ollama.client import OllamaClient


def get_settings(request: Request) -> Settings:
    """Settings resolved once at app start and stashed on ``app.state``."""
    return request.app.state.settings


def get_keystore(request: Request) -> KeyStore:
    return request.app.state.keystore


def get_ollama(request: Request) -> OllamaClient:
    return request.app.state.ollama


def _bearer_token(request: Request) -> str:
    """Extract the bearer token, raising 401 when absent or malformed.

    Input:  header "Authorization: Bearer sk-x"
    Output: "sk-x"
    """
    header = request.headers.get("authorization") or ""
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise AuthenticationError(
            "Missing bearer token. Send 'Authorization: Bearer <api_key>'."
        )
    return token.strip()


def require_api_key(
    request: Request,
    keystore: Annotated[KeyStore, Depends(get_keystore)],
) -> KeyRecord:
    """Auth dependency for every ``/v1/*`` route (never applied to ``/healthz``)."""
    token = _bearer_token(request)
    record = keystore.validate(token)
    if record is None:
        raise AuthenticationError("Incorrect API key provided.")
    if record.revoked:
        raise KeyRevokedError("This API key has been revoked.")
    request.state.api_key = record
    return record


ApiKey = Annotated[KeyRecord, Depends(require_api_key)]
Ollama = Annotated[OllamaClient, Depends(get_ollama)]
Config = Annotated[Settings, Depends(get_settings)]
