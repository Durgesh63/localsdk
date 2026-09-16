"""FastAPI dependencies: settings, key store, Ollama client, bearer auth."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.auth.keystore import KeyRecord, KeyStore
from app.config import Settings
from app.errors import AuthenticationError, KeyRevokedError
from app.ollama.client import OllamaClient

#: Declaring the scheme is what advertises bearer auth in the OpenAPI document,
#: which is what puts the "Authorize" button in Swagger UI at /docs.
#:
#: ``auto_error=False`` matters: left on, FastAPI would raise its own 401 with a
#: ``{"detail": ...}`` body, which does not match the OpenAI error envelope the
#: contract requires. Returning ``None`` instead lets us raise our own.
bearer_scheme = HTTPBearer(
    scheme_name="API key",
    description=(
        "Paste any key from the server's API_KEYS environment variable. "
        "Swagger then sends it as 'Authorization: Bearer <key>' on every "
        "/v1 request. Type the key alone - do not prefix it with 'Bearer'."
    ),
    auto_error=False,
)

Credentials = Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)]


def get_settings(request: Request) -> Settings:
    """Settings resolved once at app start and stashed on ``app.state``."""
    return request.app.state.settings


def get_keystore(request: Request) -> KeyStore:
    return request.app.state.keystore


def get_ollama(request: Request) -> OllamaClient:
    return request.app.state.ollama


def require_api_key(
    request: Request,
    credentials: Credentials,
    keystore: Annotated[KeyStore, Depends(get_keystore)],
) -> KeyRecord:
    """Auth dependency for every ``/v1/*`` route (never applied to ``/healthz``).

    Input:  header "Authorization: Bearer sk-x", with sk-x in the key store
    Output: KeyRecord(key="sk-x", ...)
    """
    # None covers both a missing header and a non-bearer scheme, since
    # HTTPBearer rejects anything that is not "Bearer <token>".
    token = (credentials.credentials or "").strip() if credentials else ""
    if not token:
        raise AuthenticationError(
            "Missing bearer token. Send 'Authorization: Bearer <api_key>'."
        )

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
