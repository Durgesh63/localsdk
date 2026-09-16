"""Thin async wrapper around the Ollama HTTP API.

One shared :class:`httpx.AsyncClient` is created in the app lifespan and handed to
this wrapper; the wrapper never creates or closes clients of its own.  Transport
failures are translated into the contract's upstream errors:

* connection refused / DNS / network error -> 502 ``ollama_unavailable``
* read or connect timeout                  -> 504 ``ollama_timeout``
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

import httpx

from app.errors import (
    APIError,
    ModelNotFoundError,
    OllamaTimeoutError,
    OllamaUnavailableError,
)

logger = logging.getLogger(__name__)


class OllamaClient:
    """Async client for the handful of Ollama endpoints this service needs."""

    def __init__(self, http: httpx.AsyncClient, base_url: str = "") -> None:
        self._http = http
        self._base_url = base_url.rstrip("/")

    # -- internals ---------------------------------------------------------- #
    def _url(self, path: str) -> str:
        return f"{self._base_url}{path}" if self._base_url else path

    @staticmethod
    def _raise_transport(exc: Exception) -> APIError:
        if isinstance(exc, httpx.TimeoutException):
            return OllamaTimeoutError(
                "Ollama did not respond within REQUEST_TIMEOUT_S."
            )
        return OllamaUnavailableError(f"Cannot reach Ollama: {exc}")

    @staticmethod
    def _raise_status(response: httpx.Response, body: str) -> APIError:
        detail = body.strip()
        try:
            parsed = json.loads(detail)
            detail = str(parsed.get("error", detail))
        except (ValueError, AttributeError):
            pass
        if response.status_code == 404:
            return ModelNotFoundError(
                detail or "The requested model is not present in Ollama."
            )
        return OllamaUnavailableError(
            f"Ollama returned {response.status_code}: {detail or 'upstream error'}"
        )

    @staticmethod
    def _decode(response: httpx.Response) -> dict[str, Any]:
        """Parse a JSON body, turning garbage from upstream into a 502."""
        try:
            data = response.json()
        except ValueError as exc:
            raise OllamaUnavailableError(
                "Ollama returned a non-JSON response."
            ) from exc
        if not isinstance(data, dict):
            raise OllamaUnavailableError("Ollama returned an unexpected JSON shape.")
        return data

    async def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = await self._http.post(self._url(path), json=payload)
        except httpx.HTTPError as exc:  # includes timeouts and connect errors
            raise self._raise_transport(exc) from exc
        if response.status_code >= 400:
            raise self._raise_status(response, response.text)
        return self._decode(response)

    # -- endpoints ---------------------------------------------------------- #
    async def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST /api/chat with ``stream: false``."""
        body = {**payload, "stream": False}
        return await self._post_json("/api/chat", body)

    async def chat_stream(self, payload: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        """POST /api/chat with ``stream: true``, yielding one dict per NDJSON line."""
        body = {**payload, "stream": True}
        try:
            async with self._http.stream(
                "POST", self._url("/api/chat"), json=body
            ) as response:
                if response.status_code >= 400:
                    raw = (await response.aread()).decode("utf-8", "replace")
                    raise self._raise_status(response, raw)
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except ValueError:
                        logger.warning("skipping non-JSON line from ollama: %r", line[:200])
        except httpx.HTTPError as exc:
            raise self._raise_transport(exc) from exc

    async def embeddings(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST /api/embed (falls back to the legacy /api/embeddings on 404)."""
        try:
            return await self._post_json("/api/embed", payload)
        except ModelNotFoundError:
            legacy = {
                "model": payload.get("model"),
                "prompt": payload.get("input"),
            }
            return await self._post_json("/api/embeddings", legacy)

    async def list_models(self) -> dict[str, Any]:
        """GET /api/tags."""
        try:
            response = await self._http.get(self._url("/api/tags"))
        except httpx.HTTPError as exc:
            raise self._raise_transport(exc) from exc
        if response.status_code >= 400:
            raise self._raise_status(response, response.text)
        return self._decode(response)

    async def ping(self) -> bool:
        """``True`` when Ollama answers /api/tags. Never raises."""
        try:
            response = await self._http.get(self._url("/api/tags"), timeout=5.0)
            return response.status_code < 500
        except Exception:  # noqa: BLE001 - health probe must never raise
            return False
