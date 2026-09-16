"""OpenAI-shaped error envelope, exception classes and FastAPI handlers.

Everything the service can fail with funnels through :class:`APIError` so that
every response body matches CONTRACT section 6:

    {"error": {"message": "...", "type": "...", "param": null, "code": "..."}}
"""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


def error_body(
    message: str,
    type_: str,
    code: str | None = None,
    param: str | None = None,
) -> dict[str, Any]:
    """Build the OpenAI error envelope.

    Input:  error_body("nope", "authentication_error", "invalid_api_key")
    Output: {"error": {"message": "nope", "type": "authentication_error",
                       "param": None, "code": "invalid_api_key"}}
    """
    return {
        "error": {
            "message": message,
            "type": type_,
            "param": param,
            "code": code,
        }
    }


class APIError(Exception):
    """Base class for every error the API deliberately returns."""

    status_code: int = 500
    type: str = "api_error"
    code: str | None = None

    def __init__(
        self,
        message: str,
        *,
        param: str | None = None,
        code: str | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.param = param
        if code is not None:
            self.code = code
        if status_code is not None:
            self.status_code = status_code

    def to_dict(self) -> dict[str, Any]:
        return error_body(self.message, self.type, self.code, self.param)

    def to_response(self) -> JSONResponse:
        return JSONResponse(status_code=self.status_code, content=self.to_dict())

    def to_sse(self) -> str:
        """Render as the mid-stream SSE error event (CONTRACT section 6)."""
        return f"data: {json.dumps(self.to_dict(), separators=(',', ':'))}\n\n"


class InvalidRequestError(APIError):
    status_code = 400
    type = "invalid_request_error"
    code = "invalid_request"


class AuthenticationError(APIError):
    status_code = 401
    type = "authentication_error"
    code = "invalid_api_key"


class KeyRevokedError(APIError):
    status_code = 403
    type = "permission_error"
    code = "key_revoked"


class ModelNotFoundError(APIError):
    status_code = 404
    type = "invalid_request_error"
    code = "model_not_found"


class RateLimitError(APIError):
    """Reserved: not enforced in v1 (CONTRACT section 6)."""

    status_code = 429
    type = "rate_limit_error"
    code = "rate_limit_exceeded"


class OllamaUnavailableError(APIError):
    status_code = 502
    type = "upstream_error"
    code = "ollama_unavailable"


class OllamaTimeoutError(APIError):
    status_code = 504
    type = "upstream_error"
    code = "ollama_timeout"


_STATUS_FALLBACK: dict[int, tuple[str, str]] = {
    400: ("invalid_request_error", "invalid_request"),
    401: ("authentication_error", "invalid_api_key"),
    403: ("permission_error", "key_revoked"),
    404: ("invalid_request_error", "model_not_found"),
    405: ("invalid_request_error", "invalid_request"),
    429: ("rate_limit_error", "rate_limit_exceeded"),
    502: ("upstream_error", "ollama_unavailable"),
    504: ("upstream_error", "ollama_timeout"),
}


def _format_validation_errors(exc: RequestValidationError) -> tuple[str, str | None]:
    """Turn pydantic's error list into a message plus the offending param name."""
    errors = exc.errors()
    if not errors:
        return "Invalid request body.", None
    first = errors[0]
    location = [str(part) for part in first.get("loc", ()) if part != "body"]
    param = ".".join(location) or None
    message = first.get("msg", "Invalid request body.")
    if param:
        message = f"{param}: {message}"
    return message, param


def register_exception_handlers(app: FastAPI) -> None:
    """Install handlers so *every* failure path emits the OpenAI envelope."""

    @app.exception_handler(APIError)
    async def _api_error_handler(_: Request, exc: APIError) -> JSONResponse:
        return exc.to_response()

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(
        _: Request, exc: RequestValidationError
    ) -> JSONResponse:
        message, param = _format_validation_errors(exc)
        return JSONResponse(
            status_code=400,
            content=error_body(message, "invalid_request_error", "invalid_request", param),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_handler(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        type_, code = _STATUS_FALLBACK.get(exc.status_code, ("api_error", "internal_error"))
        detail = exc.detail if isinstance(exc.detail, str) else "Request failed."
        return JSONResponse(
            status_code=exc.status_code,
            content=error_body(detail, type_, code),
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(Exception)
    async def _unhandled_handler(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled server error: %s", exc)
        return JSONResponse(
            status_code=500,
            content=error_body(
                "Internal server error.", "api_error", "internal_error"
            ),
        )
