r"""Exception hierarchy.

::

    LocalSDKError
    |- APIError                  (the server answered with a non-2xx status)
    |  |- BadRequestError        400
    |  |- AuthenticationError    401
    |  |- PermissionDeniedError  403
    |  |- NotFoundError          404
    |  |- RateLimitError         429
    |  \- UpstreamError          502 / 504
    |- APIConnectionError        could not reach / did not get JSON back
    |  \- APITimeoutError
    |- ConfigurationError        missing api_key / base_url, bad config file
    |- ToolExecutionError        a python tool raised, or its arguments were junk
    |- MaxRoundsExceeded         run_tools() hit its max_rounds cap
    \- StructuredOutputError     structured() got JSON it could not validate

Every :class:`APIError` carries ``.status_code``, ``.code``, ``.type`` and
``.param`` parsed from the OpenAI error envelope (CONTRACT section 6).
"""

from __future__ import annotations

from typing import Any


class LocalSDKError(Exception):
    """Base class for every error raised by this SDK."""


class APIError(LocalSDKError):
    """The server returned a non-2xx status.

    Attributes:
        status_code: HTTP status code, e.g. ``401``.
        code: Machine-readable code from the envelope, e.g. ``"invalid_api_key"``.
        type: Error family from the envelope, e.g. ``"authentication_error"``.
        param: Offending request field, when the server names one.
        body: The parsed response body, if it was JSON.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        code: str | None = None,
        type: str | None = None,  # noqa: A002 - mirrors the wire field name
        param: str | None = None,
        body: Any = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code
        self.type = type
        self.param = param
        self.body = body

    def __str__(self) -> str:
        bits = [self.message]
        detail = ", ".join(
            f"{label}={value!r}"
            for label, value in (
                ("status", self.status_code),
                ("type", self.type),
                ("code", self.code),
                ("param", self.param),
            )
            if value is not None
        )
        if detail:
            bits.append(f"({detail})")
        return " ".join(bits)


class BadRequestError(APIError):
    """400 - the request body was malformed or rejected."""


class AuthenticationError(APIError):
    """401 - missing, malformed or unknown API key."""


class PermissionDeniedError(APIError):
    """403 - the key is known but revoked or not allowed here."""


class NotFoundError(APIError):
    """404 - unknown route, or the model is not present on the server."""


class RateLimitError(APIError):
    """429 - too many requests."""


class UpstreamError(APIError):
    """502 / 504 - the server could not reach Ollama, or Ollama timed out."""


class APIConnectionError(LocalSDKError):
    """The request never produced a usable JSON response.

    Raised for DNS/TCP/TLS failures and, importantly, when the response is
    HTML -- which on a free ngrok tunnel almost always means the browser
    interstitial or a rotated tunnel URL.
    """

    def __init__(self, message: str, *, cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.cause = cause


class APITimeoutError(APIConnectionError):
    """The request exceeded ``timeout`` seconds.

    A 14B model on one box is slow; a large generation legitimately takes
    minutes. Raise ``timeout`` or use :meth:`stream` before assuming a bug.
    """


class ConfigurationError(LocalSDKError):
    """The client could not be constructed: missing or invalid configuration."""


class ToolExecutionError(LocalSDKError):
    """A tool requested by the model could not be executed.

    Attributes:
        tool_name: Name the model asked for.
        tool_call_id: ``id`` of the offending tool call, when known.
    """

    def __init__(
        self,
        message: str,
        *,
        tool_name: str | None = None,
        tool_call_id: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.tool_name = tool_name
        self.tool_call_id = tool_call_id
        self.cause = cause


class MaxRoundsExceeded(LocalSDKError):
    """``run_tools()`` kept being handed tool calls until ``max_rounds`` ran out.

    Attributes:
        rounds: The cap that was hit.
        messages: The conversation as it stood, so you can inspect or resume it.
    """

    def __init__(self, message: str, *, rounds: int, messages: list[dict[str, Any]] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.rounds = rounds
        self.messages = messages or []


class StructuredOutputError(LocalSDKError):
    """``structured()`` got back something that is not valid JSON for the schema.

    Attributes:
        text: The raw assistant content that failed to parse or validate.
    """

    def __init__(self, message: str, *, text: str | None = None, cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.text = text
        self.cause = cause


#: HTTP status -> exception class (CONTRACT section 6).
STATUS_TO_ERROR: dict[int, type[APIError]] = {
    400: BadRequestError,
    401: AuthenticationError,
    403: PermissionDeniedError,
    404: NotFoundError,
    429: RateLimitError,
    502: UpstreamError,
    504: UpstreamError,
}


def error_class_for_status(status_code: int) -> type[APIError]:
    """Return the exception class that maps to ``status_code``.

    Input: ``401``            Output: ``AuthenticationError``
    Input: ``418``            Output: ``APIError``
    """
    return STATUS_TO_ERROR.get(status_code, APIError)


__all__ = [
    "LocalSDKError",
    "APIError",
    "BadRequestError",
    "AuthenticationError",
    "PermissionDeniedError",
    "NotFoundError",
    "RateLimitError",
    "UpstreamError",
    "APIConnectionError",
    "APITimeoutError",
    "ConfigurationError",
    "ToolExecutionError",
    "MaxRoundsExceeded",
    "StructuredOutputError",
    "STATUS_TO_ERROR",
    "error_class_for_status",
]
