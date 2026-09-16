"""Shared HTTP layer: headers, request building, response parsing, SSE, retries.

Everything in this module is either a **pure function** (build a payload, parse
a body, decide whether to retry) or a small stateful decoder. None of it does
I/O. That is what lets :class:`localsdk.client.AsyncClient` and
:class:`localsdk._sync.Client` share one implementation and differ only in
the single line that actually sends bytes.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

import httpx
from pydantic import BaseModel, ValidationError

from ._version import USER_AGENT
from .config import Config
from .errors import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    StructuredOutputError,
    error_class_for_status,
)
from .types import (
    ChatResponse,
    Chunk,
    Health,
    Model,
    normalize_messages,
)

#: Without this header ngrok serves an HTML interstitial instead of the API
#: (CONTRACT section 2). It goes on *every* request, including /healthz.
NGROK_HEADER = "ngrok-skip-browser-warning"

#: SSE terminator (CONTRACT section 3).
DONE_SENTINEL = "[DONE]"

#: Statuses worth retrying: transient upstream failures only.
RETRYABLE_STATUS = frozenset({502, 504})

_HTML_HINT = (
    "The server replied with HTML, not JSON.\n"
    "On a free ngrok tunnel this almost always means one of:\n"
    "  1. the ngrok browser interstitial -- the request reached ngrok but the\n"
    "     'ngrok-skip-browser-warning: true' header was stripped by a proxy; or\n"
    "  2. the tunnel URL is stale. Free ngrok mints a NEW URL every restart, so\n"
    "     the base_url you configured may now point at a dead or foreign tunnel.\n"
    "What to do: run 'curl -H \"ngrok-skip-browser-warning: true\" {base}/healthz'.\n"
    "If that is also HTML, get the current URL from the ngrok console and update\n"
    "base_url (or {env})."
)


# --------------------------------------------------------------------------- #
# request building
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RequestSpec:
    """A fully built, transport-agnostic HTTP request.

    Both clients build one of these with the same code; only the send differs.
    """

    method: str
    url: str
    headers: dict[str, str]
    json_body: dict[str, Any] | None = None
    stream: bool = False


def build_headers(
    api_key: str | None,
    *,
    stream: bool = False,
    extra: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build the header set for one request.

    ``Authorization`` is omitted for unauthenticated routes (``/healthz``).
    The ngrok header and User-Agent are always present (CONTRACT section 2).

    Input:  build_headers("sk-1")
    Output: {"Accept": "application/json", "Content-Type": "application/json",
             "User-Agent": "localsdk-python/0.1.0",
             "ngrok-skip-browser-warning": "true",
             "Authorization": "Bearer sk-1"}
    """
    headers: dict[str, str] = {
        "Accept": "text/event-stream" if stream else "application/json",
        "Content-Type": "application/json",
        "User-Agent": USER_AGENT,
        NGROK_HEADER: "true",
    }
    if api_key:
        headers["Authorization"] = "Bearer {0}".format(api_key)
    if extra:
        headers.update({str(k): str(v) for k, v in extra.items()})
    return headers


def _drop_none(payload: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in payload.items() if v is not None}


def build_chat_payload(
    messages: Any,
    *,
    model: str,
    stream: bool = False,
    temperature: float | None = None,
    max_tokens: int | None = None,
    tools: list[Any] | None = None,
    tool_choice: Any = None,
    response_format: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the body for ``POST /v1/chat/completions``.

    Unset options are omitted entirely rather than sent as null, so the server
    applies its own defaults.

    Input:  build_chat_payload([{"role": "user", "content": "hi"}], model="m")
    Output: {"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": False}
    """
    payload: dict[str, Any] = {
        "model": model,
        "messages": normalize_messages(messages),
        "stream": stream,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "tools": list(tools) if tools else None,
        "tool_choice": tool_choice,
        "response_format": dict(response_format) if response_format else None,
    }
    payload = _drop_none(payload)
    if extra:
        # Caller-supplied passthrough (top_p, stop, seed, ...). Explicit
        # None means "do not send", matching the named options above.
        payload.update({k: v for k, v in extra.items() if v is not None})
    return payload


def build_embeddings_payload(
    texts: str | Iterable[str],
    *,
    model: str | None = None,
) -> dict[str, Any]:
    """Build the body for ``POST /v1/embeddings``.

    Input:  build_embeddings_payload("hi")        Output: {"input": ["hi"]}
    Input:  build_embeddings_payload(["a", "b"])  Output: {"input": ["a", "b"]}
    """
    if isinstance(texts, str):
        items = [texts]
    else:
        items = [str(t) for t in texts]
    if not items:
        raise ValueError("embed() needs at least one string")
    payload: dict[str, Any] = {"input": items}
    if model:
        payload["model"] = model
    return payload


# --------------------------------------------------------------------------- #
# response checking / parsing
# --------------------------------------------------------------------------- #


def html_interstitial_message(base_url: str, env_var_name: str) -> str:
    """The long-form explanation raised when we get HTML instead of JSON."""
    return _HTML_HINT.format(base=base_url, env=env_var_name)


def looks_like_html(content_type: str | None, body: str | None = None) -> bool:
    """True when a response is an HTML page rather than a JSON API reply.

    Input:  ("text/html; charset=utf-8", None)   Output: True
    Input:  ("application/json", None)           Output: False
    """
    if content_type and "text/html" in content_type.lower():
        return True
    if body:
        head = body.lstrip()[:64].lower()
        if head.startswith("<!doctype html") or head.startswith("<html"):
            return True
    return False


def parse_error_envelope(body: str | None) -> dict[str, Any]:
    """Pull ``message``/``type``/``code``/``param`` out of the OpenAI envelope.

    Input:  '{"error": {"message": "bad key", "code": "invalid_api_key"}}'
    Output: {"message": "bad key", "code": "invalid_api_key", ...}
    """
    empty: dict[str, Any] = {"message": None, "type": None, "code": None, "param": None, "body": None}
    if not body:
        return empty
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, TypeError):
        return {**empty, "message": body.strip()[:500] or None}
    if not isinstance(data, dict):
        return {**empty, "body": data}
    err = data.get("error")
    if isinstance(err, dict):
        return {
            "message": err.get("message"),
            "type": err.get("type"),
            "code": err.get("code"),
            "param": err.get("param"),
            "body": data,
        }
    if isinstance(err, str):
        return {**empty, "message": err, "body": data}
    return {**empty, "message": data.get("message"), "body": data}


def raise_for_response(
    *,
    status_code: int,
    content_type: str | None,
    body: str | None,
    base_url: str,
    env_var_name: str,
) -> None:
    """Raise the mapped exception for a bad response; return None if it is fine.

    HTML is checked *first*: a stale ngrok tunnel answers 404 with an HTML
    page, and "not found" would be a misleading thing to tell the user.
    """
    if looks_like_html(content_type, body):
        raise APIConnectionError(html_interstitial_message(base_url, env_var_name))
    if 200 <= status_code < 300:
        return
    parsed = parse_error_envelope(body)
    message = parsed["message"] or "HTTP {0} from the API".format(status_code)
    raise error_class_for_status(status_code)(
        message,
        status_code=status_code,
        code=parsed["code"],
        type=parsed["type"],
        param=parsed["param"],
        body=parsed["body"],
    )


def decode_json(body: str, *, base_url: str, env_var_name: str) -> Any:
    """Parse a 2xx body as JSON, with an ngrok-aware error when it is not.

    Raises:
        APIConnectionError: the body was HTML or otherwise not JSON.
    """
    try:
        return json.loads(body)
    except (json.JSONDecodeError, TypeError) as exc:
        if looks_like_html(None, body):
            raise APIConnectionError(html_interstitial_message(base_url, env_var_name)) from exc
        raise APIConnectionError(
            "expected JSON from the API but got {0} bytes that do not parse: "
            "{1!r}".format(len(body or ""), (body or "")[:200]),
            cause=exc,
        ) from exc


def parse_chat_response(data: Any) -> ChatResponse:
    """Turn a chat completion body into a :class:`ChatResponse` (keeping ``.raw``)."""
    if not isinstance(data, dict):
        raise APIConnectionError(
            "expected a JSON object from /v1/chat/completions, got {0}".format(type(data).__name__)
        )
    return ChatResponse.model_validate({**data, "raw": data})


def parse_chunk(data: Mapping[str, Any]) -> Chunk:
    """Turn one decoded SSE event into a :class:`Chunk` (keeping ``.raw``)."""
    return Chunk.model_validate({**dict(data), "raw": dict(data)})


def parse_models(data: Any) -> list[Model]:
    """Extract the model list from ``GET /v1/models``."""
    items = data.get("data", []) if isinstance(data, dict) else []
    return [Model.model_validate(item) for item in items if isinstance(item, Mapping)]


def parse_health(data: Any) -> Health:
    """Parse ``GET /healthz``."""
    return Health.model_validate(data if isinstance(data, Mapping) else {})


def parse_embeddings(data: Any) -> list[list[float]]:
    """Extract vectors from ``POST /v1/embeddings``, ordered by ``index``.

    Input:  {"data": [{"index": 0, "embedding": [0.1, 0.2]}]}
    Output: [[0.1, 0.2]]
    """
    if not isinstance(data, Mapping):
        raise APIConnectionError("expected a JSON object from /v1/embeddings")
    items = list(data.get("data") or [])
    items.sort(key=lambda item: item.get("index", 0) if isinstance(item, Mapping) else 0)
    return [list(item.get("embedding") or []) for item in items if isinstance(item, Mapping)]


# --------------------------------------------------------------------------- #
# SSE
# --------------------------------------------------------------------------- #


@dataclass
class SSEDecoder:
    """Incremental server-sent-events decoder.

    Feed it one line at a time (no trailing newline required). It accumulates
    ``data:`` lines, joins multi-line events with ``\\n``, ignores comment
    lines starting with ``:``, and emits a payload on each blank separator.

    Input:  feed("data: {\\"a\\": 1}") -> None ; feed("") -> '{"a": 1}'
    """

    _data: list[str] = field(default_factory=list)
    _event: str | None = None
    _buffer: str = ""

    def feed_text(self, text: str) -> list[str]:
        """Consume an arbitrary chunk of the response body; return whole events.

        Chunk boundaries do not have to align with lines, so the body is never
        buffered beyond the current partial line.

        Input:  feed_text('data: {"a": 1}\\n\\ndata: [DO')
        Output: ['{"a": 1}']
        """
        self._buffer += text
        *lines, self._buffer = self._buffer.split("\n")
        payloads: list[str] = []
        for line in lines:
            payload = self.feed(line)
            if payload is not None:
                payloads.append(payload)
        return payloads

    def finish(self) -> list[str]:
        """Flush a body that ended without a trailing blank line."""
        payloads: list[str] = []
        if self._buffer:
            tail, self._buffer = self._buffer, ""
            payload = self.feed(tail)
            if payload is not None:
                payloads.append(payload)
        payload = self.flush()
        if payload is not None:
            payloads.append(payload)
        return payloads

    def feed(self, line: str) -> str | None:
        """Consume one line; return a complete event payload, or None."""
        line = line.rstrip("\n").rstrip("\r")
        if line == "":
            return self._emit()
        if line.startswith(":"):
            return None  # comment / keep-alive
        field_name, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if field_name == "data":
            self._data.append(value)
        elif field_name == "event":
            self._event = value
        # id/retry and unknown fields are accepted and ignored
        return None

    def flush(self) -> str | None:
        """Emit a trailing event that was not followed by a blank line."""
        return self._emit()

    def _emit(self) -> str | None:
        if not self._data:
            self._event = None
            return None
        payload = "\n".join(self._data)
        self._data = []
        self._event = None
        return payload


def parse_sse_payload(payload: str) -> Chunk | None:
    """Decode one SSE data payload into a :class:`Chunk`.

    Returns None for the ``[DONE]`` terminator and for payloads that are not
    JSON objects (defensive: a stray keep-alive should not kill a stream).

    Raises:
        APIError subclass: if the event carries an ``error`` object
            (CONTRACT section 6, mid-stream errors).
    """
    text = payload.strip()
    if not text:
        return None
    if text == DONE_SENTINEL:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise APIConnectionError(
            "could not decode a streaming event as JSON: {0!r}".format(text[:200]),
            cause=exc,
        ) from exc
    if not isinstance(data, dict):
        return None
    if "error" in data:
        raise_stream_error(data)
    return parse_chunk(data)


def is_done(payload: str) -> bool:
    """True for the literal ``data: [DONE]`` terminator payload."""
    return payload.strip() == DONE_SENTINEL


def raise_stream_error(data: Mapping[str, Any]) -> None:
    """Raise the mapped exception for a mid-stream ``{"error": {...}}`` event."""
    err = data.get("error")
    if isinstance(err, Mapping):
        message = err.get("message") or "the stream ended with an error"
        code = err.get("code")
        etype = err.get("type")
        param = err.get("param")
        status = err.get("status") or err.get("status_code")
    else:
        message = str(err) if err else "the stream ended with an error"
        code = etype = param = None
        status = None
    status_code = int(status) if isinstance(status, int) else _status_for_type(etype, code)
    cls = error_class_for_status(status_code) if status_code else APIError
    raise cls(
        "mid-stream error: {0}".format(message),
        status_code=status_code,
        code=code,
        type=etype,
        param=param,
        body=dict(data),
    )


_TYPE_TO_STATUS = {
    "invalid_request_error": 400,
    "authentication_error": 401,
    "permission_error": 403,
    "rate_limit_error": 429,
    "upstream_error": 502,
}
_CODE_TO_STATUS = {
    "invalid_request": 400,
    "invalid_api_key": 401,
    "key_revoked": 403,
    "model_not_found": 404,
    "rate_limit_exceeded": 429,
    "ollama_unavailable": 502,
    "ollama_timeout": 504,
}


def _status_for_type(etype: Any, code: Any) -> int | None:
    """Best-effort HTTP status for an error event that carries no status."""
    if isinstance(code, str) and code in _CODE_TO_STATUS:
        return _CODE_TO_STATUS[code]
    if isinstance(etype, str) and etype in _TYPE_TO_STATUS:
        return _TYPE_TO_STATUS[etype]
    return None


# --------------------------------------------------------------------------- #
# retries
# --------------------------------------------------------------------------- #


def is_connect_error(exc: BaseException) -> bool:
    """True for failures that happened before the server saw the request.

    Only these are safe to retry: the request was never delivered, so a retry
    cannot double-charge a metered generation.
    """
    return isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout, httpx.NetworkError))


def backoff_delay(
    attempt: int,
    *,
    base: float = 0.5,
    cap: float = 8.0,
    rng: Callable[[], float] = random.random,
) -> float:
    """Exponential backoff with full jitter for retry number ``attempt`` (0-based).

    Input:  backoff_delay(0, rng=lambda: 1.0)   Output: 0.5
    Input:  backoff_delay(3, rng=lambda: 1.0)   Output: 4.0
    """
    ceiling = min(cap, base * (2**attempt))
    return ceiling * rng()


def retry_delay(
    *,
    attempt: int,
    max_retries: int,
    status_code: int | None = None,
    exc: BaseException | None = None,
    stream: bool = False,
    rng: Callable[[], float] = random.random,
) -> float | None:
    """Decide whether to retry, and for how long to wait first.

    Returns the sleep in seconds, or None to give up and raise. Streaming
    requests are never retried (a partially consumed stream cannot be
    replayed, and the generation has already been paid for).

    Input:  retry_delay(attempt=0, max_retries=0, status_code=502)  Output: None
    Input:  retry_delay(attempt=0, max_retries=2, status_code=502, rng=lambda: 1.0)
    Output: 0.5
    """
    if stream or max_retries <= 0 or attempt >= max_retries:
        return None
    retryable = (status_code in RETRYABLE_STATUS) or (exc is not None and is_connect_error(exc))
    if not retryable:
        return None
    return backoff_delay(attempt, rng=rng)


# --------------------------------------------------------------------------- #
# structured output
# --------------------------------------------------------------------------- #


def build_response_format(schema: type[BaseModel]) -> dict[str, Any]:
    """Build the ``response_format`` for a pydantic model (CONTRACT section 5).

    Input:  build_response_format(Person)
    Output: {"type": "json_schema", "json_schema": {"name": "Person",
             "schema": {...}, "strict": True}}
    """
    if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
        raise TypeError(
            "structured() needs a pydantic BaseModel subclass, got {0!r}".format(schema)
        )
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema.__name__,
            "schema": schema.model_json_schema(),
            "strict": True,
        },
    }


def _strip_code_fence(text: str) -> str:
    """Drop a ```json ... ``` wrapper that small models like to add."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    body = stripped[3:]
    if body.lower().startswith("json"):
        body = body[4:]
    if body.endswith("```"):
        body = body[:-3]
    return body.strip()


def parse_structured(text: str, schema: type[BaseModel]) -> BaseModel:
    """Validate an assistant reply into ``schema``.

    Raises:
        StructuredOutputError: the reply was not JSON, or did not fit the model.
            The offending text is kept on ``.text``.
    """
    candidate = _strip_code_fence(text or "")
    if not candidate:
        raise StructuredOutputError(
            "the model returned an empty message, so there is no JSON to validate "
            "against {0}.".format(schema.__name__),
            text=text,
        )
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise StructuredOutputError(
            "the model did not return valid JSON for {0}: {1}. Raw content: "
            "{2!r}".format(schema.__name__, exc, candidate[:500]),
            text=text,
            cause=exc,
        ) from exc
    try:
        return schema.model_validate(data)
    except ValidationError as exc:
        raise StructuredOutputError(
            "the model returned JSON that does not match {0}: {1}".format(
                schema.__name__, exc
            ),
            text=text,
            cause=exc,
        ) from exc


# --------------------------------------------------------------------------- #
# exception translation for httpx
# --------------------------------------------------------------------------- #


def translate_transport_error(exc: Exception, *, base_url: str, timeout: float) -> Exception:
    """Map an httpx transport exception onto this SDK's hierarchy."""
    if isinstance(exc, httpx.TimeoutException):
        return APITimeoutError(
            "request to {0} timed out after {1:g}s. A 14B model on one box is slow: "
            "raise timeout=, or use stream() so you see tokens as they arrive.".format(
                base_url, timeout
            ),
            cause=exc,
        )
    if isinstance(exc, httpx.TransportError):
        return APIConnectionError(
            "could not reach {0}: {1}. Check that the server is running and that "
            "base_url matches the CURRENT ngrok URL -- free ngrok rotates it on every "
            "restart.".format(base_url, exc),
            cause=exc,
        )
    return exc


# --------------------------------------------------------------------------- #
# shared client base
# --------------------------------------------------------------------------- #


class BaseClient:
    """Configuration, header and request-spec logic shared by both clients.

    Subclasses supply only the I/O: :class:`~localsdk.client.AsyncClient`
    with ``httpx.AsyncClient``, :class:`~localsdk._sync.Client` with
    ``httpx.Client``.
    """

    def __init__(self, config: Config, default_headers: Mapping[str, str] | None = None) -> None:
        self._config = config
        self._default_headers = dict(default_headers or {})

    # -- introspection ----------------------------------------------------- #

    @property
    def config(self) -> Config:
        """The resolved configuration this client was built with."""
        return self._config

    @property
    def base_url(self) -> str:
        """Configured base URL, without the ``/v1`` suffix."""
        return self._config.base_url

    @property
    def model(self) -> str:
        """Default model used when a call does not name one."""
        return self._config.model

    @property
    def max_retries(self) -> int:
        """How many times a *safe* failure is retried. Zero by default."""
        return self._config.max_retries

    @property
    def timeout(self) -> float:
        """Per-request timeout in seconds."""
        return self._config.timeout

    # -- request construction ---------------------------------------------- #

    def _url(self, path: str) -> str:
        """Absolute URL for a route. Leading-slash paths are server-root
        relative (``/healthz``, ``/v1/models``); anything else is relative to
        ``{base_url}/v1``."""
        if path.startswith("/"):
            return "{0}{1}".format(self._config.base_url, path)
        return "{0}/{1}".format(self._config.api_base, path)

    def _spec(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        stream: bool = False,
        authenticated: bool = True,
    ) -> RequestSpec:
        """Build a :class:`RequestSpec`. Pure: no I/O, safe to unit test."""
        return RequestSpec(
            method=method,
            url=self._url(path),
            headers=build_headers(
                self._config.api_key if authenticated else None,
                stream=stream,
                extra=self._default_headers,
            ),
            json_body=json_body,
            stream=stream,
        )

    def _chat_spec(
        self,
        messages: Any,
        *,
        model: str | None,
        stream: bool,
        temperature: float | None,
        max_tokens: int | None,
        tools: list[Any] | None,
        tool_choice: Any,
        response_format: Mapping[str, Any] | None,
        extra: Mapping[str, Any] | None,
    ) -> RequestSpec:
        payload = build_chat_payload(
            messages,
            model=model or self._config.model,
            stream=stream,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            extra=extra,
        )
        return self._spec("POST", "/v1/chat/completions", json_body=payload, stream=stream)

    def _embed_spec(self, texts: str | Iterable[str], *, model: str | None) -> RequestSpec:
        payload = build_embeddings_payload(texts, model=model)
        return self._spec("POST", "/v1/embeddings", json_body=payload)

    def _models_spec(self) -> RequestSpec:
        return self._spec("GET", "/v1/models")

    def _health_spec(self) -> RequestSpec:
        # /healthz takes no auth (CONTRACT section 3) but still needs the
        # ngrok header, which is exactly why it is built here too.
        return self._spec("GET", "/healthz", authenticated=False)

    # -- shared response handling ------------------------------------------ #

    def _env_var_name(self) -> str:
        from .config import env_var  # local import keeps the module import-light

        return env_var("base_url")

    def _check_and_decode(self, *, status_code: int, content_type: str | None, body: str) -> Any:
        """Validate a non-streaming response and decode it."""
        raise_for_response(
            status_code=status_code,
            content_type=content_type,
            body=body,
            base_url=self._config.base_url,
            env_var_name=self._env_var_name(),
        )
        return decode_json(
            body, base_url=self._config.base_url, env_var_name=self._env_var_name()
        )

    def _check_stream_start(self, *, status_code: int, content_type: str | None, body: str) -> None:
        """Validate the headers/status of a streaming response before iterating."""
        raise_for_response(
            status_code=status_code,
            content_type=content_type,
            body=body,
            base_url=self._config.base_url,
            env_var_name=self._env_var_name(),
        )

    def _translate(self, exc: Exception) -> Exception:
        return translate_transport_error(
            exc, base_url=self._config.base_url, timeout=self._config.timeout
        )

    def _httpx_timeout(self) -> httpx.Timeout:
        """Long read timeout (generations are slow), short connect timeout."""
        return httpx.Timeout(self._config.timeout, connect=min(10.0, self._config.timeout))

    def __repr__(self) -> str:
        return "{0}(base_url={1!r}, model={2!r}, max_retries={3})".format(
            type(self).__name__, self._config.base_url, self._config.model, self._config.max_retries
        )


__all__ = [
    "NGROK_HEADER",
    "DONE_SENTINEL",
    "RETRYABLE_STATUS",
    "RequestSpec",
    "BaseClient",
    "SSEDecoder",
    "backoff_delay",
    "build_chat_payload",
    "build_embeddings_payload",
    "build_headers",
    "decode_json",
    "html_interstitial_message",
    "is_connect_error",
    "is_done",
    "looks_like_html",
    "parse_chat_response",
    "parse_chunk",
    "parse_embeddings",
    "parse_error_envelope",
    "parse_health",
    "parse_models",
    "parse_sse_payload",
    "raise_for_response",
    "raise_stream_error",
    "retry_delay",
    "build_response_format",
    "parse_structured",
    "translate_transport_error",
]
