"""The synchronous client: a thin mirror of :mod:`localsdk.client`.

Every method here has the same name and the same signature as its async
counterpart, minus the ``await``. There is no event loop anywhere in this file
-- no ``asyncio.run``, no "run this coroutine in a thread" trick -- so it is
safe to use from inside a running loop, a notebook, or a WSGI worker.

The logic is not duplicated: request building, response parsing, SSE decoding
and retry decisions all come from :mod:`localsdk._transport`, the same
functions the async client calls. Only the I/O line differs.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Iterator, Mapping

import httpx
from pydantic import BaseModel

from . import _transport as _t
from .config import Config, resolve_config
from .errors import MaxRoundsExceeded
from .tools import append_tool_results, build_tool_registry, tool_schemas
from .types import ChatResponse, Chunk, Health, MessageInput, Model, normalize_messages


class Client(_t.BaseClient):
    """Synchronous client for a LocalSDK (OpenAI-compatible) server.

    Example::

        with Client(api_key="sk-...", base_url="https://x.ngrok-free.app") as client:
            print(client.chat([{"role": "user", "content": "hi"}]).text)

    See :class:`localsdk.client.AsyncClient` for the argument reference;
    the two constructors are identical.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        *,
        model: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        config_file: str | None = None,
        default_headers: Mapping[str, str] | None = None,
        transport: Any = None,
        http_client: Any = None,
    ) -> None:
        config: Config = resolve_config(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
            config_file=config_file,
        )
        super().__init__(config, default_headers)
        self._owns_http = http_client is None
        self._http: httpx.Client = http_client or httpx.Client(
            timeout=self._httpx_timeout(),
            transport=transport,
            follow_redirects=True,
        )

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        """Close the underlying HTTP pool (no-op for an injected client)."""
        if self._owns_http:
            self._http.close()

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # the only two methods that touch the network
    # ------------------------------------------------------------------ #

    def _send(self, spec: _t.RequestSpec) -> Any:
        """Send a non-streaming request, with retries if they were enabled."""
        attempt = 0
        while True:
            try:
                response = self._http.request(
                    spec.method, spec.url, headers=spec.headers, json=spec.json_body
                )
            except Exception as exc:  # noqa: BLE001 - re-raised below, translated
                delay = _t.retry_delay(
                    attempt=attempt, max_retries=self.max_retries, exc=exc, stream=False
                )
                if delay is None:
                    raise self._translate(exc) from exc
                time.sleep(delay)
                attempt += 1
                continue

            delay = _t.retry_delay(
                attempt=attempt,
                max_retries=self.max_retries,
                status_code=response.status_code,
                stream=False,
            )
            if delay is not None:
                time.sleep(delay)
                attempt += 1
                continue

            return self._check_and_decode(
                status_code=response.status_code,
                content_type=response.headers.get("content-type"),
                body=response.text,
            )

    def _stream_chunks(self, spec: _t.RequestSpec) -> Iterator[Chunk]:
        """Send a streaming request and yield chunks as they arrive. Never retried."""
        decoder = _t.SSEDecoder()
        try:
            with self._http.stream(
                spec.method, spec.url, headers=spec.headers, json=spec.json_body
            ) as response:
                content_type = response.headers.get("content-type")
                if not (200 <= response.status_code < 300) or _t.looks_like_html(content_type):
                    body = response.read().decode("utf-8", "replace")
                    self._check_stream_start(
                        status_code=response.status_code,
                        content_type=content_type,
                        body=body,
                    )
                for text in response.iter_text():
                    for payload in decoder.feed_text(text):
                        if _t.is_done(payload):
                            return
                        chunk = _t.parse_sse_payload(payload)
                        if chunk is not None:
                            yield chunk
                for payload in decoder.finish():
                    if _t.is_done(payload):
                        return
                    chunk = _t.parse_sse_payload(payload)
                    if chunk is not None:
                        yield chunk
        except httpx.HTTPError as exc:
            raise self._translate(exc) from exc

    # ------------------------------------------------------------------ #
    # public API -- same signatures as AsyncClient, minus await
    # ------------------------------------------------------------------ #

    def chat(
        self,
        messages: MessageInput,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[Any] | None = None,
        tool_choice: Any = None,
        response_format: Mapping[str, Any] | None = None,
        **kw: Any,
    ) -> ChatResponse:
        """Send a chat completion and wait for the whole answer.

        Input:  client.chat([{"role": "user", "content": "hi"}])
        Output: ChatResponse with ``.text``, ``.tool_calls``, ``.finish_reason``, ``.usage``
        """
        spec = self._chat_spec(
            messages,
            model=model,
            stream=False,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tool_schemas(tools) if tools else None,
            tool_choice=tool_choice,
            response_format=response_format,
            extra=kw,
        )
        return _t.parse_chat_response(self._send(spec))

    def stream(
        self,
        messages: MessageInput,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[Any] | None = None,
        tool_choice: Any = None,
        response_format: Mapping[str, Any] | None = None,
        **kw: Any,
    ) -> Iterator[Chunk]:
        """Stream a chat completion, yielding :class:`Chunk` objects.

        Example::

            for chunk in client.stream(messages):
                print(chunk.text, end="", flush=True)
        """
        spec = self._chat_spec(
            messages,
            model=model,
            stream=True,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tool_schemas(tools) if tools else None,
            tool_choice=tool_choice,
            response_format=response_format,
            extra=kw,
        )
        return self._stream_chunks(spec)

    def structured(
        self,
        messages: MessageInput,
        schema: type[BaseModel],
        **kw: Any,
    ) -> BaseModel:
        """Ask for JSON matching ``schema`` and return a validated instance.

        Raises:
            StructuredOutputError: the reply was not JSON, or did not fit ``schema``.
        """
        response = self.chat(messages, response_format=_t.build_response_format(schema), **kw)
        return _t.parse_structured(response.text, schema)

    def embed(
        self,
        texts: str | list[str],
        *,
        model: str | None = None,
    ) -> list[list[float]]:
        """Embed one string or a list of strings; always returns a list of vectors."""
        return _t.parse_embeddings(self._send(self._embed_spec(texts, model=model)))

    def models(self) -> list[Model]:
        """List the models the server can serve (``GET /v1/models``)."""
        return _t.parse_models(self._send(self._models_spec()))

    def health(self) -> Health:
        """Probe the server (``GET /healthz``, no auth). Check ``health.ok``."""
        return _t.parse_health(self._send(self._health_spec()))

    def run_tools(
        self,
        messages: MessageInput,
        tools: list[Callable[..., Any]],
        *,
        max_rounds: int = 3,
        **kw: Any,
    ) -> ChatResponse:
        """Opt-in auto-loop: call the model, run the tools it asks for, repeat.

        Raises:
            MaxRoundsExceeded: still requesting tools after ``max_rounds`` calls.
            ToolExecutionError: a tool was unknown, mis-called, or raised.
        """
        if max_rounds < 1:
            raise ValueError("max_rounds must be at least 1, got {0}".format(max_rounds))
        registry = build_tool_registry(tools)
        schemas = tool_schemas(tools)
        conversation = normalize_messages(messages)

        for _ in range(max_rounds):
            response = self.chat(conversation, tools=schemas, **kw)
            if not response.tool_calls:
                return response
            append_tool_results(conversation, response, registry)

        raise MaxRoundsExceeded(
            "the model was still requesting tool calls after {0} round(s). Raise "
            "max_rounds, or check that your tools return something the model can "
            "finish with -- the conversation so far is on .messages.".format(max_rounds),
            rounds=max_rounds,
            messages=conversation,
        )


__all__ = ["Client"]
