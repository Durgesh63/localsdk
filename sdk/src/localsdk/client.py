"""The asynchronous client -- this is where the real logic lives.

:class:`localsdk._sync.Client` is a line-for-line mirror of this module that
swaps ``httpx.AsyncClient`` for ``httpx.Client``. Everything that is not the
actual I/O call (building requests, parsing responses, decoding SSE, deciding
on retries) lives in :mod:`localsdk._transport` and is shared.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Callable, Mapping

import httpx
from pydantic import BaseModel

from . import _transport as _t
from .config import Config, resolve_config
from .errors import MaxRoundsExceeded
from .tools import append_tool_results, build_tool_registry, tool_schemas
from .types import ChatResponse, Chunk, Health, MessageInput, Model, normalize_messages


class AsyncClient(_t.BaseClient):
    """Async client for a LocalSDK (OpenAI-compatible) server.

    Construction resolves configuration by precedence -- explicit argument,
    then environment variable, then ``~/.localsdk/config.toml``, then default
    -- and fails immediately with :class:`~localsdk.errors.ConfigurationError`
    if ``api_key`` or ``base_url`` cannot be found.

    Example::

        async with AsyncClient(api_key="sk-...", base_url="https://x.ngrok-free.app") as client:
            resp = await client.chat([{"role": "user", "content": "hi"}])
            print(resp.text)

    Args:
        api_key: Bearer token. Falls back to ``LOCALSDK_API_KEY``.
        base_url: Server root *without* ``/v1``. Falls back to ``LOCALSDK_BASE_URL``.
            Free ngrok rotates this URL on every restart -- keep it in config.
        model: Default model for calls that do not name one.
        timeout: Per-request timeout in seconds. Default 300; a 14B model is slow.
        max_retries: Retries for *safe* failures only, and **0 by default**:
            requests are metered, so nothing is ever re-sent behind your back.
        config_file: Override the config file path (mostly for tests).
        default_headers: Extra headers merged into every request.
        transport: An ``httpx`` transport, e.g. ``httpx.MockTransport`` in tests.
        http_client: Bring your own ``httpx.AsyncClient``; it is not closed by
            :meth:`close`.
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
        self._http: httpx.AsyncClient = http_client or httpx.AsyncClient(
            timeout=self._httpx_timeout(),
            transport=transport,
            follow_redirects=True,
        )

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #

    async def close(self) -> None:
        """Close the underlying HTTP pool (no-op for an injected client)."""
        if self._owns_http:
            await self._http.aclose()

    async def __aenter__(self) -> "AsyncClient":
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------ #
    # the only two methods that touch the network
    # ------------------------------------------------------------------ #

    async def _send(self, spec: _t.RequestSpec) -> Any:
        """Send a non-streaming request, with retries if they were enabled."""
        attempt = 0
        while True:
            try:
                response = await self._http.request(
                    spec.method, spec.url, headers=spec.headers, json=spec.json_body
                )
            except Exception as exc:  # noqa: BLE001 - re-raised below, translated
                delay = _t.retry_delay(
                    attempt=attempt, max_retries=self.max_retries, exc=exc, stream=False
                )
                if delay is None:
                    raise self._translate(exc) from exc
                await asyncio.sleep(delay)
                attempt += 1
                continue

            delay = _t.retry_delay(
                attempt=attempt,
                max_retries=self.max_retries,
                status_code=response.status_code,
                stream=False,
            )
            if delay is not None:
                await asyncio.sleep(delay)
                attempt += 1
                continue

            return self._check_and_decode(
                status_code=response.status_code,
                content_type=response.headers.get("content-type"),
                body=response.text,
            )

    async def _stream_chunks(self, spec: _t.RequestSpec) -> AsyncIterator[Chunk]:
        """Send a streaming request and yield chunks as they arrive.

        Never retried: a stream that has started has already been paid for and
        cannot be replayed without duplicating tokens.
        """
        decoder = _t.SSEDecoder()
        try:
            async with self._http.stream(
                spec.method, spec.url, headers=spec.headers, json=spec.json_body
            ) as response:
                content_type = response.headers.get("content-type")
                if not (200 <= response.status_code < 300) or _t.looks_like_html(content_type):
                    body = (await response.aread()).decode("utf-8", "replace")
                    self._check_stream_start(
                        status_code=response.status_code,
                        content_type=content_type,
                        body=body,
                    )
                async for text in response.aiter_text():
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
    # public API
    # ------------------------------------------------------------------ #

    async def chat(
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

        Any extra keyword (``top_p``, ``stop``, ``seed``, ...) is passed
        through to the server untouched.

        Input:  await client.chat([{"role": "user", "content": "hi"}])
        Output: ChatResponse with ``.text``, ``.tool_calls``, ``.finish_reason``, ``.usage``

        Raises:
            APIError subclass: on a non-2xx response.
            APIConnectionError: if the reply is HTML (ngrok interstitial or a
                stale tunnel URL) or the server is unreachable.
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
        return _t.parse_chat_response(await self._send(spec))

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
    ) -> AsyncIterator[Chunk]:
        """Stream a chat completion, yielding :class:`Chunk` objects.

        The body is consumed incrementally -- it is never buffered whole.

        Example::

            async for chunk in client.stream(messages):
                print(chunk.text, end="", flush=True)

        Raises:
            APIError subclass: including errors that arrive *mid-stream* as a
                final ``data: {"error": ...}`` event (CONTRACT section 6).
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

    async def structured(
        self,
        messages: MessageInput,
        schema: type[BaseModel],
        **kw: Any,
    ) -> BaseModel:
        """Ask for JSON matching ``schema`` and return a validated instance.

        Sends ``response_format={"type": "json_schema", ...}`` (CONTRACT
        section 5) and validates the reply.

        Input:  await client.structured(messages, Person)
        Output: Person(name="Ada", age=36)

        Raises:
            StructuredOutputError: the model's reply was not JSON, or did not
                fit ``schema``. The raw text is on ``.text``.
        """
        response = await self.chat(messages, response_format=_t.build_response_format(schema), **kw)
        return _t.parse_structured(response.text, schema)

    async def embed(
        self,
        texts: str | list[str],
        *,
        model: str | None = None,
    ) -> list[list[float]]:
        """Embed one string or a list of strings.

        A single string still returns a list of one vector, so callers do not
        have to branch.

        Input:  await client.embed("hello")   Output: [[0.1, 0.2, ...]]
        """
        return _t.parse_embeddings(await self._send(self._embed_spec(texts, model=model)))

    async def models(self) -> list[Model]:
        """List the models the server can serve (``GET /v1/models``)."""
        return _t.parse_models(await self._send(self._models_spec()))

    async def health(self) -> Health:
        """Probe the server (``GET /healthz``, no auth).

        Returns 200 even when Ollama is down, so check ``health.ok``.
        """
        return _t.parse_health(await self._send(self._health_spec()))

    async def run_tools(
        self,
        messages: MessageInput,
        tools: list[Callable[..., Any]],
        *,
        max_rounds: int = 3,
        **kw: Any,
    ) -> ChatResponse:
        """Opt-in auto-loop: call the model, run the tools it asks for, repeat.

        Each round sends the conversation, executes every returned tool call,
        appends one ``{"role": "tool", ...}`` message per call, and calls again.
        It stops as soon as the model answers without tool calls.

        This is a convenience, not a requirement: :meth:`chat` with ``tools=``
        plus :func:`localsdk.tools.append_tool_results` gives you the same
        thing under manual control.

        Input:  await client.run_tools(messages, [get_weather], max_rounds=3)
        Output: the first ChatResponse that carries no tool calls

        Raises:
            MaxRoundsExceeded: the model was still asking for tools after
                ``max_rounds`` calls. The partial conversation is on
                ``.messages`` so you can inspect or resume it.
            ToolExecutionError: a tool was unknown, mis-called, or raised.
        """
        if max_rounds < 1:
            raise ValueError("max_rounds must be at least 1, got {0}".format(max_rounds))
        registry = build_tool_registry(tools)
        schemas = tool_schemas(tools)
        conversation = normalize_messages(messages)

        for _ in range(max_rounds):
            response = await self.chat(conversation, tools=schemas, **kw)
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


__all__ = ["AsyncClient"]
