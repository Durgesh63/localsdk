"""``POST /v1/chat/completions`` -- streaming and non-streaming.

Streaming is the priority path: Ollama's NDJSON is forwarded to the client as
OpenAI SSE chunks as it arrives, never buffered.  The upstream request is
*primed* (its first line pulled) before the ``StreamingResponse`` is returned so
that a dead or slow Ollama still produces a real 502/504 status code rather than
a 200 whose body happens to contain an error.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.deps import ApiKey, Config, Ollama
from app.errors import APIError, InvalidRequestError
from app.ollama import translate
from app.ollama.client import OllamaClient
from app.ollama.translate import SSE_DONE, chunk, sse
from app.schemas.openai import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChunkDelta,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])

STREAM_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}


def _validate(request: ChatCompletionRequest) -> None:
    """Contract-level checks pydantic cannot express on its own."""
    for index, message in enumerate(request.messages):
        if message.role == "tool" and not message.tool_call_id:
            raise InvalidRequestError(
                "A message with role 'tool' must carry 'tool_call_id'.",
                param=f"messages[{index}].tool_call_id",
            )
        if message.role in {"system", "user", "tool"} and message.content is None:
            raise InvalidRequestError(
                f"A message with role '{message.role}' must carry 'content'.",
                param=f"messages[{index}].content",
            )


@router.post("/chat/completions", response_model=None)
async def chat_completions(
    body: ChatCompletionRequest,
    ollama: Ollama,
    settings: Config,
    _key: ApiKey,
) -> ChatCompletionResponse | StreamingResponse:
    _validate(body)
    model = body.model or settings.default_model
    payload = translate.to_ollama_chat_request(body, settings.default_model)

    if body.stream:
        return await _stream(ollama, payload, model)

    data = await ollama.chat(payload)
    return translate.from_ollama_chat_response(data, model=model)


async def _stream(
    ollama: OllamaClient, payload: dict[str, Any], model: str
) -> StreamingResponse:
    """Open the upstream stream, pull its first line, then start responding."""
    upstream = ollama.chat_stream(payload)
    try:
        first: dict[str, Any] | None = await upstream.__anext__()
    except StopAsyncIteration:
        first = None
    except APIError:
        await upstream.aclose()
        raise

    generator = _sse_events(upstream, first, model=model)
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers=STREAM_HEADERS,
    )


async def _sse_events(
    upstream: AsyncIterator[dict[str, Any]],
    first: dict[str, Any] | None,
    *,
    model: str,
) -> AsyncIterator[str]:
    """Map Ollama NDJSON dicts onto the contract's SSE chunk sequence."""
    completion_id = translate.new_completion_id()
    created = translate.now()

    def make(delta: ChunkDelta | None, finish_reason: str | None = None) -> str:
        return sse(
            chunk(
                completion_id=completion_id,
                created=created,
                model=model,
                delta=delta,
                finish_reason=finish_reason,
            )
        )

    saw_tool_calls = False
    finish_reason: str | None = None

    try:
        # 1. role-only opening chunk
        yield make(ChunkDelta(role="assistant"))

        # 2. content / tool-call deltas, streamed as they arrive
        pending = first
        while pending is not None:
            message = pending.get("message") or {}

            raw_tool_calls = message.get("tool_calls")
            if raw_tool_calls:
                calls = translate.tool_calls_from_ollama(raw_tool_calls)
                saw_tool_calls = True
                yield make(
                    ChunkDelta(
                        tool_calls=[
                            {"index": index, **call} for index, call in enumerate(calls)
                        ]
                    )
                )

            content = message.get("content")
            if content:
                yield make(ChunkDelta(content=content))

            if pending.get("done"):
                finish_reason = translate.finish_reason_from_ollama(
                    pending, saw_tool_calls
                )

            pending = await anext(upstream, None)  # type: ignore[arg-type]

        # 3. terminal chunk carrying finish_reason, then the literal [DONE]
        yield make(None, finish_reason or ("tool_calls" if saw_tool_calls else "stop"))
        yield SSE_DONE
    except APIError as exc:
        logger.warning("stream failed mid-flight: %s", exc.message)
        yield exc.to_sse()
        yield SSE_DONE
    except Exception as exc:  # noqa: BLE001 - never leak a raw traceback mid-stream
        logger.exception("unexpected streaming failure: %s", exc)
        yield APIError(
            "Internal server error while streaming.", code="internal_error"
        ).to_sse()
        yield SSE_DONE
    finally:
        aclose = getattr(upstream, "aclose", None)
        if aclose is not None:
            await aclose()
