"""OpenAI <-> Ollama request and response mapping.

Pure functions only: no I/O, no globals.  The tricky parts are all here:

* OpenAI ``function.arguments`` is a JSON-*encoded string*, Ollama uses an object,
  so the two directions need ``json.loads`` / ``json.dumps``.
* ``response_format`` collapses into Ollama's single ``format`` field.
* SSE chunk shapes are built here so the route stays a plain loop.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from app.schemas.openai import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    ChunkChoice,
    ChunkDelta,
    EmbeddingsResponse,
    Model,
    ModelList,
    ResponseMessage,
    Usage,
)

SSE_DONE = "data: [DONE]\n\n"


# --------------------------------------------------------------------------- #
# ids / misc
# --------------------------------------------------------------------------- #
def new_completion_id() -> str:
    """``chatcmpl-<uuid4 hex>``."""
    return f"chatcmpl-{uuid.uuid4().hex}"


def new_tool_call_id() -> str:
    """``call_<uuid4 hex[:24]>`` (OpenAI-ish shape)."""
    return f"call_{uuid.uuid4().hex[:24]}"


def now() -> int:
    return int(time.time())


def sse(payload: dict[str, Any]) -> str:
    """Serialize one SSE ``data:`` event, blank-line terminated."""
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


# --------------------------------------------------------------------------- #
# OpenAI -> Ollama
# --------------------------------------------------------------------------- #
def _tool_name_by_call_id(messages: list[ChatMessage]) -> dict[str, str]:
    """Map ``tool_call_id -> function name`` from prior assistant messages."""
    mapping: dict[str, str] = {}
    for message in messages:
        for call in message.tool_calls or []:
            mapping[call.id] = call.function.name
    return mapping


def _decode_arguments(arguments: str) -> dict[str, Any]:
    """Ollama wants an object; OpenAI sends a string.

    Input:  '{"city": "Pune"}'
    Output: {'city': 'Pune'}
    """
    if not arguments:
        return {}
    try:
        decoded = json.loads(arguments)
    except ValueError:
        return {"_raw": arguments}
    return decoded if isinstance(decoded, dict) else {"_raw": decoded}


def to_ollama_messages(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    """Translate OpenAI messages into Ollama messages."""
    names = _tool_name_by_call_id(messages)
    out: list[dict[str, Any]] = []
    for message in messages:
        item: dict[str, Any] = {"role": message.role, "content": message.content or ""}
        if message.role == "tool":
            # Ollama identifies tool results by name, OpenAI by tool_call_id.
            name = message.name or names.get(message.tool_call_id or "")
            if name:
                item["tool_name"] = name
            if message.tool_call_id:
                item["tool_call_id"] = message.tool_call_id
        elif message.tool_calls:
            item["tool_calls"] = [
                {
                    "function": {
                        "name": call.function.name,
                        "arguments": _decode_arguments(call.function.arguments),
                    }
                }
                for call in message.tool_calls
            ]
        out.append(item)
    return out


def to_ollama_format(request: ChatCompletionRequest) -> Any | None:
    """Map ``response_format`` onto Ollama's ``format`` field (CONTRACT section 5).

    Input:  {"type": "json_object"}                      -> "json"
    Input:  {"type": "json_schema", "json_schema": {...}} -> the raw JSON schema
    """
    response_format = request.response_format
    if response_format is None or response_format.type == "text":
        return None
    if response_format.type == "json_object":
        return "json"
    wrapper = response_format.json_schema or {}
    schema = wrapper.get("schema", wrapper)
    return schema or "json"


def to_ollama_options(request: ChatCompletionRequest) -> dict[str, Any]:
    """Map sampling parameters onto Ollama's ``options`` block."""
    options: dict[str, Any] = {}
    if request.temperature is not None:
        options["temperature"] = request.temperature
    if request.top_p is not None:
        options["top_p"] = request.top_p
    if request.max_tokens is not None:
        options["num_predict"] = request.max_tokens
    if request.seed is not None:
        options["seed"] = request.seed
    if request.stop is not None:
        stop = [request.stop] if isinstance(request.stop, str) else list(request.stop)
        options["stop"] = stop
    return options


def to_ollama_chat_request(
    request: ChatCompletionRequest, default_model: str
) -> dict[str, Any]:
    """Build the full Ollama ``/api/chat`` payload."""
    payload: dict[str, Any] = {
        "model": request.model or default_model,
        "messages": to_ollama_messages(request.messages),
        "stream": request.stream,
    }
    options = to_ollama_options(request)
    if options:
        payload["options"] = options
    if request.tools:
        payload["tools"] = [tool.model_dump(exclude_none=True) for tool in request.tools]
        if request.tool_choice is not None:
            # Ollama has no tool_choice knob; forwarded for forward compatibility.
            payload["tool_choice"] = request.tool_choice
    fmt = to_ollama_format(request)
    if fmt is not None:
        payload["format"] = fmt
    return payload


def to_ollama_embeddings_request(
    model: str, text_input: str | list[str]
) -> dict[str, Any]:
    """Build the Ollama ``/api/embed`` payload."""
    return {"model": model, "input": text_input}


# --------------------------------------------------------------------------- #
# Ollama -> OpenAI
# --------------------------------------------------------------------------- #
def tool_calls_from_ollama(
    raw_calls: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Convert Ollama tool calls into OpenAI shape.

    Ollama returns ``arguments`` as an object; OpenAI requires a JSON-encoded
    string, so it is dumped back out here.

    Input:  [{"function": {"name": "get_weather", "arguments": {"city": "Pune"}}}]
    Output: [{"id": "call_...", "type": "function", "function":
              {"name": "get_weather", "arguments": '{"city":"Pune"}'}}]
    """
    calls: list[dict[str, Any]] = []
    for raw in raw_calls or []:
        function = raw.get("function", raw) or {}
        arguments = function.get("arguments", {})
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, separators=(",", ":"))
        calls.append(
            {
                "id": raw.get("id") or new_tool_call_id(),
                "type": "function",
                "function": {"name": function.get("name", ""), "arguments": arguments},
            }
        )
    return calls


def finish_reason_from_ollama(data: dict[str, Any], has_tool_calls: bool) -> str:
    """Map Ollama's ``done_reason`` onto the contract's three finish reasons."""
    if has_tool_calls:
        return "tool_calls"
    reason = str(data.get("done_reason") or "stop").lower()
    if reason in {"length", "limit"}:
        return "length"
    return "stop"


def usage_from_ollama(data: dict[str, Any]) -> Usage:
    """Ollama's eval counters mapped onto OpenAI usage (0 when absent)."""
    prompt = int(data.get("prompt_eval_count") or 0)
    completion = int(data.get("eval_count") or 0)
    return Usage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
    )


def from_ollama_chat_response(
    data: dict[str, Any],
    *,
    model: str,
    completion_id: str | None = None,
    created: int | None = None,
) -> ChatCompletionResponse:
    """Build the non-streaming ``chat.completion`` body."""
    message = data.get("message") or {}
    tool_calls = tool_calls_from_ollama(message.get("tool_calls"))
    content = message.get("content") or None
    return ChatCompletionResponse(
        id=completion_id or new_completion_id(),
        created=created or now(),
        model=data.get("model") or model,
        choices=[
            Choice(
                index=0,
                message=ResponseMessage(
                    role="assistant",
                    content=None if (tool_calls and not content) else (content or ""),
                    tool_calls=tool_calls or None,
                ),
                finish_reason=finish_reason_from_ollama(data, bool(tool_calls)),
            )
        ],
        usage=usage_from_ollama(data),
    )


def chunk(
    *,
    completion_id: str,
    created: int,
    model: str,
    delta: ChunkDelta | None = None,
    finish_reason: str | None = None,
) -> dict[str, Any]:
    """One ``chat.completion.chunk`` dict.

    ``delta`` is dumped with ``exclude_none=True`` so an empty delta serializes to
    ``{}``, while ``finish_reason`` is always present (null until the last chunk).
    """
    choice = ChunkChoice(
        index=0,
        delta=(delta or ChunkDelta()).model_dump(exclude_none=True),
        finish_reason=finish_reason,
    )
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [choice.model_dump()],
    }


def from_ollama_embeddings(data: dict[str, Any], *, model: str) -> EmbeddingsResponse:
    """Normalize /api/embed (``embeddings``) and legacy /api/embeddings (``embedding``)."""
    vectors = data.get("embeddings")
    if vectors is None:
        single = data.get("embedding")
        vectors = [single] if single is not None else []
    items = [
        {"object": "embedding", "index": index, "embedding": list(vector)}
        for index, vector in enumerate(vectors)
    ]
    prompt_tokens = int(data.get("prompt_eval_count") or 0)
    return EmbeddingsResponse.model_validate(
        {
            "object": "list",
            "data": items,
            "model": data.get("model") or model,
            "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
        }
    )


def from_ollama_tags(data: dict[str, Any]) -> ModelList:
    """Map ``/api/tags`` onto the OpenAI model list."""
    models: list[Model] = []
    for entry in data.get("models") or []:
        name = entry.get("model") or entry.get("name")
        if not name:
            continue
        models.append(Model(id=name, object="model", created=now(), owned_by="ollama"))
    return ModelList(object="list", data=models)
