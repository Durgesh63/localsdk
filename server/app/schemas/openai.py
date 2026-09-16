"""Request/response models for every endpoint in the contract.

All request models use ``extra="ignore"``: CONTRACT section 3 requires unknown
fields to be accepted silently rather than rejected with a 400.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

FinishReason = Literal["stop", "length", "tool_calls"]

_LENIENT = ConfigDict(extra="ignore", populate_by_name=True, protected_namespaces=())
_STRICT = ConfigDict(protected_namespaces=())


# --------------------------------------------------------------------------- #
# chat
# --------------------------------------------------------------------------- #
class FunctionCall(BaseModel):
    """The function half of a tool call. ``arguments`` is a JSON-encoded string."""

    model_config = _LENIENT

    name: str
    arguments: str = "{}"


class ToolCall(BaseModel):
    model_config = _LENIENT

    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class FunctionDef(BaseModel):
    model_config = _LENIENT

    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None


class Tool(BaseModel):
    model_config = _LENIENT

    type: Literal["function"] = "function"
    function: FunctionDef


class ChatMessage(BaseModel):
    model_config = _LENIENT

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None


class ResponseFormat(BaseModel):
    model_config = _LENIENT

    type: Literal["text", "json_object", "json_schema"] = "text"
    json_schema: dict[str, Any] | None = None


class ChatCompletionRequest(BaseModel):
    model_config = _LENIENT

    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    stop: str | list[str] | None = None
    seed: int | None = None
    tools: list[Tool] | None = None
    tool_choice: str | dict[str, Any] | None = None
    response_format: ResponseFormat | None = None


class ResponseMessage(BaseModel):
    """The assistant message the server returns (no request-only fields)."""

    model_config = _STRICT

    role: Literal["assistant"] = "assistant"
    content: str | None = None
    tool_calls: list[ToolCall] | None = None


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class Choice(BaseModel):
    index: int = 0
    message: ResponseMessage
    finish_reason: FinishReason | None = None


class ChatCompletionResponse(BaseModel):
    model_config = _STRICT

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage = Field(default_factory=Usage)


class ChunkDelta(BaseModel):
    """Streaming delta. Dumped with ``exclude_none=True`` so empty deltas are ``{}``."""

    role: Literal["assistant"] | None = None
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class ChunkChoice(BaseModel):
    index: int = 0
    delta: dict[str, Any] = Field(default_factory=dict)
    finish_reason: FinishReason | None = None


class ChatCompletionChunk(BaseModel):
    model_config = _STRICT

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChunkChoice]


# --------------------------------------------------------------------------- #
# embeddings
# --------------------------------------------------------------------------- #
class EmbeddingsRequest(BaseModel):
    model_config = _LENIENT

    model: str | None = None
    input: str | list[str]


class EmbeddingItem(BaseModel):
    object: Literal["embedding"] = "embedding"
    index: int
    embedding: list[float]


class EmbeddingsUsage(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0


class EmbeddingsResponse(BaseModel):
    model_config = _STRICT

    object: Literal["list"] = "list"
    data: list[EmbeddingItem]
    model: str
    usage: EmbeddingsUsage = Field(default_factory=EmbeddingsUsage)


# --------------------------------------------------------------------------- #
# models / health
# --------------------------------------------------------------------------- #
class Model(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "ollama"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[Model]


class HealthResponse(BaseModel):
    model_config = _STRICT

    status: str = "ok"
    ollama: Literal["up", "down"]
    model: str
    version: str
