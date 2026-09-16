"""Pydantic v2 models for everything that crosses the wire.

The models are deliberately permissive (``extra="allow"``, most fields
optional): the server may grow fields, and a strict client would break on a
harmless addition. When you need the body exactly as it arrived, every
top-level response keeps it in ``.raw``.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["system", "user", "assistant", "tool"]

_MODEL_CONFIG = ConfigDict(extra="allow", populate_by_name=True)


class FunctionCall(BaseModel):
    """The ``function`` half of a tool call.

    ``arguments`` arrives as a **JSON-encoded string**, not an object
    (CONTRACT section 4). Use :attr:`ToolCall.arguments` to get a dict.
    """

    model_config = _MODEL_CONFIG

    name: str
    arguments: str = "{}"


class ToolCall(BaseModel):
    """One tool invocation requested by the model."""

    model_config = _MODEL_CONFIG

    id: str
    type: str = "function"
    function: FunctionCall

    @property
    def name(self) -> str:
        """Name of the function the model wants called."""
        return self.function.name

    @property
    def arguments(self) -> dict[str, Any]:
        """``function.arguments`` decoded from its JSON string.

        Input:  arguments == '{"city": "Pune"}'
        Output: {"city": "Pune"}

        Raises:
            ValueError: if the model emitted arguments that are not valid JSON.
        """
        raw = self.function.arguments or "{}"
        if isinstance(raw, dict):  # tolerate a server that pre-decodes
            return raw
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "tool call {id} for {name!r} carried arguments that are not "
                "valid JSON: {raw!r}".format(id=self.id, name=self.function.name, raw=raw)
            ) from exc
        if not isinstance(decoded, dict):
            raise ValueError(
                "tool call {id} for {name!r} decoded to {kind}, expected a JSON object".format(
                    id=self.id, name=self.function.name, kind=type(decoded).__name__
                )
            )
        return decoded


class Message(BaseModel):
    """A single chat message.

    ``tool`` messages must carry ``tool_call_id`` (CONTRACT section 3).
    """

    model_config = _MODEL_CONFIG

    role: Role
    content: str | None = None
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None

    def to_wire(self) -> dict[str, Any]:
        """Serialise for the request body, dropping keys that are None."""
        return self.model_dump(exclude_none=True)


class Usage(BaseModel):
    """Token accounting. The server may report zeros; that is not an error."""

    model_config = _MODEL_CONFIG

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class Choice(BaseModel):
    """One completion candidate from a non-streaming response."""

    model_config = _MODEL_CONFIG

    index: int = 0
    message: Message = Field(default_factory=lambda: Message(role="assistant", content=None))
    finish_reason: str | None = None


class ChatResponse(BaseModel):
    """A ``chat.completion`` response, plus conveniences.

    Input:  the JSON body of POST /v1/chat/completions
    Output: ``resp.text`` -> "Hello", ``resp.finish_reason`` -> "stop"
    """

    model_config = _MODEL_CONFIG

    id: str = ""
    object: str = "chat.completion"
    created: int = 0
    model: str = ""
    choices: list[Choice] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    raw: dict[str, Any] = Field(default_factory=dict, repr=False)

    @property
    def message(self) -> Message | None:
        """The first choice's assistant message, or None if there is none."""
        return self.choices[0].message if self.choices else None

    @property
    def text(self) -> str:
        """Content of the first choice, or "" when the model only called tools."""
        msg = self.message
        return (msg.content if msg and msg.content else "") or ""

    @property
    def tool_calls(self) -> list[ToolCall]:
        """Tool calls on the first choice; empty list when there are none."""
        msg = self.message
        return list(msg.tool_calls) if msg and msg.tool_calls else []

    @property
    def finish_reason(self) -> str | None:
        """One of "stop", "length", "tool_calls" (CONTRACT section 3)."""
        return self.choices[0].finish_reason if self.choices else None


class Delta(BaseModel):
    """The incremental payload inside a streaming choice."""

    model_config = _MODEL_CONFIG

    role: str | None = None
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class ChunkChoice(BaseModel):
    """One choice inside a streaming chunk."""

    model_config = _MODEL_CONFIG

    index: int = 0
    delta: Delta = Field(default_factory=Delta)
    finish_reason: str | None = None


class Chunk(BaseModel):
    """A ``chat.completion.chunk`` SSE event.

    Input:  {"choices": [{"delta": {"content": "Hel"}}]}
    Output: ``chunk.text`` -> "Hel"
    """

    model_config = _MODEL_CONFIG

    id: str = ""
    object: str = "chat.completion.chunk"
    created: int = 0
    model: str = ""
    choices: list[ChunkChoice] = Field(default_factory=list)
    usage: Usage | None = None
    raw: dict[str, Any] = Field(default_factory=dict, repr=False)

    @property
    def text(self) -> str:
        """Delta content of the first choice, or "" for a non-text chunk."""
        if not self.choices:
            return ""
        return self.choices[0].delta.content or ""

    @property
    def finish_reason(self) -> str | None:
        """Set only on the final content chunk."""
        return self.choices[0].finish_reason if self.choices else None

    @property
    def tool_calls(self) -> list[dict[str, Any]]:
        """Raw streaming tool-call fragments (they arrive partial, keyed by index)."""
        if not self.choices:
            return []
        return self.choices[0].delta.tool_calls or []


class Model(BaseModel):
    """An entry from GET /v1/models."""

    model_config = _MODEL_CONFIG

    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = ""


class Health(BaseModel):
    """The GET /healthz probe body (CONTRACT section 3).

    /healthz returns 200 even when Ollama is down, so check :attr:`ok`
    rather than the status code.
    """

    model_config = _MODEL_CONFIG

    status: str = "unknown"
    ollama: str = "unknown"
    model: str = ""
    version: str = ""

    @property
    def ok(self) -> bool:
        """True only when the server is up *and* Ollama is reachable."""
        return self.status == "ok" and self.ollama == "up"


#: Anything accepted where messages are expected.
MessageInput = Sequence["Mapping[str, Any] | Message"]


def normalize_messages(messages: MessageInput | Iterable[Any]) -> list[dict[str, Any]]:
    """Coerce messages to plain wire dicts.

    Accepts :class:`Message` instances, plain dicts, or a mix.

    Input:  [Message(role="user", content="hi"), {"role": "assistant", "content": "yo"}]
    Output: [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]

    Raises:
        TypeError: if ``messages`` is a bare string or holds unsupported items.
    """
    if isinstance(messages, (str, bytes)):
        raise TypeError(
            "messages must be a list of message dicts, not a bare string; "
            'try [{"role": "user", "content": "..."}]'
        )
    out: list[dict[str, Any]] = []
    for item in messages:
        if isinstance(item, Message):
            out.append(item.to_wire())
        elif isinstance(item, BaseModel):
            out.append(item.model_dump(exclude_none=True))
        elif isinstance(item, Mapping):
            if "role" not in item:
                raise TypeError("message is missing a 'role' key: {0!r}".format(item))
            out.append({k: v for k, v in dict(item).items() if v is not None})
        else:
            raise TypeError(
                "unsupported message type {0}: {1!r}".format(type(item).__name__, item)
            )
    return out


__all__ = [
    "Role",
    "FunctionCall",
    "ToolCall",
    "Message",
    "Usage",
    "Choice",
    "ChatResponse",
    "Delta",
    "ChunkChoice",
    "Chunk",
    "Model",
    "Health",
    "MessageInput",
    "normalize_messages",
]
