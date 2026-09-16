"""Turn plain Python callables into OpenAI tool schemas, and run them.

Two halves:

* :func:`function_schema` -- derive the JSON schema the model needs from a
  function's signature, type hints and docstring. Nothing to register, no
  decorator, no DSL.
* :func:`execute_tool_call` / :func:`append_tool_results` -- the mechanics of
  running what the model asked for. ``run_tools()`` on the clients is a thin
  loop over these; you can equally drive them yourself.

Remember: ``function.arguments`` arrives as a **JSON-encoded string**
(CONTRACT section 4). Everything here decodes it for you.
"""

from __future__ import annotations

import enum
import inspect
import json
import types
import typing
from typing import Any, Callable, Iterable, Literal, Mapping, Union, get_args, get_origin

from pydantic import BaseModel

from .errors import ToolExecutionError
from .types import ChatResponse, ToolCall

#: Tools are plain callables. Their ``__name__`` is the name the model sees.
Tool = Callable[..., Any]

_PRIMITIVES: dict[Any, dict[str, Any]] = {
    str: {"type": "string"},
    int: {"type": "integer"},
    float: {"type": "number"},
    bool: {"type": "boolean"},
    list: {"type": "array"},
    dict: {"type": "object"},
    type(None): {"type": "null"},
}


#: ``X | None`` produces ``types.UnionType`` while ``Union[X, None]`` produces
#: ``typing.Union``; both must be recognised.
_UNION_ORIGINS: tuple[Any, ...] = (Union, getattr(types, "UnionType", Union))


def _is_union(origin: Any) -> bool:
    return any(origin is candidate for candidate in _UNION_ORIGINS)


def _is_optional(annotation: Any) -> bool:
    return _is_union(get_origin(annotation)) and type(None) in get_args(annotation)


def _strip_optional(annotation: Any) -> Any:
    """``str | None`` -> ``str``; leaves everything else alone."""
    if _is_union(get_origin(annotation)):
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def json_schema_for_annotation(annotation: Any) -> dict[str, Any]:
    """Map a Python type hint onto a JSON-schema fragment.

    Supports str/int/float/bool/list/dict, ``list[T]``/``dict[str, T]``,
    ``Literal[...]``, ``enum.Enum`` subclasses, pydantic ``BaseModel``
    subclasses, unions, and bare/unknown hints (which become an open schema).

    Input:  list[int]              Output: {"type": "array", "items": {"type": "integer"}}
    Input:  Literal["a", "b"]      Output: {"type": "string", "enum": ["a", "b"]}
    """
    if annotation is inspect.Parameter.empty or annotation is Any or annotation is None:
        return {}

    if _is_optional(annotation):
        inner = json_schema_for_annotation(_strip_optional(annotation))
        # JSON schema has no "optional"; nullability is expressed in the type.
        if "type" in inner and isinstance(inner["type"], str):
            return {**inner, "type": [inner["type"], "null"]}
        return inner

    origin = get_origin(annotation)

    if origin is Literal:
        values = list(get_args(annotation))
        schema: dict[str, Any] = {"enum": values}
        kinds = {type(v) for v in values}
        if len(kinds) == 1:
            primitive = _PRIMITIVES.get(kinds.pop())
            if primitive:
                schema = {**primitive, **schema}
        return schema

    if _is_union(origin):
        variants = [json_schema_for_annotation(a) for a in get_args(annotation)]
        return {"anyOf": [v for v in variants if v]}

    if origin in (list, set, tuple, frozenset) or annotation in (list, set, tuple):
        args = get_args(annotation)
        if args and args[0] is not Ellipsis:
            return {"type": "array", "items": json_schema_for_annotation(args[0])}
        return {"type": "array"}

    if origin is dict or annotation is dict:
        args = get_args(annotation)
        if len(args) == 2:
            value_schema = json_schema_for_annotation(args[1])
            if value_schema:
                return {"type": "object", "additionalProperties": value_schema}
        return {"type": "object"}

    if isinstance(annotation, type):
        if issubclass(annotation, BaseModel):
            return annotation.model_json_schema()
        if issubclass(annotation, enum.Enum):
            values = [member.value for member in annotation]
            base = _PRIMITIVES.get(type(values[0])) if values else None
            return {**(base or {}), "enum": values}
        if annotation in _PRIMITIVES:
            return dict(_PRIMITIVES[annotation])
        if issubclass(annotation, bool):
            return {"type": "boolean"}

    return {}


def parse_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    """Split a docstring into a one-line description and per-parameter text.

    Understands Google style (``Args:``) and reST style (``:param x:``).

    Input:  "Get weather.\\n\\nArgs:\\n    city: City name."
    Output: ("Get weather.", {"city": "City name."})
    """
    if not doc:
        return "", {}
    cleaned = inspect.cleandoc(doc)
    lines = cleaned.splitlines()
    description = lines[0].strip() if lines else ""

    params: dict[str, str] = {}
    in_args = False
    current: str | None = None
    for raw in lines[1:]:
        line = raw.strip()
        lowered = line.lower().rstrip(":")
        if line.endswith(":") and lowered in {"args", "arguments", "parameters", "params"}:
            in_args = True
            current = None
            continue
        if line.endswith(":") and lowered in {"returns", "return", "raises", "yields", "examples", "example", "note", "notes"}:
            in_args = False
            current = None
            continue
        if line.startswith(":param "):
            name, _, text = line[len(":param ") :].partition(":")
            params[name.strip()] = text.strip()
            current = name.strip()
            continue
        if in_args and line:
            if ":" in line and not line.startswith(" "):
                name, _, text = line.partition(":")
                name = name.strip()
                # "city (str): ..." -> "city"
                name = name.split("(")[0].strip()
                if name:
                    params[name] = text.strip()
                    current = name
                    continue
            if current:
                params[current] = (params[current] + " " + line).strip()
    return description, params


def function_schema(fn: Tool, *, name: str | None = None, description: str | None = None) -> dict[str, Any]:
    """Derive an OpenAI tool definition from a Python callable.

    The returned dict is ready to hand to ``chat(..., tools=[...])``: it is the
    full ``{"type": "function", "function": {...}}`` envelope from CONTRACT
    section 4, not just the inner function object.

    Input::

        def get_weather(city: str, units: str = "c") -> str:
            '''Look up the weather.

            Args:
                city: City to look up.
            '''

    Output::

        {"type": "function", "function": {
            "name": "get_weather",
            "description": "Look up the weather.",
            "parameters": {"type": "object",
                           "properties": {"city": {"type": "string",
                                                   "description": "City to look up."},
                                          "units": {"type": "string", "default": "c"}},
                           "required": ["city"]}}}

    Raises:
        TypeError: if ``fn`` is not callable.
    """
    if not callable(fn):
        raise TypeError("tools must be callables, got {0}".format(type(fn).__name__))

    signature = inspect.signature(fn)
    try:
        hints = typing.get_type_hints(fn)
    except Exception:  # unresolvable forward refs should not be fatal
        hints = getattr(fn, "__annotations__", {}) or {}

    doc_description, param_docs = parse_docstring(inspect.getdoc(fn))

    properties: dict[str, Any] = {}
    required: list[str] = []
    for param_name, param in signature.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue  # *args/**kwargs cannot be described to the model
        if param_name in {"self", "cls"}:
            continue
        annotation = hints.get(param_name, param.annotation)
        schema = json_schema_for_annotation(annotation)
        if param_name in param_docs:
            schema["description"] = param_docs[param_name]
        if param.default is not inspect.Parameter.empty:
            if _is_jsonable(param.default):
                schema["default"] = param.default
        else:
            required.append(param_name)
        properties[param_name] = schema

    return {
        "type": "function",
        "function": {
            "name": name or getattr(fn, "__name__", "tool"),
            "description": description or doc_description or "",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def _is_jsonable(value: Any) -> bool:
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False


def build_tool_registry(tools: Iterable[Tool]) -> dict[str, Tool]:
    """Index callables by the name the model will use.

    Input:  [get_weather]        Output: {"get_weather": <function get_weather>}

    Raises:
        ValueError: on duplicate tool names -- the model could not tell them apart.
    """
    registry: dict[str, Tool] = {}
    for fn in tools:
        if not callable(fn):
            raise TypeError("tools must be callables, got {0}".format(type(fn).__name__))
        key = getattr(fn, "__name__", None)
        if not key:
            raise ValueError("tool {0!r} has no __name__; pass a named function".format(fn))
        if key in registry:
            raise ValueError("duplicate tool name {0!r}".format(key))
        registry[key] = fn
    return registry


def stringify_tool_result(value: Any) -> str:
    """Render a tool's return value as the string the ``tool`` message carries.

    Input:  {"temp": 30}   Output: '{"temp": 30}'
    Input:  "sunny"        Output: 'sunny'
    """
    if isinstance(value, str):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump_json()
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return str(value)


def execute_tool_call(tool_call: ToolCall, registry: Mapping[str, Tool]) -> str:
    """Run one tool call and return its result as a string.

    Raises:
        ToolExecutionError: unknown tool name, unparseable arguments, arguments
            that do not fit the signature, or an exception from the tool itself.
    """
    name = tool_call.function.name
    fn = registry.get(name)
    if fn is None:
        raise ToolExecutionError(
            "the model asked for tool {0!r}, which was not supplied. Known tools: {1}".format(
                name, ", ".join(sorted(registry)) or "(none)"
            ),
            tool_name=name,
            tool_call_id=tool_call.id,
        )
    try:
        arguments = tool_call.arguments  # JSON string -> dict (CONTRACT section 4)
    except ValueError as exc:
        raise ToolExecutionError(
            str(exc), tool_name=name, tool_call_id=tool_call.id, cause=exc
        ) from exc
    try:
        result = fn(**arguments)
    except TypeError as exc:
        raise ToolExecutionError(
            "tool {0!r} rejected the arguments the model produced ({1!r}): {2}".format(
                name, arguments, exc
            ),
            tool_name=name,
            tool_call_id=tool_call.id,
            cause=exc,
        ) from exc
    except Exception as exc:
        raise ToolExecutionError(
            "tool {0!r} raised {1}: {2}".format(name, type(exc).__name__, exc),
            tool_name=name,
            tool_call_id=tool_call.id,
            cause=exc,
        ) from exc
    if inspect.isawaitable(result):
        raise ToolExecutionError(
            "tool {0!r} returned an awaitable. run_tools() executes tools "
            "synchronously; wrap async tools in a sync shim.".format(name),
            tool_name=name,
            tool_call_id=tool_call.id,
        )
    return stringify_tool_result(result)


def assistant_message_from(response: ChatResponse) -> dict[str, Any]:
    """The assistant turn to append before tool results, preserving ``tool_calls``."""
    raw_choices = response.raw.get("choices") if isinstance(response.raw, Mapping) else None
    if raw_choices:
        message = raw_choices[0].get("message")
        if isinstance(message, Mapping):
            return {k: v for k, v in dict(message).items() if v is not None}
    message_model = response.message
    return message_model.to_wire() if message_model else {"role": "assistant", "content": ""}


def append_tool_results(
    messages: list[dict[str, Any]],
    response: ChatResponse,
    registry: Mapping[str, Tool],
) -> list[dict[str, Any]]:
    """Append the assistant turn and one ``tool`` message per call, in place.

    This is the whole of manual tool handling; ``run_tools()`` just calls it in
    a loop. Each tool message carries ``tool_call_id`` as the contract requires.
    """
    messages.append(assistant_message_from(response))
    for tool_call in response.tool_calls:
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": execute_tool_call(tool_call, registry),
            }
        )
    return messages


def tool_schemas(tools: Iterable[Any]) -> list[dict[str, Any]]:
    """Normalise a mixed list of callables and raw dicts into tool schemas."""
    out: list[dict[str, Any]] = []
    for item in tools:
        if isinstance(item, Mapping):
            out.append(dict(item))
        else:
            out.append(function_schema(item))
    return out


__all__ = [
    "Tool",
    "function_schema",
    "json_schema_for_annotation",
    "parse_docstring",
    "build_tool_registry",
    "execute_tool_call",
    "append_tool_results",
    "assistant_message_from",
    "stringify_tool_result",
    "tool_schemas",
]
