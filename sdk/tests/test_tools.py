"""function_schema derivation, manual tool handling, and the run_tools loop."""

from __future__ import annotations

import json
from typing import Literal

import httpx
import pytest
from pydantic import BaseModel

from localsdk import MaxRoundsExceeded, ToolExecutionError, function_schema
from localsdk.tools import (
    append_tool_results,
    build_tool_registry,
    execute_tool_call,
    parse_docstring,
)
from localsdk.types import ChatResponse, ToolCall

from conftest import chat_body, make_async_client, make_client

MESSAGES = [{"role": "user", "content": "weather in Pune?"}]


class Filters(BaseModel):
    """Search filters."""

    since: str
    limit: int = 10


def get_weather(city: str, units: str = "c") -> str:
    """Look up the current weather.

    Args:
        city: City to look up.
        units: Either c or f.
    """
    return "{0} is 30{1}".format(city, units)


def search(query: str, tags: list[str], filters: Filters, mode: Literal["fast", "deep"] = "fast"):
    """Search the index."""
    return {"query": query, "tags": tags, "mode": mode}


def no_args():
    """Takes nothing."""
    return "ok"


# --------------------------------------------------------------------------- #
# schema derivation
# --------------------------------------------------------------------------- #


def test_function_schema_shape_matches_the_contract():
    schema = function_schema(get_weather)

    assert schema["type"] == "function"
    fn = schema["function"]
    assert fn["name"] == "get_weather"
    assert fn["description"] == "Look up the current weather."
    assert fn["parameters"]["type"] == "object"
    assert fn["parameters"]["required"] == ["city"]


def test_parameter_types_and_docstrings_are_derived():
    props = function_schema(get_weather)["function"]["parameters"]["properties"]

    assert props["city"] == {"type": "string", "description": "City to look up."}
    assert props["units"]["type"] == "string"
    assert props["units"]["default"] == "c"
    assert props["units"]["description"] == "Either c or f."


def test_complex_annotations():
    props = function_schema(search)["function"]["parameters"]["properties"]

    assert props["tags"] == {"type": "array", "items": {"type": "string"}}
    assert props["filters"]["type"] == "object"
    assert "since" in props["filters"]["properties"]
    assert props["mode"]["enum"] == ["fast", "deep"]
    assert props["mode"]["type"] == "string"
    assert function_schema(search)["function"]["parameters"]["required"] == [
        "query",
        "tags",
        "filters",
    ]


def test_primitive_and_optional_annotations():
    def fn(a: int, b: float, c: bool, d: dict, e: str | None = None):
        """Types."""

    props = function_schema(fn)["function"]["parameters"]["properties"]
    assert props["a"]["type"] == "integer"
    assert props["b"]["type"] == "number"
    assert props["c"]["type"] == "boolean"
    assert props["d"]["type"] == "object"
    assert props["e"]["type"] == ["string", "null"]


def test_unannotated_parameters_produce_an_open_schema():
    def fn(anything):
        """No hints."""

    props = function_schema(fn)["function"]["parameters"]["properties"]
    assert props["anything"] == {}


def test_zero_argument_tool():
    schema = function_schema(no_args)["function"]
    assert schema["parameters"] == {"type": "object", "properties": {}, "required": []}


def test_docstring_parsing_handles_rest_style():
    description, params = parse_docstring("Do a thing.\n\n:param x: the x value\n")
    assert description == "Do a thing."
    assert params == {"x": "the x value"}


def test_duplicate_tool_names_are_rejected():
    with pytest.raises(ValueError):
        build_tool_registry([get_weather, get_weather])


# --------------------------------------------------------------------------- #
# executing a call -- arguments arrive as a JSON string
# --------------------------------------------------------------------------- #


def make_tool_call(name: str = "get_weather", arguments: str = '{"city": "Pune"}') -> ToolCall:
    return ToolCall.model_validate(
        {"id": "call_abc", "type": "function", "function": {"name": name, "arguments": arguments}}
    )


def test_arguments_are_json_decoded_before_the_call():
    call = make_tool_call()
    assert call.arguments == {"city": "Pune"}
    assert execute_tool_call(call, {"get_weather": get_weather}) == "Pune is 30c"


def test_non_string_results_are_serialised():
    def stats() -> dict:
        """Stats."""
        return {"n": 2}

    assert execute_tool_call(make_tool_call("stats", "{}"), {"stats": stats}) == '{"n": 2}'


def test_unknown_tool_raises_tool_execution_error():
    with pytest.raises(ToolExecutionError) as excinfo:
        execute_tool_call(make_tool_call("nope", "{}"), {"get_weather": get_weather})
    assert "nope" in str(excinfo.value)
    assert excinfo.value.tool_call_id == "call_abc"


def test_unparseable_arguments_raise_tool_execution_error():
    with pytest.raises(ToolExecutionError):
        execute_tool_call(make_tool_call(arguments="{not json"), {"get_weather": get_weather})


def test_bad_arguments_for_the_signature_raise_tool_execution_error():
    with pytest.raises(ToolExecutionError):
        execute_tool_call(
            make_tool_call(arguments='{"town": "Pune"}'), {"get_weather": get_weather}
        )


def test_tool_exceptions_are_wrapped():
    def boom(x: int):
        """Explodes."""
        raise RuntimeError("nope")

    with pytest.raises(ToolExecutionError) as excinfo:
        execute_tool_call(make_tool_call("boom", '{"x": 1}'), {"boom": boom})
    assert isinstance(excinfo.value.cause, RuntimeError)


# --------------------------------------------------------------------------- #
# manual handling stays possible without run_tools
# --------------------------------------------------------------------------- #


TOOL_CALL_RESPONSE = chat_body(
    None,
    finish_reason="tool_calls",
    tool_calls=[
        {
            "id": "call_abc",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "Pune"}'},
        }
    ],
)


def test_manual_tool_loop_without_run_tools():
    response = ChatResponse.model_validate({**TOOL_CALL_RESPONSE, "raw": TOOL_CALL_RESPONSE})
    messages = list(MESSAGES)

    append_tool_results(messages, response, {"get_weather": get_weather})

    assert messages[1]["role"] == "assistant"
    assert messages[1]["tool_calls"][0]["id"] == "call_abc"
    assert messages[2] == {
        "role": "tool",
        "tool_call_id": "call_abc",
        "content": "Pune is 30c",
    }


# --------------------------------------------------------------------------- #
# the opt-in auto loop
# --------------------------------------------------------------------------- #


def scripted_handler(bodies, seen: list[httpx.Request]):
    queue = list(bodies)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        body = queue[min(len(seen) - 1, len(queue) - 1)]
        return httpx.Response(200, json=body)

    return handler


def test_run_tools_executes_and_finishes():
    seen: list[httpx.Request] = []
    client = make_client(
        scripted_handler([TOOL_CALL_RESPONSE, chat_body("It is 30c in Pune.")], seen)
    )
    try:
        response = client.run_tools(MESSAGES, [get_weather])
    finally:
        client.close()

    assert response.text == "It is 30c in Pune."
    assert len(seen) == 2

    first = json.loads(seen[0].content)
    assert first["tools"][0]["function"]["name"] == "get_weather"

    second = json.loads(seen[1].content)
    assert [m["role"] for m in second["messages"]] == ["user", "assistant", "tool"]
    assert second["messages"][2]["content"] == "Pune is 30c"
    assert second["messages"][2]["tool_call_id"] == "call_abc"


def test_run_tools_raises_when_the_cap_is_hit():
    seen: list[httpx.Request] = []
    client = make_client(scripted_handler([TOOL_CALL_RESPONSE], seen))
    try:
        with pytest.raises(MaxRoundsExceeded) as excinfo:
            client.run_tools(MESSAGES, [get_weather], max_rounds=2)
    finally:
        client.close()

    assert len(seen) == 2
    assert excinfo.value.rounds == 2
    assert "max_rounds" in str(excinfo.value)
    assert [m["role"] for m in excinfo.value.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
    ]


def test_run_tools_rejects_a_zero_cap():
    client = make_client(scripted_handler([TOOL_CALL_RESPONSE], []))
    try:
        with pytest.raises(ValueError):
            client.run_tools(MESSAGES, [get_weather], max_rounds=0)
    finally:
        client.close()


def test_run_tools_returns_immediately_when_no_tools_are_called():
    seen: list[httpx.Request] = []
    client = make_client(scripted_handler([chat_body("no tools needed")], seen))
    try:
        assert client.run_tools(MESSAGES, [get_weather]).text == "no tools needed"
    finally:
        client.close()
    assert len(seen) == 1


async def test_async_run_tools():
    seen: list[httpx.Request] = []
    async with make_async_client(
        scripted_handler([TOOL_CALL_RESPONSE, chat_body("done")], seen)
    ) as client:
        response = await client.run_tools(MESSAGES, [get_weather])
    assert response.text == "done"
    assert len(seen) == 2


async def test_async_run_tools_cap():
    async with make_async_client(scripted_handler([TOOL_CALL_RESPONSE], [])) as client:
        with pytest.raises(MaxRoundsExceeded):
            await client.run_tools(MESSAGES, [get_weather], max_rounds=1)
