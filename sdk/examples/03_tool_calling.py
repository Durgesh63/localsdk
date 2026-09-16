"""Example 3 -- tool calling, manual and automatic.

The schema the model sees is derived from the function itself: signature, type
hints and docstring. There is no registry and no decorator.

Run:
    python examples/03_tool_calling.py
"""

from localsdk import Client, MaxRoundsExceeded, LocalSDKError, function_schema
from localsdk.tools import append_tool_results, build_tool_registry

RAINFALL = {"pune": 12, "mumbai": 31, "chennai": 4}


def get_rainfall(city: str, unit: str = "mm") -> str:
    """Get yesterday's rainfall for a city.

    Args:
        city: City name, e.g. "Pune".
        unit: Measurement unit, "mm" or "in".
    """
    value = RAINFALL.get(city.lower(), 0)
    if unit == "in":
        value = round(value / 25.4, 2)
    return "{0}: {1}{2}".format(city, value, unit)


QUESTION = [{"role": "user", "content": "How much rain did Pune get yesterday, in inches?"}]


def show_derived_schema() -> None:
    print("derived schema:")
    print(function_schema(get_rainfall))


def automatic() -> None:
    """The opt-in loop: run_tools() executes the calls and calls back."""
    print("\n--- run_tools (automatic) ---")
    with Client() as client:
        try:
            response = client.run_tools(QUESTION, [get_rainfall], max_rounds=3)
        except MaxRoundsExceeded as exc:
            # The conversation is preserved so you can inspect or resume it.
            print("gave up after {0} rounds; {1} messages so far".format(
                exc.rounds, len(exc.messages)
            ))
            return
        print(response.text)


def manual() -> None:
    """The same thing by hand -- run_tools() is a convenience, not a gate."""
    print("\n--- manual ---")
    tools = [function_schema(get_rainfall)]
    registry = build_tool_registry([get_rainfall])
    messages = list(QUESTION)

    with Client() as client:
        response = client.chat(messages, tools=tools)

        while response.tool_calls:
            for call in response.tool_calls:
                # arguments arrive as a JSON *string*; .arguments decodes it
                print("model wants {0}({1})".format(call.name, call.arguments))
            append_tool_results(messages, response, registry)
            response = client.chat(messages, tools=tools)

        print(response.text)


if __name__ == "__main__":
    try:
        show_derived_schema()
        automatic()
        manual()
    except LocalSDKError as exc:
        raise SystemExit("error: {0}".format(exc))
