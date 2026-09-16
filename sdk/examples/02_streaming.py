"""Example 2 -- streaming, sync and async.

One 14B model on one box means one generation at a time, so long answers feel
much better streamed. The body is consumed incrementally; nothing is buffered.

Run:
    python examples/02_streaming.py
"""

import asyncio

from localsdk import AsyncClient, Client, LocalSDKError

PROMPT = [{"role": "user", "content": "Explain SSE to a backend dev in one paragraph."}]


def sync_stream() -> None:
    print("--- sync ---")
    with Client() as client:
        for chunk in client.stream(PROMPT, max_tokens=300):
            print(chunk.text, end="", flush=True)
            if chunk.finish_reason:
                print("\n[finish_reason={0}]".format(chunk.finish_reason))


async def async_stream() -> None:
    print("\n--- async ---")
    async with AsyncClient() as client:
        async for chunk in client.stream(PROMPT, max_tokens=300):
            print(chunk.text, end="", flush=True)
    print()


if __name__ == "__main__":
    try:
        sync_stream()
        asyncio.run(async_stream())
    except LocalSDKError as exc:
        raise SystemExit("error: {0}".format(exc))
