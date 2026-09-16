"""Example 1 -- a single chat completion.

Run:
    export LOCALSDK_API_KEY=sk-your-own-key
    export LOCALSDK_BASE_URL=https://<current>.ngrok-free.app
    python examples/01_chat.py

Nothing is hardcoded: a free ngrok tunnel gets a new URL every restart, so
base_url is configuration.
"""

from localsdk import Client, LocalSDKError


def main() -> None:
    # api_key / base_url fall back to LOCALSDK_API_KEY, LOCALSDK_BASE_URL,
    # then ~/.localsdk/config.toml.
    with Client() as client:
        health = client.health()
        print("server: {0} | ollama: {1} | model: {2}".format(
            health.status, health.ollama, health.model
        ))
        if not health.ok:
            print("warning: ollama is not reachable; the chat below will probably fail")

        print("models:", ", ".join(m.id for m in client.models()))

        response = client.chat(
            [
                {"role": "system", "content": "You are terse."},
                {"role": "user", "content": "Name three uses for a paperclip."},
            ],
            temperature=0.2,
            max_tokens=200,
        )

        print("\n" + response.text)
        print(
            "\nfinish_reason={0} tokens={1}".format(
                response.finish_reason, response.usage.total_tokens
            )
        )


if __name__ == "__main__":
    try:
        main()
    except LocalSDKError as exc:
        # Every failure this SDK raises is a LocalSDKError, and the message
        # says what to do about it.
        raise SystemExit("error: {0}".format(exc))
