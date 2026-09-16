# localsdk

A small, typed Python client for a self-hosted, OpenAI-compatible LLM endpoint
(FastAPI in front of Ollama, exposed through ngrok).

Async core, sync mirror, no heavyweight dependencies — just `httpx` and
`pydantic`.

```python
from localsdk import Client

client = Client(api_key="sk-your-own-key", base_url="https://xxxx.ngrok-free.app")
print(client.chat([{"role": "user", "content": "hello"}]).text)
```

---

## Install

```bash
pip install "git+https://github.com/Durgesh63/localsdk.git#subdirectory=sdk"
```

From a checkout:

```bash
cd sdk
pip install -e ".[dev]"
```

Python 3.10+. On 3.10 add the `toml` extra if you want the config file to be
read (`pip install "localsdk[toml]"`); 3.11+ uses the stdlib `tomllib`.

---

## Configure

Precedence, highest first:

```
explicit argument  >  environment variable  >  ~/.localsdk/config.toml  >  default
```

| setting | argument | env var | default |
|---|---|---|---|
| API key | `api_key` | `LOCALSDK_API_KEY` | — (required) |
| Server URL | `base_url` | `LOCALSDK_BASE_URL` | — (required) |
| Model | `model` | `LOCALSDK_MODEL` | `qwen2.5:14b` |
| Timeout (s) | `timeout` | `LOCALSDK_TIMEOUT` | `300` |
| Retries | `max_retries` | `LOCALSDK_MAX_RETRIES` | `0` |

```toml
# ~/.localsdk/config.toml
[default]
api_key  = "sk-your-own-key"
base_url = "https://xxxx.ngrok-free.app"
model    = "qwen2.5:14b"
```

A missing `api_key` or `base_url` raises `ConfigurationError` at construction
time, naming the environment variable to set — you never discover it on the
first request.

> ### ⚠️ `base_url` changes every time free ngrok restarts
>
> A free ngrok tunnel is assigned a **new random URL on every restart**. Keep
> `base_url` in an env var or in `~/.localsdk/config.toml` and update it when
> the tunnel bounces. Never hardcode it, and never commit it.
>
> If the URL is stale you will get an `APIConnectionError` whose message says
> exactly that — see [When it breaks](#when-it-breaks).

---

## Quickstart — the five capabilities

### 1. Chat

```python
from localsdk import Client

with Client() as client:                      # reads env / config file
    resp = client.chat(
        [{"role": "user", "content": "Summarise SSE in one line."}],
        temperature=0.2,
        max_tokens=200,
    )

resp.text           # "Server-sent events are ..."
resp.finish_reason  # "stop" | "length" | "tool_calls"
resp.usage          # Usage(prompt_tokens=..., completion_tokens=..., total_tokens=...)
resp.tool_calls     # [ToolCall, ...]
resp.raw            # the untouched response dict
```

### 2. Streaming

```python
for chunk in client.stream(messages):
    print(chunk.text, end="", flush=True)
```

The response body is consumed incrementally — never buffered whole. Errors
that happen mid-generation arrive as a final SSE event and are raised as the
mapped exception, so a truncated answer is never mistaken for a complete one.

### 3. Tool calling

Schemas are derived from the function itself — signature, type hints and the
first line of the docstring:

```python
def get_weather(city: str, units: str = "c") -> str:
    """Look up the current weather.

    Args:
        city: City to look up.
    """
    return f"{city}: 30{units}"

resp = client.run_tools(messages, [get_weather], max_rounds=3)
print(resp.text)
```

`run_tools()` is **opt-in**: it executes the calls the model asks for, appends
the `role: "tool"` results, and calls back until the model answers — capped by
`max_rounds`, raising `MaxRoundsExceeded` (with the conversation attached)
rather than looping forever.

Manual control is equally supported:

```python
from localsdk import function_schema
from localsdk.tools import append_tool_results, build_tool_registry

tools = [function_schema(get_weather)]
registry = build_tool_registry([get_weather])

resp = client.chat(messages, tools=tools)
while resp.tool_calls:
    for call in resp.tool_calls:
        call.name        # "get_weather"
        call.arguments   # {"city": "Pune"}  <- decoded from the JSON *string*
    append_tool_results(messages, resp, registry)
    resp = client.chat(messages, tools=tools)
```

### 4. Structured output

```python
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

person = client.structured(messages, Person)   # -> Person(name="Ada", age=36)
```

Sends `response_format={"type": "json_schema", ...}` and validates the reply.
Unparseable or non-conforming JSON raises `StructuredOutputError`, with the raw
text on `.text`.

### 5. Embeddings

**Setup first.** Embeddings use a *different model* from chat, and it is a
separate download. On the machine running Ollama:

```bash
ollama pull nomic-embed-text      # ~270 MB
```

Until that exists, `embed()` raises `NotFoundError` (404). Chat, streaming,
tools and structured output are unaffected - they use the chat model. Nothing
needs redeploying afterwards: the server resolves the model per request.

```python
client.embed("one string")        # -> [[0.1, 0.2, ...]]   always a LIST of vectors
client.embed(["a", "b"])          # -> [[...], [...]]
client.embed(texts, model="mxbai-embed-large")   # override per call
```

Embeddings do not generate text. They turn text into a fixed-length vector
whose *distance* to another vector reflects how related the two texts are -
which is what makes semantic search possible.

**Batch your calls.** `embed()` takes a list, and one call with 200 texts costs
one request where a loop would cost 200. That matters on a metered tunnel.
Better still, run an indexing pass *on the server machine itself* against
`http://localhost:8000` so only the per-question embedding crosses the network.

The SDK gives you vectors, not a vector store - see
[examples/05_rag.py](examples/05_rag.py) for the full
chunk to embed to search to answer loop in dependency-free Python.

### 6. Models and health

```python
client.models()                   # [Model(id="qwen2.5:14b", ...)]

health = client.health()          # GET /healthz, no auth
health.ok                         # True only if the server AND ollama are up
```

Because `/healthz` needs no auth, a green health check does **not** mean your
API key is valid.

---

## Async

`AsyncClient` has the same methods with the same signatures; drop the
`await`/`async for` to get the sync version.

```python
from localsdk import AsyncClient

async with AsyncClient() as client:
    resp = await client.chat(messages)
    async for chunk in client.stream(messages):
        ...
```

The sync client is a thin mirror over the same request-building and
response-parsing code — it does **not** spin up an event loop, so it is safe to
call from inside one.

---

## Retries are off by default

`max_retries=0`. Requests are metered on free ngrok and the box runs one
generation at a time, so nothing is ever re-sent behind your back.

When you opt in (`max_retries=2`), only genuinely safe failures are retried —
connect errors, `502` and `504` — with exponential backoff and jitter. Read
timeouts are not retried (the generation already started, and you already paid
for it), and **a stream is never retried mid-flight**.

---

## When it breaks

| Symptom | Exception | What it means |
|---|---|---|
| HTML instead of JSON | `APIConnectionError` | ngrok interstitial, or a stale tunnel URL. The message tells you which and what to run. |
| `401` | `AuthenticationError` | missing/unknown key |
| `403` | `PermissionDeniedError` | key revoked |
| `400` | `BadRequestError` | malformed body |
| `404` | `NotFoundError` | model not present on the server |
| `429` | `RateLimitError` | reserved |
| `502` / `504` | `UpstreamError` | cannot reach Ollama / Ollama timed out |
| no response in time | `APITimeoutError` | raise `timeout=`, or use `stream()` |
| missing config | `ConfigurationError` | raised at construction, names the env var |

Every exception derives from `LocalSDKError`, and every `APIError` exposes
`.status_code`, `.code`, `.type` and `.message`.

The most common real-world failure is the ngrok one. Sanity-check the tunnel
with:

```bash
curl -H "ngrok-skip-browser-warning: true" "$LOCALSDK_BASE_URL/healthz"
```

If that returns HTML, the tunnel URL has rotated — get the current one from the
ngrok console and update `LOCALSDK_BASE_URL`.

---

## Development

```bash
cd sdk
pip install -e ".[dev]"
python -m pytest
```

Tests run entirely against `httpx.MockTransport`: no network, no server, no
Ollama.
