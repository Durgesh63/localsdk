# LocalSDK

Talk to a self-hosted Ollama model (`qwen2.5:14b`) the way you would talk to a
hosted provider: `pip install`, hand it an API key and a base URL, call it.

Two halves, one repo:

```
  your app
     │  pip install localsdk
     │
     │  client = Client(api_key="sk-...", base_url="https://xxxx.ngrok-free.app")
     │  client.chat([...])
     ▼
  ┌───────────────┐
  │  ngrok tunnel │           (so the GPU box needs no public IP)
  └──────┬────────┘
         │  Authorization: Bearer sk-...
         ▼
  ┌──────────────────────────┐
  │  server/   FastAPI       │   /v1/chat/completions
  │  API-key auth            │   /v1/embeddings
  │  OpenAI-compatible       │   /v1/models
  └──────┬───────────────────┘   /healthz
         │
         ▼
  Ollama :11434  →  qwen2.5:14b
```

| path | what it is |
| --- | --- |
| [CONTRACT.md](CONTRACT.md) | the HTTP contract between the two halves — **read this before changing either side** |
| [sdk/](sdk/) | the Python client package your apps install ([details](sdk/README.md)) |
| [server/](server/) | the FastAPI gateway that fronts Ollama ([details](server/README.md)) |

`sdk/` and `server/` never import each other. They agree only on `CONTRACT.md`.

---

## Setup

You set up **two roles** (they can be the same machine):

| role | what runs there | who does it |
| --- | --- | --- |
| **A. the GPU box** | Ollama + `server/` + ngrok | once, by whoever owns the hardware |
| **B. the client** | just `pip install localsdk` | every developer / app |

---

### Part A — Server setup (the GPU box)

#### A1. Prerequisites

- **Python 3.10+** — check with `python --version`
- **Ollama** — https://ollama.com/download
- **ngrok** — https://ngrok.com/download (skip if clients are on the same network)

#### A2. Pull the model

```bash
ollama pull qwen2.5:14b          # ~9 GB, this takes a while
ollama list                      # confirm it is there
```

Optional, only if you need `/v1/embeddings` (RAG / semantic search):

```bash
ollama pull nomic-embed-text     # ~270 MB
```

Skip it and chat, streaming, tools and structured output all still work.
`/v1/embeddings` simply returns `404 model_not_found` until you pull it.

Confirm Ollama is serving:

```bash
curl http://localhost:11434/api/tags
```

#### A3. Install the server

```bash
cd server
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

#### A4. Configure

```bash
cp .env.example .env
```

Then edit `.env`:

| variable | default | what it does |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | where Ollama is listening |
| `DEFAULT_MODEL` | `qwen2.5:14b` | chat model used when a request omits `model` |
| `EMBED_MODEL` | `nomic-embed-text` | embedding model used when a request omits `model` |
| `KEYSTORE_BACKEND` | `static` | `static` (env vars, no DB) or `postgres` (not implemented yet) |
| `API_KEYS` | — | **comma-separated list of valid keys.** Change these. |
| `REQUEST_TIMEOUT_S` | `300` | read timeout for a generation. A 14B is slow; do not lower this much |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | where the server listens |

> **Change `API_KEYS` before exposing anything.** Anyone holding a key can spend
> your GPU. Generate real ones with:
> `python -c "import secrets; print('sk-' + secrets.token_urlsafe(32))"`

#### A5. Run it

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Verify locally, in a second terminal:

```bash
curl http://localhost:8000/healthz
# {"status":"ok","ollama":"up","model":"qwen2.5:14b","version":"0.1.0"}

curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-local-dev-001" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"say hi"}]}'
```

If `healthz` reports `"ollama":"down"`, the server is fine and Ollama is not —
check `OLLAMA_BASE_URL` and that Ollama is actually running.

#### A6. Expose it with ngrok

```bash
ngrok http 8000
```

ngrok prints a line like:

```
Forwarding   https://a1b2-49-36-x-x.ngrok-free.app -> http://localhost:8000
```

**That URL is your `base_url`.** Verify it end to end:

```bash
curl https://a1b2-49-36-x-x.ngrok-free.app/healthz \
  -H "ngrok-skip-browser-warning: true"
```

That header matters — without it, free ngrok returns an HTML warning page
instead of your JSON. The SDK always sends it for you; `curl` does not.

---

### Part B — Client setup (every app / developer)

#### B1. Install

```bash
pip install "git+https://github.com/Durgesh63/localsdk.git#subdirectory=sdk"
```

On Python 3.10, add the `toml` extra if you want config-file support:

```bash
pip install "localsdk[toml] @ git+https://github.com/Durgesh63/localsdk.git#subdirectory=sdk"
```

#### B2. Configure — pick one of three

Resolution order is **explicit argument → environment variable → config file → default**.

**1. In code** (simplest, fine for scripts):

```python
from localsdk import Client

client = Client(api_key="sk-local-dev-001", base_url="https://a1b2.ngrok-free.app")
```

**2. Environment variables** (best for servers and containers):

```bash
export LOCALSDK_API_KEY=sk-local-dev-001
export LOCALSDK_BASE_URL=https://a1b2.ngrok-free.app
export LOCALSDK_MODEL=qwen2.5:14b        # optional
export LOCALSDK_TIMEOUT=300              # optional
export LOCALSDK_MAX_RETRIES=0            # optional
```

```python
client = Client()          # picks everything up from the environment
```

**3. Config file** at `~/.localsdk/config.toml` (best for a developer laptop):

```toml
[default]
api_key  = "sk-local-dev-001"
base_url = "https://a1b2.ngrok-free.app"
model    = "qwen2.5:14b"
```

#### B3. Verify

```python
from localsdk import Client

client = Client()
print(client.health())                  # Health(status='ok', ollama='up', ...)
print([m.id for m in client.models()])  # ['qwen2.5:14b']
print(client.chat([{"role": "user", "content": "hello"}]).text)
```

---

## The full workflow

All five capabilities. Runnable versions live in [sdk/examples/](sdk/examples/).

### 1. Chat

The SDK is **stateless** — your app owns the conversation history. To continue a
conversation, append to the list and send it again.

```python
messages = [
    {"role": "system", "content": "You are terse."},
    {"role": "user",   "content": "What is Pune known for?"},
]
reply = client.chat(messages)
print(reply.text)

messages.append({"role": "assistant", "content": reply.text})
messages.append({"role": "user", "content": "And its weather?"})
print(client.chat(messages).text)
```

`ChatResponse` gives you `.text`, `.tool_calls`, `.finish_reason`, `.usage`, and
`.raw` for the untouched response dict.

### 2. Streaming

Use this for anything a human waits on. A 14B can take tens of seconds to
finish; streaming returns the first token almost immediately.

```python
for chunk in client.stream(messages):
    print(chunk.text, end="", flush=True)
```

### 3. Tool calling — the agent part

**Manual (recommended).** Your app stays in control of every model call, which
matters because each round trip costs one metered ngrok request.

```python
from localsdk import function_schema, append_tool_results

def get_order_status(order_id: str) -> str:
    """Look up the delivery status of an order."""
    return db.lookup(order_id)

reply = client.chat(messages, tools=[function_schema(get_order_status)])

if reply.finish_reason == "tool_calls":
    for call in reply.tool_calls:
        print(call.name, call.arguments)   # arguments is already a dict
        result = get_order_status(**call.arguments)
        messages = append_tool_results(messages, reply, {call.id: result})
    reply = client.chat(messages)          # second call, now with the tool result

print(reply.text)
```

**Automatic (opt-in).** The same thing, looped for you. Deliberately not the
default — it hides how many requests you are spending.

```python
reply = client.run_tools(messages, tools=[get_order_status], max_rounds=3)
print(reply.text)
```

`max_rounds` is a hard cap. Hitting it raises `MaxRoundsExceeded` rather than
looping forever.

### 4. Structured output

```python
from pydantic import BaseModel

class Order(BaseModel):
    order_id: str
    status: str
    days_late: int

order = client.structured(messages, schema=Order)   # a real Order instance
print(order.days_late)
```

Ollama has no true "strict" mode, so the SDK validates the reply and raises
`StructuredOutputError` if the model returns something unparseable. With a 14B,
treat that as a normal occasional outcome — catch it and retry.

### 5. Embeddings

Requires `ollama pull nomic-embed-text` (see A2).

```python
vectors = client.embed(["first document", "second document"])
len(vectors[0])     # 768 floats
```

These are for semantic search / RAG: embed your documents once, embed the
question, retrieve the nearest chunks, then paste those into a chat prompt.

### Async

Identical API with `await` in front. Use it if your app is itself FastAPI.

```python
from localsdk import AsyncClient

async with AsyncClient() as client:
    reply = await client.chat(messages)

    async for chunk in client.stream(messages):   # note: no await on stream()
        print(chunk.text, end="")
```

### Error handling

```python
from localsdk import (
    AuthenticationError, NotFoundError, UpstreamError,
    APIConnectionError, APITimeoutError, StructuredOutputError,
)

try:
    reply = client.chat(messages)
except AuthenticationError:
    ...   # bad or missing API key
except NotFoundError:
    ...   # that model is not pulled on the server
except UpstreamError:
    ...   # Ollama is down or timed out (502 / 504)
except APIConnectionError:
    ...   # tunnel is dead, URL is stale, or ngrok returned its HTML page
```

Every error carries `.status_code`, `.code` and `.message`.

---

## Day to day: the ngrok restart ritual

On the free plan **the tunnel URL changes every time ngrok restarts.** This is
the thing most likely to break your setup. When it happens:

1. On the GPU box, read the new URL from the ngrok output.
2. Update each client — one place each, thanks to config precedence:
   - env var: `export LOCALSDK_BASE_URL=https://<new>.ngrok-free.app`
   - or edit `~/.localsdk/config.toml`
3. Verify: `python -c "from localsdk import Client; print(Client().health())"`

**Symptom when you forget:** an `APIConnectionError` mentioning an HTML page or
a dead tunnel — the SDK tells you explicitly to check `LOCALSDK_BASE_URL`.

Permanent fixes, both out of scope for v1: claim ngrok's one free **static
domain**, or run a resolver endpoint the SDK queries on startup.

---

## Testing

```bash
cd server && python -m pytest      # 46 tests
cd sdk    && python -m pytest      # 113 tests
```

Neither suite needs a live Ollama or a network — both mock the transport.

---

## Operational notes

These are not incidental; they shaped the design:

- **Free ngrok meters requests.** An agent loop costs one request *per round*.
  That is why `run_tools` is opt-in with a hard cap, and why SDK retries default
  to `max_retries=0` — a silent retry doubles spend.
- **ngrok serves an HTML interstitial** unless `ngrok-skip-browser-warning: true`
  is sent. The SDK always sends it, and raises a specific, actionable error if
  HTML comes back anyway.
- **One 14B on one box is roughly one concurrent generation.** Streaming is a
  first-class path, and the server read timeout is 300s rather than httpx's 5s
  default.
- **Revoking a key today means editing `API_KEYS` and restarting.** The static
  key store has no revocation channel; that arrives with the Postgres backend.
- **`/healthz` is unauthenticated**, so it cannot validate a key — a wrong key
  still gets a green health check.

## Troubleshooting

| symptom | cause | fix |
| --- | --- | --- |
| `ConfigurationError` at construction | no api_key / base_url found anywhere | set `LOCALSDK_API_KEY` and `LOCALSDK_BASE_URL` |
| `APIConnectionError` mentioning HTML | stale tunnel URL, or ngrok interstitial | get the current ngrok URL, update `base_url` |
| `AuthenticationError` (401) | key not in the server's `API_KEYS` | check the server `.env`, restart it after editing |
| `NotFoundError` (404) on chat | model not pulled | `ollama pull qwen2.5:14b` |
| `NotFoundError` (404) on embed | embedding model not pulled | `ollama pull nomic-embed-text` |
| `UpstreamError` (502) | Ollama not running | start Ollama, check `OLLAMA_BASE_URL` |
| `UpstreamError` (504) | generation exceeded the timeout | raise `REQUEST_TIMEOUT_S`, or use `stream()` |
| `healthz` says `"ollama":"down"` | server is up, Ollama is not | start Ollama |

## Roadmap — not in v1

Postgres-backed key store (the `KeyStore` seam already exists), usage metering
and quotas, vision models (`qwen2.5:14b` is text-only), a vector store,
multi-tenancy, and an external developer portal.
