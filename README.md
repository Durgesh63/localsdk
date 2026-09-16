# LocalSDK Local LLM Platform

A Python SDK + FastAPI gateway for talking to a self-hosted Ollama model
(`qwen2.5:14b`) the same way you would talk to a hosted provider: install a
package, hand it an API key and a base URL, call it.

```
your app ──pip install localsdk──┐
                                    │  Authorization: Bearer sk-...
                                    ▼
                          ┌──────────────────────┐
                          │  server/ (FastAPI)   │  OpenAI-compatible
                          │  auth + passthrough  │  /v1/chat/completions
                          └──────────┬───────────┘  /v1/embeddings
                                     │              /v1/models
                                     ▼
                          Ollama :11434 → qwen2.5:14b
```

## Repo layout

| path | what it is |
|---|---|
| [CONTRACT.md](CONTRACT.md) | **the frozen HTTP contract** between the two halves — read this first |
| [sdk/](sdk/) | the Python client package your apps install |
| [server/](server/) | the FastAPI gateway that fronts Ollama |

`sdk/` and `server/` are independently developed against `CONTRACT.md`. Neither
imports the other.

## Why OpenAI-compatible

The wire format is the OpenAI Chat Completions spec. That buys three things:

1. The server is a thin auth + passthrough layer, not a translation layer — Ollama already speaks this format.
2. Anything can talk to it as a fallback (`openai` package, LangChain, curl) when the SDK lacks something.
3. The two halves can be built in parallel, because the contract is externally defined.

## Quickstart

**Server** (on the box with the GPU):
```bash
cd server
cp .env.example .env          # set API_KEYS=sk-local-dev-001
pip install -e ".[dev]"
uvicorn app.main:app --host 0.0.0.0 --port 8000
ngrok http 8000               # note the https URL it prints
```

**Client** (anywhere):
```bash
pip install "git+https://<your-repo>.git#subdirectory=sdk"
```
```python
from localsdk import Client

client = Client(api_key="sk-local-dev-001", base_url="https://xxxx.ngrok-free.app")
print(client.chat([{"role": "user", "content": "hello"}]).text)
```

## Operational gotchas

These are not incidental — they shaped the design:

- **The ngrok URL rotates on every restart.** `base_url` is configuration, never a
  hardcoded default. Every consumer must be re-pointed after a tunnel restart.
  Fixing this properly needs a static domain or a resolver endpoint — deliberately deferred.
- **Free ngrok meters requests.** An agent loop costs one request *per round*, which is
  why the auto-loop is opt-in and capped rather than the default, and why SDK retries
  default to off.
- **ngrok serves an HTML interstitial** unless `ngrok-skip-browser-warning: true` is sent.
  The SDK always sends it and raises a specific, actionable error if HTML comes back anyway.
- **One 14B on one box ≈ one concurrent generation.** Long generations are why streaming
  is a first-class path and the server timeout is 300s, not httpx's 5s default.

## Roadmap (explicitly not in v1)

Postgres-backed key store (the `KeyStore` seam exists), usage metering and quotas,
vision models (`qwen2.5:14b` is text-only), a vector store, multi-tenancy and an
external developer portal.
