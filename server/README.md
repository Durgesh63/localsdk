# localsdk-server

An OpenAI-compatible FastAPI façade in front of a local [Ollama](https://ollama.com)
instance running `qwen2.5:14b`. It is a thin auth + passthrough layer: bearer-key
auth on every `/v1/*` route, OpenAI request/response shapes on the wire, Ollama's
native API underneath.

```
consumer app -> sdk -> ngrok -> this server -> Ollama :11434 -> qwen2.5:14b
```

Implements `CONTRACT.md` at the repo root: `GET /healthz`, `GET /v1/models`,
`POST /v1/chat/completions` (streaming and not), `POST /v1/embeddings`, tool
calling, structured output, and the OpenAI error envelope.

---

## Run it

```bash
cd server
python -m venv venv && . venv/Scripts/activate    # Windows; use bin/activate on POSIX
python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows PowerShell
# source .venv/bin/activate         # Linux / macOS
pip install -r requirements-dev.txt

cp .env.example .env        # then set API_KEYS - it ships empty on purpose
uvicorn app.main:app --reload
```

Prerequisite: Ollama running with the models pulled.

```bash
ollama serve
ollama pull qwen2.5:14b
ollama pull nomic-embed-text
```

`/healthz` never fails on account of Ollama — it reports `"ollama": "down"` and
still returns 200, because it is a probe, not a gate.

## Tests

```bash
cd server
python -m pytest
```

The suite uses `httpx.MockTransport`, so it needs neither a live Ollama nor a
network connection.

## Environment

| var | default | meaning |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | where Ollama listens |
| `DEFAULT_MODEL` | `qwen2.5:14b` | used when a request omits `model` |
| `EMBED_MODEL` | `nomic-embed-text` | used by `/v1/embeddings` |
| `KEYSTORE_BACKEND` | `static` | `static` \| `postgres` (postgres is a v1 stub) |
| `API_KEYS` | **none** | **Required.** Comma-separated keys for the `static` backend. The server refuses to start if empty. |
| `DATABASE_URL` | — | reserved for the postgres backend |
| `REQUEST_TIMEOUT_S` | `300` | upstream timeout; a 14B is slow, keep it generous |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | bind address for your own launcher |
| `LOG_LEVEL` | `INFO` | root log level |

Keys are read from `API_KEYS` at startup. To rotate a key, change the env var and
restart. `PostgresKeyStore` exists to prove the seam and raises
`NotImplementedError`; a real implementation is out of scope for v1.

## Expose it via ngrok

```bash
ngrok http 8000
# Forwarding  https://<random>.ngrok-free.app -> http://localhost:8000
```

Notes that matter for clients:

- The free ngrok URL **rotates on every restart**. Treat it as configuration
  (`LOCALSDK_BASE_URL`), never hardcode it.
- ngrok serves an HTML interstitial unless the request carries
  `ngrok-skip-browser-warning: true`. The SDK sends it on every request; `curl`
  against an ngrok URL should too.
- Requests are metered on the free tier — do not poll `/healthz` through ngrok.

```bash
export BASE=https://<random>.ngrok-free.app
curl -s "$BASE/healthz" -H "ngrok-skip-browser-warning: true"
```

## Curl examples

Non-streaming:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-your-own-key" \
  -H "Content-Type: application/json" \
  -d '{
        "model": "qwen2.5:14b",
        "messages": [{"role": "user", "content": "Say hello in five words."}],
        "temperature": 0.2
      }'
```

```json
{"id":"chatcmpl-...","object":"chat.completion","created":1700000000,"model":"qwen2.5:14b",
 "choices":[{"index":0,"message":{"role":"assistant","content":"Hello there, nice to meet you.","tool_calls":null},"finish_reason":"stop"}],
 "usage":{"prompt_tokens":11,"completion_tokens":8,"total_tokens":19}}
```

Streaming (`-N` disables curl's own buffering):

```bash
curl -sN http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-your-own-key" \
  -H "Content-Type: application/json" \
  -d '{
        "messages": [{"role": "user", "content": "Count to three."}],
        "stream": true
      }'
```

```
data: {"id":"chatcmpl-x","object":"chat.completion.chunk","created":1,"model":"qwen2.5:14b","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-x","object":"chat.completion.chunk","created":1,"model":"qwen2.5:14b","choices":[{"index":0,"delta":{"content":"One"},"finish_reason":null}]}

data: {"id":"chatcmpl-x","object":"chat.completion.chunk","created":1,"model":"qwen2.5:14b","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

Embeddings:

```bash
curl -s http://localhost:8000/v1/embeddings \
  -H "Authorization: Bearer sk-your-own-key" \
  -H "Content-Type: application/json" \
  -d '{"input": ["first", "second"]}'
```

Tool calling — the returned `function.arguments` is a **JSON-encoded string**:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-your-own-key" \
  -H "Content-Type: application/json" \
  -d '{
        "messages": [{"role": "user", "content": "Weather in Pune?"}],
        "tools": [{"type":"function","function":{"name":"get_weather",
                   "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]
      }'
```

## Errors

Every failure — including request-validation failures, which FastAPI would
otherwise render as `{"detail": [...]}` — comes back as the OpenAI envelope:

```json
{"error": {"message": "...", "type": "authentication_error", "param": null, "code": "invalid_api_key"}}
```

| HTTP | type | code |
|---|---|---|
| 400 | `invalid_request_error` | `invalid_request` |
| 401 | `authentication_error` | `invalid_api_key` |
| 403 | `permission_error` | `key_revoked` |
| 404 | `invalid_request_error` | `model_not_found` |
| 429 | `rate_limit_error` | `rate_limit_exceeded` (reserved, not enforced) |
| 502 | `upstream_error` | `ollama_unavailable` |
| 504 | `upstream_error` | `ollama_timeout` |

A failure that happens *after* streaming has begun cannot change the status code,
so it is emitted as a final SSE event followed by the terminator:

```
data: {"error":{"message":"...","type":"upstream_error","param":null,"code":"ollama_timeout"}}

data: [DONE]
```

## Layout

```
app/
  main.py            app factory, lifespan, shared httpx.AsyncClient, router wiring
  config.py          env-backed settings
  deps.py            FastAPI dependencies (bearer auth, ollama client, settings)
  errors.py          error envelope, exception classes, handlers
  auth/keystore.py   KeyStore Protocol, StaticKeyStore, PostgresKeyStore stub
  ollama/client.py   async httpx wrapper: chat, chat_stream, embeddings, tags, ping
  ollama/translate.py  OpenAI <-> Ollama mapping (pure functions)
  routes/            healthz, models, chat, embeddings
  schemas/openai.py  pydantic v2 wire models
tests/               pytest + httpx.MockTransport, no live Ollama needed
```
