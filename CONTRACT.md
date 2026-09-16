# Shared Contract — LocalSDK Local LLM SDK

Frozen interface between `server/` and `sdk/`. **Neither agent may edit this file.**
If something here is wrong or missing, STOP and report it — do not unilaterally change it.

Wire protocol is **OpenAI-compatible**. Where this doc is silent, follow the OpenAI
Chat Completions spec exactly.

---

## 0. Ownership

| path | owner |
|---|---|
| `server/**` | server agent only |
| `sdk/**` | sdk agent only |
| `CONTRACT.md`, `README.md`, `.gitignore`, root | orchestrator only |

Do not create, edit, or delete files outside your own directory.

---

## 1. Topology

```
consumer app -> sdk (python pkg) --HTTPS--> ngrok --> FastAPI server --HTTP--> Ollama :11434 -> qwen2.5:14b
```

Constraints that shape everything:
- Free ngrok: requests are metered. Never spend a request implicitly.
- ngrok URL rotates on restart -> `base_url` is config, never hardcoded.
- ngrok serves an HTML interstitial unless header `ngrok-skip-browser-warning: true` is sent.
- One 14B on one box -> effectively 1 concurrent generation. Long generations need streaming.

---

## 2. Transport

- Base path: `{base_url}/v1`
- Auth: `Authorization: Bearer <api_key>` on every `/v1/*` route.
- SDK MUST send `ngrok-skip-browser-warning: true` on every request.
- SDK MUST send `User-Agent: localsdk-python/<version>`.
- Content type: `application/json`. Streaming responses: `text/event-stream`.

---

## 3. Endpoints

### `GET /healthz`  (NO auth)
```json
{"status": "ok", "ollama": "up", "model": "qwen2.5:14b", "version": "0.1.0"}
```
`ollama` is `"up"` or `"down"`. Returns 200 even when ollama is down (it is a probe, not a gate).

### `GET /v1/models`
```json
{"object": "list", "data": [{"id": "qwen2.5:14b", "object": "model", "created": 1700000000, "owned_by": "ollama"}]}
```

### `POST /v1/chat/completions`
Request fields that MUST be supported (ignore unknown fields, do not 400 on them):
`model` (str, optional -> DEFAULT_MODEL), `messages` (required), `temperature`, `top_p`,
`max_tokens`, `stream` (bool), `stop` (str|list[str]), `seed`, `tools`, `tool_choice`,
`response_format`.

Message roles: `system`, `user`, `assistant`, `tool`.
- assistant message may carry `tool_calls`
- tool message MUST carry `tool_call_id` and `content`

Non-streaming response:
```json
{
  "id": "chatcmpl-<uuid>",
  "object": "chat.completion",
  "created": 1700000000,
  "model": "qwen2.5:14b",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "...", "tool_calls": null},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
}
```
`finish_reason` is one of `stop`, `length`, `tool_calls`.

Streaming (`"stream": true`) — SSE, one JSON object per `data:` line:
```
data: {"id":"chatcmpl-x","object":"chat.completion.chunk","created":1,"model":"qwen2.5:14b","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-x","object":"chat.completion.chunk","created":1,"model":"qwen2.5:14b","choices":[{"index":0,"delta":{"content":"Hel"},"finish_reason":null}]}

data: {"id":"chatcmpl-x","object":"chat.completion.chunk","created":1,"model":"qwen2.5:14b","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]

```
Terminator is the literal line `data: [DONE]`. Every SSE event ends with a blank line.

### `POST /v1/embeddings`
Request: `{"model": "<optional>", "input": "str" | ["str", ...]}`
Response:
```json
{"object":"list","data":[{"object":"embedding","index":0,"embedding":[0.1,0.2]}],
 "model":"nomic-embed-text","usage":{"prompt_tokens":0,"total_tokens":0}}
```

---

## 4. Tool calling (OpenAI format)

Request carries:
```json
"tools": [{"type":"function","function":{"name":"get_weather","description":"...","parameters":{"type":"object","properties":{},"required":[]}}}]
```
When the model calls a tool, response has `finish_reason: "tool_calls"` and:
```json
"message": {"role":"assistant","content":null,"tool_calls":[
  {"id":"call_abc","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Pune\"}"}}
]}
```
`arguments` is a **JSON-encoded string**, not an object. The app executes the tool and appends
`{"role":"tool","tool_call_id":"call_abc","content":"<result as string>"}` then calls again.

---

## 5. Structured output

`response_format` accepts:
- `{"type": "json_object"}`
- `{"type": "json_schema", "json_schema": {"name": "X", "schema": {...}, "strict": true}}`

Server translates to Ollama's `format` field (Ollama accepts a raw JSON schema).

---

## 6. Errors — OpenAI envelope

```json
{"error": {"message": "...", "type": "invalid_request_error", "param": null, "code": "model_not_found"}}
```

| HTTP | type | code | when |
|---|---|---|---|
| 400 | invalid_request_error | invalid_request | malformed body |
| 401 | authentication_error | invalid_api_key | missing/bad/unknown key |
| 403 | permission_error | key_revoked | known but revoked key |
| 404 | invalid_request_error | model_not_found | model not present in ollama |
| 429 | rate_limit_error | rate_limit_exceeded | reserved, not enforced in v1 |
| 502 | upstream_error | ollama_unavailable | cannot reach ollama |
| 504 | upstream_error | ollama_timeout | ollama exceeded REQUEST_TIMEOUT_S |

Errors that occur **mid-stream** are emitted as a final SSE event
`data: {"error": {...}}` followed by `data: [DONE]`. SDK must raise on it.

---

## 7. Config

### Server env (`server/.env.example`)
```
OLLAMA_BASE_URL=http://localhost:11434
DEFAULT_MODEL=qwen2.5:14b
EMBED_MODEL=nomic-embed-text
KEYSTORE_BACKEND=static          # static | postgres
API_KEYS=
DATABASE_URL=                    # only for postgres backend
REQUEST_TIMEOUT_S=300
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO
```

### SDK config, precedence: explicit arg > env > config file > default
```
LOCALSDK_API_KEY
LOCALSDK_BASE_URL
LOCALSDK_MODEL
LOCALSDK_TIMEOUT
```
Config file: `~/.localsdk/config.toml` with `[default] api_key= base_url= model=`.

---

## 8. Versions / deps

- Python **3.10+** (use `X | None`, no `Optional` imports needed)
- pydantic **v2**, httpx, pytest
- server adds: fastapi, uvicorn[standard], python-dotenv
- Keep deps minimal. No langchain, no openai package, no sqlalchemy in v1.

## 9. Out of scope for v1
vision/images, vector store, Postgres implementation (interface only), quotas/billing,
multi-tenancy, external developer portal, ngrok URL auto-discovery.
