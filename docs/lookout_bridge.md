# Lookout ↔ PromptChain bridge

A small SSE server (`promptchain/server/web_bridge.py`, started by `dev-bridge.sh`) that gives the
Lookout overlay a model list and a headless PromptChain chat backend. It listens on
`127.0.0.1:7788` and is **fully env-configurable** — nothing host-specific is hardcoded, so you can
point it at any ollama endpoint (this machine, a box on your LAN, or a remote/hosted API).

## Start it

```bash
./dev-bridge.sh
```

It prints e.g. `[dev-bridge] <dir>  model=openai/gpt-4o-mini  ollama=http://localhost:11434  port=7788`.

## Configuration (environment variables)

| Var | Default | Purpose |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Which ollama the bridge lists + drives. **This is the "where do the local models live" switch.** |
| `BRIDGE_MODEL` | `openai/gpt-4o-mini` | Default model when the caller doesn't specify one. |
| `BRIDGE_PORT` | `7788` | Port the bridge listens on (loopback only). |
| `BRIDGE_PYTHON` | `$HOME/miniconda3/bin/python` (falls back to `python3`) | Interpreter that has `promptchain` installed. |
| `OPENAI_API_KEY` / other provider keys | — | Enable the matching cloud models in the list (via litellm). |

`dev-bridge.sh` also sources, if present: `~/dev/.env`,
`~/.config/environment.d/ollama-cloud.conf`, `~/.config/environment.d/sio-ollama.conf` — a
convenient place to set `OLLAMA_HOST` / API keys without exporting them by hand.

## Pointing at a specific ollama

```bash
# this machine (default)
OLLAMA_HOST=http://localhost:11434 ./dev-bridge.sh

# a box on your LAN (e.g. a Mac Studio / GPU server) — LAN ollama needs no auth
OLLAMA_HOST=http://<lan-ip>:11434 ./dev-bridge.sh

# a remote / hosted endpoint reachable over the network (outside API)
OLLAMA_HOST=https://ollama.example.com ./dev-bridge.sh
```

Whatever `OLLAMA_HOST` resolves to, the bridge lists that host's `/api/tags` under
`GET /models` and routes `ollama/<name>` turns to it. Switching hosts = restart the bridge with a
different `OLLAMA_HOST` (model switching within a host is per-turn, no restart needed).

> Calling from outside the box: the server binds to `127.0.0.1` on purpose. To expose it, front it
> with a reverse proxy (nginx/caddy) or an SSH tunnel — do not bind it to `0.0.0.0` without auth.

## HTTP API

- `GET /models` → `{ "default": "<id>", "models": [ { "id", "label", "provider", "note" } ] }`
- `POST /chat/turn` → `text/event-stream` of `{type, content}` events (`start`, `loop`, `thinking`,
  `tokens`, `answer`). Body:
  ```json
  { "message": "what do you see?", "model": "ollama/qwen3-vl:32b",
    "context": "optional text context", "images": ["data:image/png;base64,..."],
    "tools": [] }
  ```
  - `images` — data-URI screen frames for a **vision** model (text-only models skip them).
  - `tools` — **omit / `[]` for answer-only mode** (the default; no file/task/terminal side effects).
    Pass an explicit list of registry tool names to enable agentic execution for an action request.
