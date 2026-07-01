#!/usr/bin/env bash
# dev-bridge.sh — start the PromptChain↔Lookout SSE bridge (headless agentic chat for Lookout).
#
# Config is entirely via environment (nothing host-specific is hardcoded). Point it at ANY ollama:
#   OLLAMA_HOST=http://localhost:11434         # this machine (default)
#   OLLAMA_HOST=http://<lan-ip>:11434          # a box on your LAN (e.g. a Mac/GPU server)
#   OLLAMA_HOST=https://ollama.example.com     # a remote/hosted endpoint
# Optional overrides: BRIDGE_MODEL (default openai/gpt-4o-mini), BRIDGE_PORT (default 7788),
#   BRIDGE_PYTHON (python interpreter with promptchain installed). See docs/lookout_bridge.md.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"
set -a
[ -f ~/dev/.env ] && . ~/dev/.env
[ -f ~/.config/environment.d/ollama-cloud.conf ] && . ~/.config/environment.d/ollama-cloud.conf
[ -f ~/.config/environment.d/sio-ollama.conf ] && . ~/.config/environment.d/sio-ollama.conf   # may set OLLAMA_HOST
set +a
export PYTHONUNBUFFERED=1
# route ollama models through PromptChain's host (defaults to this machine; override via OLLAMA_HOST)
export OLLAMA_API_BASE="${OLLAMA_HOST:-http://localhost:11434}"
: "${BRIDGE_MODEL:=openai/gpt-4o-mini}"; export BRIDGE_MODEL
: "${BRIDGE_PORT:=7788}"; export BRIDGE_PORT
BRIDGE_PYTHON="${BRIDGE_PYTHON:-$HOME/miniconda3/bin/python}"
command -v "$BRIDGE_PYTHON" >/dev/null 2>&1 || BRIDGE_PYTHON="$(command -v python3 || command -v python)"
echo "[dev-bridge] $HERE  model=$BRIDGE_MODEL  ollama=$OLLAMA_API_BASE  port=$BRIDGE_PORT"
exec "$BRIDGE_PYTHON" -m uvicorn promptchain.server.web_bridge:app --host 127.0.0.1 --port "$BRIDGE_PORT"
