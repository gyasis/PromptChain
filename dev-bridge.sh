#!/usr/bin/env bash
# dev-bridge.sh — start the PromptChain↔Lookout SSE bridge (headless agentic chat for Lookout).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"
set -a
[ -f ~/dev/.env ] && . ~/dev/.env
[ -f ~/.config/environment.d/ollama-cloud.conf ] && . ~/.config/environment.d/ollama-cloud.conf
[ -f ~/.config/environment.d/sio-ollama.conf ] && . ~/.config/environment.d/sio-ollama.conf   # OLLAMA_HOST = Mac Studio
set +a
export PYTHONUNBUFFERED=1
# route ollama models (Mac Studio / local) through PromptChain's host
export OLLAMA_API_BASE="${OLLAMA_HOST:-http://192.168.0.159:11434}"
: "${BRIDGE_MODEL:=openai/gpt-4o-mini}"; export BRIDGE_MODEL
echo "[dev-bridge] $HERE  model=$BRIDGE_MODEL  port=7788"
exec /home/gyasis/miniconda3/bin/python -m uvicorn promptchain.server.web_bridge:app --host 127.0.0.1 --port 7788
