#!/usr/bin/env bash
# Dev launcher — ALWAYS runs THIS worktree's promptchain, never the global
# editable install (which is pinned to the main repo dir). It self-locates to
# the directory it lives in, so each worktree gets a copy that runs its own code.
#
# Usage:  ./dev-tui.sh [--dev] [--model ...] [any promptchain args]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
# Load Ollama Cloud key etc. so the model catalog can reach cloud models.
[ -f "$HOME/.config/environment.d/ollama-cloud.conf" ] && { set -a; . "$HOME/.config/environment.d/ollama-cloud.conf"; set +a; }
echo "[dev-tui] worktree: $HERE  (branch: $(git -C "$HERE" branch --show-current 2>/dev/null))"
exec /home/gyasis/miniconda3/bin/python -m promptchain.cli.main "$@"
