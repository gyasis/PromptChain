#!/usr/bin/env bash
# observe.sh — run a PromptChain script with MLflow tracking enabled.
#
# Usage:
#   bash scripts/observe.sh runs/2026-05-05_my-script
#   bash scripts/observe.sh runs/2026-05-05_my-script/run.py
#
# Behaviour:
#   - Sources .env from repo root if present (loads OPENAI_API_KEY etc.)
#   - Sets MLFLOW_TRACKING_URI to ./mlruns/ (project-local) if not set
#   - Sets PROMPTCHAIN_MLFLOW_BACKGROUND=1 (non-blocking writes)
#   - Tees stdout+stderr to <run-dir>/output.log
#
# Exits with the script's exit code.

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <run-dir-or-script>" >&2
  echo "Example: $0 runs/2026-05-05_my-script" >&2
  exit 64
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="$1"

# Resolve target to a script path
if [ -d "$REPO_ROOT/scripts/$TARGET" ]; then
  SCRIPT="$REPO_ROOT/scripts/$TARGET/run.py"
  RUN_DIR="$REPO_ROOT/scripts/$TARGET"
elif [ -f "$REPO_ROOT/scripts/$TARGET" ]; then
  SCRIPT="$REPO_ROOT/scripts/$TARGET"
  RUN_DIR="$(dirname "$SCRIPT")"
elif [ -f "$TARGET" ]; then
  SCRIPT="$TARGET"
  RUN_DIR="$(dirname "$SCRIPT")"
else
  echo "Could not resolve target: $TARGET" >&2
  exit 65
fi

if [ ! -f "$SCRIPT" ]; then
  echo "No run.py found in $RUN_DIR" >&2
  exit 66
fi

# Source .env if present
if [ -f "$REPO_ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$REPO_ROOT/.env"
  set +a
fi

# Default MLflow config (project-local)
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-file://$REPO_ROOT/mlruns}"
export PROMPTCHAIN_MLFLOW_BACKGROUND="${PROMPTCHAIN_MLFLOW_BACKGROUND:-1}"

LOG="$RUN_DIR/output.log"
echo "[observe.sh] Running $SCRIPT" | tee "$LOG"
echo "[observe.sh] MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI" | tee -a "$LOG"
echo "[observe.sh] $(date -Iseconds)" | tee -a "$LOG"
echo "---" | tee -a "$LOG"

# Run with output tee'd to log
python "$SCRIPT" 2>&1 | tee -a "$LOG"
exit "${PIPESTATUS[0]}"
