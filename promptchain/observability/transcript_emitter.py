"""Opt-in JSONL transcript emitter — writes one append-only JSONL transcript per PromptChain run for SIO mining."""

import json
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ..utils.execution_events import ExecutionEvent, ExecutionEventType

# ---------------------------------------------------------------------------
# Schema constants (locked — downstream F2 + SIO harness depend on these)
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 1

# Maps ExecutionEventType → the "type" field value in the JSONL transcript line.
# Event types not in this mapping are silently ignored by _build_line.
_EVENT_TYPE_TO_LINE_TYPE: dict[ExecutionEventType, str] = {
    ExecutionEventType.CHAIN_START: "chain_start",
    ExecutionEventType.STEP_START: "step",
    ExecutionEventType.STEP_END: "step",
    ExecutionEventType.STEP_SKIPPED: "step",
    ExecutionEventType.MODEL_CALL_START: "model_call",
    ExecutionEventType.MODEL_CALL_END: "model_call",
    ExecutionEventType.MODEL_CALL_ERROR: "model_call",
    ExecutionEventType.TOOL_CALL_START: "tool_call",
    ExecutionEventType.TOOL_CALL_END: "tool_result",
    ExecutionEventType.TOOL_CALL_ERROR: "tool_result",
    ExecutionEventType.FUNCTION_CALL_START: "tool_call",
    ExecutionEventType.FUNCTION_CALL_END: "tool_result",
    ExecutionEventType.FUNCTION_CALL_ERROR: "tool_result",
    ExecutionEventType.CHAIN_END: "chain_end",
    ExecutionEventType.CHAIN_ERROR: "chain_error",
}


@dataclass
class TranscriptEmitterConfig:
    """Configuration for the JSONL transcript emitter.

    Emission is OFF by default (opt-in). Set ``enabled=True`` or the
    ``PROMPTCHAIN_TRANSCRIPTS_ENABLED`` environment variable to activate.
    """

    enabled: bool = False
    base_dir: Path = field(
        default_factory=lambda: Path.home() / ".promptchain" / "transcripts"
    )
    project: Optional[str] = None
    max_files: int = 500
    max_bytes: int = 200 * 1024 * 1024
    max_value_len: int = 8192


class TranscriptEmitter:
    """Observer that appends execution events to a per-run JSONL transcript file."""

    def __init__(
        self, config: Optional[TranscriptEmitterConfig] = None, **kwargs: Any
    ) -> None:
        """Initialise with an optional config; extra kwargs are accepted for forward-compatibility."""
        self.config = config or TranscriptEmitterConfig()
        self._session_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Line builder (pure — no I/O; returns None for unmapped event types)
    # ------------------------------------------------------------------

    def _build_line(self, event: ExecutionEvent) -> Optional[dict]:
        """Build a JSONL transcript line dict for *event*, or ``None`` if the
        event type is not mapped in ``_EVENT_TYPE_TO_LINE_TYPE``.

        This method is pure (no I/O, no side-effects).  Field shapes match the
        locked schema in ``specs/012-sio-output-integration/contracts/transcript-schema.md``.
        Redaction and truncation are wired in a later task; values are emitted raw here.
        """
        event_type = event.event_type
        line_type = _EVENT_TYPE_TO_LINE_TYPE.get(event_type)
        if line_type is None:
            return None

        md: dict = event.metadata or {}

        # Common envelope — present on every line.
        line: dict = {
            "type": line_type,
            "ts": event.timestamp.isoformat(),
            "session_id": self._session_id or md.get("chain_id") or "unknown",
        }

        # Per-type payload.
        if line_type == "chain_start":
            line["project"] = self.config.project or md.get("project")
            line["schema_version"] = SCHEMA_VERSION

        elif line_type == "step":
            if event_type == ExecutionEventType.STEP_START:
                phase = "start"
            elif event_type == ExecutionEventType.STEP_END:
                phase = "end"
            else:  # STEP_SKIPPED
                phase = "skipped"
            line["phase"] = phase
            line["instruction_index"] = event.step_number
            line["execution_time_ms"] = md.get("execution_time_ms")

        elif line_type == "model_call":
            if event_type == ExecutionEventType.MODEL_CALL_START:
                phase = "start"
            elif event_type == ExecutionEventType.MODEL_CALL_END:
                phase = "end"
            else:  # MODEL_CALL_ERROR
                phase = "error"
            line["phase"] = phase
            line["model"] = event.model_name or md.get("model_name")
            line["call_id"] = md.get("call_id")
            line["usage"] = md.get("usage")
            line["execution_time_ms"] = md.get("execution_time_ms")
            line["error"] = md.get("error")

        elif line_type == "tool_call":
            # TOOL_CALL_START or FUNCTION_CALL_START
            line["call_id"] = md.get("call_id")
            line["tool_name"] = md.get("tool_name")
            line["arguments"] = md.get("arguments")

        elif line_type == "tool_result":
            # TOOL_CALL_END/ERROR or FUNCTION_CALL_END/ERROR
            is_error = event_type in (
                ExecutionEventType.TOOL_CALL_ERROR,
                ExecutionEventType.FUNCTION_CALL_ERROR,
            )
            line["call_id"] = md.get("call_id")
            line["tool_name"] = md.get("tool_name")
            line["status"] = "error" if is_error else "ok"
            line["result"] = md.get("result")
            line["error"] = md.get("error")

        elif line_type == "chain_end":
            line["stop_reason"] = "completed"
            line["total_tokens"] = md.get("total_tokens")
            line["execution_time_ms"] = md.get("execution_time_ms")
            line["outcome"] = "success"
            line["correction_count"] = 0
            line["positive_signal_count"] = 0
            line["sidechain_count"] = 0

        elif line_type == "chain_error":
            line["stop_reason"] = md.get("stop_reason") or "error"
            line["error"] = md.get("error")
            line["outcome"] = "error"

        return line

    async def handle_event(self, event: ExecutionEvent) -> None:
        """Receive an execution event (no-op skeleton — writing not yet implemented)."""
        return
