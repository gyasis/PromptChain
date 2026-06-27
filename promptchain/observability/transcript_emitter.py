"""Opt-in JSONL transcript emitter — writes one append-only JSONL transcript per PromptChain run for SIO mining."""

import json
import logging
import os
import re
import threading
import uuid
from dataclasses import dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

from ..utils.execution_events import ExecutionEvent, ExecutionEventType
from ._transcript_redaction import redact, truncate

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
        """Initialise with an optional config.

        Convenience: when no ``config`` is given, recognised config fields passed
        as keyword args (e.g. ``TranscriptEmitter(enabled=True)`` per the quickstart)
        are folded into a fresh ``TranscriptEmitterConfig``. Unknown kwargs are
        ignored for forward-compatibility.
        """
        if config is None and kwargs:
            _known = {f.name for f in fields(TranscriptEmitterConfig)}
            config = TranscriptEmitterConfig(
                **{k: v for k, v in kwargs.items() if k in _known}
            )
        self.config = config or TranscriptEmitterConfig()
        # Resolve enabled once at init: config flag OR env-var opt-in.
        _env = os.environ.get("PROMPTCHAIN_TRANSCRIPTS_ENABLED", "").lower()
        self._enabled: bool = self.config.enabled or _env in {"1", "true", "yes"}
        self._session_id: Optional[str] = None
        self._project: Optional[str] = None
        self._path: Optional[Path] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _safe_value(self, value: Any) -> Any:
        """Apply secret redaction then length truncation to *value*.

        Redaction runs first (D6) so truncation markers are never mistaken
        for secrets.  Only free-text / payload fields are passed through
        here — structural fields (type, ts, call_id, …) are left as-is.
        """
        return truncate(redact(value), self.config.max_value_len)

    def _rotate_transcripts(self) -> None:
        """Delete the oldest transcript files until within max_files and max_bytes (D7).

        Called after writing a terminal event while ``self._path`` is still set.
        Deletes files by ascending (mtime, name) order — the most-recently
        written file always has the newest mtime and is the last to be removed.
        FileNotFoundError on any individual file is silently ignored.
        """
        if self._path is None:
            return
        transcript_dir = self._path.parent
        try:
            entries: list[tuple[float, str, Path, int]] = []
            for f in transcript_dir.glob("*.jsonl"):
                try:
                    st = f.stat()
                    entries.append((st.st_mtime, f.name, f, st.st_size))
                except FileNotFoundError:
                    pass
            entries.sort()  # ascending (mtime, name) — oldest first
        except Exception:
            logger.error("TranscriptEmitter: error listing transcripts for rotation", exc_info=True)
            return

        total_bytes = sum(e[3] for e in entries)
        while entries and (
            len(entries) > self.config.max_files
            or total_bytes > self.config.max_bytes
        ):
            _mtime, _name, oldest_path, oldest_size = entries.pop(0)
            try:
                oldest_path.unlink()
                total_bytes -= oldest_size
            except FileNotFoundError:
                total_bytes -= oldest_size  # already gone; keep count accurate
            except Exception:
                logger.error("TranscriptEmitter: error deleting transcript during rotation", exc_info=True)
                break  # avoid infinite loop on repeated I/O errors

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
            line["project"] = (
                self._project
                if self._project is not None
                else (self.config.project or md.get("project"))
            )
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
            line["error"] = self._safe_value(md.get("error"))

        elif line_type == "tool_call":
            # TOOL_CALL_START or FUNCTION_CALL_START
            line["call_id"] = md.get("call_id")
            line["tool_name"] = md.get("tool_name")
            line["arguments"] = self._safe_value(md.get("arguments"))

        elif line_type == "tool_result":
            # TOOL_CALL_END/ERROR or FUNCTION_CALL_END/ERROR
            is_error = event_type in (
                ExecutionEventType.TOOL_CALL_ERROR,
                ExecutionEventType.FUNCTION_CALL_ERROR,
            )
            line["call_id"] = md.get("call_id")
            line["tool_name"] = md.get("tool_name")
            line["status"] = "error" if is_error else "ok"
            line["result"] = self._safe_value(md.get("result"))
            line["error"] = self._safe_value(md.get("error"))

        elif line_type == "chain_end":
            line["stop_reason"] = "completed"
            line["total_tokens"] = md.get("total_tokens")
            line["execution_time_ms"] = md.get("execution_time_ms")
            line["outcome"] = "success"
            line["correction_count"] = 0
            line["positive_signal_count"] = 0
            line["sidechain_count"] = 0
            line["schema_version"] = SCHEMA_VERSION

        elif line_type == "chain_error":
            line["stop_reason"] = md.get("stop_reason") or "error"
            line["error"] = self._safe_value(md.get("error"))
            line["outcome"] = "error"
            line["schema_version"] = SCHEMA_VERSION

        return line

    async def handle_event(self, event: ExecutionEvent) -> None:
        """Write one JSONL line for *event* to the per-run transcript file.

        Observability must never break the chain — all exceptions are caught
        and logged without re-raising.
        """
        # Zero work when disabled — gate before any path/format work (D8).
        if not self._enabled:
            return

        try:
            md: dict = event.metadata or {}
            event_type = event.event_type

            # On CHAIN_START, resolve and store run state for this run.
            if event_type == ExecutionEventType.CHAIN_START:
                self._session_id = md.get("chain_id") or uuid.uuid4().hex
                self._project = self.config.project or os.path.basename(os.getcwd())
                self._path = (
                    Path(self.config.base_dir)
                    / self._project
                    / f"{self._session_id}.jsonl"
                )
                self._path.parent.mkdir(parents=True, exist_ok=True)

            # Defensive: events arriving before CHAIN_START bootstrap run state.
            if self._path is None:
                self._session_id = md.get("chain_id") or uuid.uuid4().hex
                self._project = self.config.project or os.path.basename(os.getcwd())
                self._path = (
                    Path(self.config.base_dir)
                    / self._project
                    / f"{self._session_id}.jsonl"
                )
                self._path.parent.mkdir(parents=True, exist_ok=True)

            line = self._build_line(event)
            if line is None:
                return

            with open(self._path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(line, default=str) + "\n")

            # After a terminal event: rotate old transcripts, then reset run state
            # so the emitter can be reused cleanly for a subsequent run.
            if event_type in (
                ExecutionEventType.CHAIN_END,
                ExecutionEventType.CHAIN_ERROR,
            ):
                self._rotate_transcripts()
                self._session_id = None
                self._project = None
                self._path = None

        except Exception:
            logger.error("TranscriptEmitter: error handling event", exc_info=True)
