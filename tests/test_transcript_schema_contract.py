"""T005 — Contract-level invariants for the JSONL transcript schema.

Every line must satisfy the common envelope (type, ts, session_id), the first
line must be chain_start, the last must be a terminal, and the session_id must
match the chain_id supplied in the CHAIN_START event metadata.

These tests INTENTIONALLY FAIL until TranscriptEmitter.handle_event is
implemented (the current stub is a no-op that writes nothing).
"""

import asyncio
import json

import pytest

from promptchain.observability.transcript_emitter import (
    TranscriptEmitter,
    TranscriptEmitterConfig,
)
from promptchain.utils.execution_events import ExecutionEvent, ExecutionEventType


# ---------------------------------------------------------------------------
# Shared helpers (duplicated per-file so tests are fully independent)
# ---------------------------------------------------------------------------

def _ev(event_type, *, model_name=None, step_number=None, **metadata):
    return ExecutionEvent(
        event_type=event_type,
        model_name=model_name,
        step_number=step_number,
        metadata=metadata,
    )


def _drive(emitter, events):
    async def _run():
        for e in events:
            await emitter.handle_event(e)

    asyncio.run(_run())


def _read_transcript(base_dir, project="testproj"):
    d = base_dir / project
    files = list(d.glob("*.jsonl")) if d.exists() else []
    assert files, f"expected a transcript file under {d}, found none"
    lines = [
        json.loads(l)
        for l in files[0].read_text().splitlines()
        if l.strip()
    ]
    return files[0], lines


def _success_events():
    """Realistic ordered US1 event sequence (success path)."""
    return [
        _ev(ExecutionEventType.CHAIN_START, chain_id="sess-1", project="testproj"),
        _ev(
            ExecutionEventType.MODEL_CALL_END,
            model_name="ollama/qwen3-coder:30b",
            call_id="model-1",
            usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            execution_time_ms=5,
        ),
        _ev(
            ExecutionEventType.TOOL_CALL_START,
            call_id="tool-1",
            tool_name="build",
            arguments={"path": "x"},
        ),
        _ev(
            ExecutionEventType.TOOL_CALL_END,
            call_id="tool-1",
            tool_name="build",
            result="ok",
        ),
        _ev(ExecutionEventType.CHAIN_END, total_tokens=3, execution_time_ms=10),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_every_line_is_valid_json_with_required_envelope(tmp_path):
    """Every emitted line must carry type, ts, and session_id — all non-None."""
    config = TranscriptEmitterConfig(
        enabled=True, base_dir=tmp_path, project="testproj"
    )
    emitter = TranscriptEmitter(config=config)

    _drive(emitter, _success_events())

    _path, lines = _read_transcript(tmp_path)

    assert len(lines) >= 1, "transcript must have at least one line"

    for i, line in enumerate(lines):
        assert "type" in line, f"line {i} missing 'type'"
        assert "ts" in line, f"line {i} missing 'ts'"
        assert "session_id" in line, f"line {i} missing 'session_id'"
        assert line["type"] is not None, f"line {i} 'type' is None"
        assert line["ts"] is not None, f"line {i} 'ts' is None"
        assert line["session_id"] is not None, f"line {i} 'session_id' is None"
        # Every line must round-trip through JSON without raising
        json.dumps(line)


def test_first_line_is_chain_start_last_is_terminal(tmp_path):
    """First line must be chain_start; last line must be chain_end or chain_error."""
    config = TranscriptEmitterConfig(
        enabled=True, base_dir=tmp_path, project="testproj"
    )
    emitter = TranscriptEmitter(config=config)

    _drive(emitter, _success_events())

    _path, lines = _read_transcript(tmp_path)

    assert lines[0]["type"] == "chain_start", (
        f"first line type should be 'chain_start', got {lines[0]['type']!r}"
    )
    assert lines[-1]["type"] in {"chain_end", "chain_error"}, (
        f"last line type should be terminal, got {lines[-1]['type']!r}"
    )


def test_session_id_matches_chain_id(tmp_path):
    """Every line's session_id must equal the chain_id from CHAIN_START; file stem must match."""
    config = TranscriptEmitterConfig(
        enabled=True, base_dir=tmp_path, project="testproj"
    )
    emitter = TranscriptEmitter(config=config)

    _drive(emitter, _success_events())

    path, lines = _read_transcript(tmp_path)

    for i, line in enumerate(lines):
        assert line["session_id"] == "sess-1", (
            f"line {i} session_id={line['session_id']!r}, expected 'sess-1'"
        )

    assert path.stem == "sess-1", (
        f"file stem should be 'sess-1', got {path.stem!r}"
    )
