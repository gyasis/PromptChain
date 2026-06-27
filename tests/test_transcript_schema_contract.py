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


# ---------------------------------------------------------------------------
# T011 — Derivability contract (FR-002, FR-004)
# ---------------------------------------------------------------------------

def test_sio_derivable_from_transcript(tmp_path):
    """A generic consumer can derive all four SIO guarantees from a transcript.

    Covers FR-002 (model_used, token totals, tool sequence) and FR-004
    (error / stop_reason).  Two runs share tmp_path but use separate base_dirs.
    """
    # ------------------------------------------------------------------ run 1: success
    config1 = TranscriptEmitterConfig(
        enabled=True, base_dir=tmp_path / "run1", project="testproj"
    )
    emitter1 = TranscriptEmitter(config=config1)
    _drive(emitter1, _success_events())
    _path1, lines1 = _read_transcript(tmp_path / "run1")

    # 1. model_used — ≥1 model_call line, model is a non-empty string
    model_call_lines = [l for l in lines1 if l["type"] == "model_call"]
    assert len(model_call_lines) >= 1, "expected ≥1 model_call line"
    end_calls = [l for l in model_call_lines if l.get("phase") == "end"]
    assert end_calls, "expected a model_call with phase='end'"
    model_used = end_calls[0]["model"]
    assert isinstance(model_used, str) and model_used, (
        f"model_used must be a non-empty string, got {model_used!r}"
    )

    # 2. token totals — at least one path yields the integer total (3)
    total_from_usage = sum(
        l["usage"]["total_tokens"]
        for l in model_call_lines
        if isinstance(l.get("usage"), dict) and "total_tokens" in l["usage"]
    )
    chain_end_lines = [l for l in lines1 if l["type"] == "chain_end"]
    total_from_chain_end = (
        chain_end_lines[0].get("total_tokens") if chain_end_lines else None
    )
    assert total_from_usage == 3 or total_from_chain_end == 3, (
        f"expected total_tokens=3 via at least one path; "
        f"usage_sum={total_from_usage}, chain_end={total_from_chain_end}"
    )

    # 3. ordered tool sequence — pair tool_call→tool_result by call_id
    tool_calls = [l for l in lines1 if l["type"] == "tool_call"]
    tool_results = [l for l in lines1 if l["type"] == "tool_result"]
    results_by_call_id = {l["call_id"]: l for l in tool_results}
    paired = []
    for tc in tool_calls:
        cid = tc["call_id"]
        assert cid in results_by_call_id, (
            f"tool_call call_id={cid!r} has no matching tool_result (orphan)"
        )
        paired.append((tc["tool_name"], results_by_call_id[cid]["status"]))
    assert paired == [("build", "ok")], (
        f"expected tool sequence [('build', 'ok')], got {paired!r}"
    )

    # 4. stop_reason / outcome on the terminal line
    terminal1 = lines1[-1]
    assert "stop_reason" in terminal1, "terminal line missing stop_reason"
    assert "outcome" in terminal1, "terminal line missing outcome"
    assert terminal1["stop_reason"] == "completed", (
        f"expected stop_reason='completed', got {terminal1['stop_reason']!r}"
    )
    assert terminal1["outcome"] == "success", (
        f"expected outcome='success', got {terminal1['outcome']!r}"
    )

    # ------------------------------------------------------------------ run 2: error path
    config2 = TranscriptEmitterConfig(
        enabled=True, base_dir=tmp_path / "run2", project="testproj"
    )
    emitter2 = TranscriptEmitter(config=config2)
    error_events = [
        _ev(ExecutionEventType.CHAIN_START, chain_id="sess-d2", project="testproj"),
        _ev(
            ExecutionEventType.TOOL_CALL_ERROR,
            call_id="tool-e1",
            tool_name="build",
            error="build failed",
        ),
        _ev(ExecutionEventType.CHAIN_ERROR, stop_reason="error", error="chain failed"),
    ]
    _drive(emitter2, error_events)
    _path2, lines2 = _read_transcript(tmp_path / "run2")

    terminal2 = lines2[-1]
    assert terminal2["stop_reason"] in {"error", "limit"}, (
        f"expected stop_reason in {{'error','limit'}}, got {terminal2['stop_reason']!r}"
    )
    assert terminal2["outcome"] == "error", (
        f"expected outcome='error', got {terminal2['outcome']!r}"
    )

    # status=="error" derivable from a tool_result line (from TOOL_CALL_ERROR)
    error_results = [
        l for l in lines2 if l["type"] == "tool_result" and l.get("status") == "error"
    ]
    assert len(error_results) >= 1, (
        "expected ≥1 tool_result with status='error' when TOOL_CALL_ERROR is present"
    )
