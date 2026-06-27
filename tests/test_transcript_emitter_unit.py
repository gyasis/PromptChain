"""T006 — Event-to-line field mapping for the TranscriptEmitter.

Assertions are made on the emitted transcript file (not private methods).
Tests cover: model_call fields, tool_call/tool_result pairing, chain_end
terminal fields, and chain_error terminal fields.

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

def test_model_call_line_fields(tmp_path):
    """model_call line must carry model, usage.total_tokens, and call_id."""
    config = TranscriptEmitterConfig(
        enabled=True, base_dir=tmp_path, project="testproj"
    )
    emitter = TranscriptEmitter(config=config)

    _drive(emitter, _success_events())

    _path, lines = _read_transcript(tmp_path)

    model_lines = [l for l in lines if l.get("type") == "model_call"]
    assert model_lines, "no model_call line found in transcript"

    mc = model_lines[0]
    assert mc.get("model"), (
        f"model_call line must have non-empty 'model', got {mc.get('model')!r}"
    )
    assert mc["model"] == "ollama/qwen3-coder:30b", (
        f"expected model='ollama/qwen3-coder:30b', got {mc['model']!r}"
    )

    usage = mc.get("usage")
    assert isinstance(usage, dict), f"'usage' must be a dict, got {type(usage)}"
    assert usage.get("total_tokens") == 3, (
        f"usage.total_tokens should be 3, got {usage.get('total_tokens')}"
    )

    assert mc.get("call_id") == "model-1", (
        f"call_id should be 'model-1', got {mc.get('call_id')!r}"
    )


def test_tool_call_and_result_pairing(tmp_path):
    """tool_call and tool_result lines must be correctly paired via call_id."""
    config = TranscriptEmitterConfig(
        enabled=True, base_dir=tmp_path, project="testproj"
    )
    emitter = TranscriptEmitter(config=config)

    _drive(emitter, _success_events())

    _path, lines = _read_transcript(tmp_path)

    # tool_call line (from TOOL_CALL_START)
    tc_lines = [l for l in lines if l.get("type") == "tool_call"]
    assert tc_lines, "no tool_call line found in transcript"
    tc = tc_lines[0]
    assert tc.get("tool_name") == "build", (
        f"tool_call.tool_name should be 'build', got {tc.get('tool_name')!r}"
    )
    assert tc.get("arguments") == {"path": "x"}, (
        f"tool_call.arguments should be {{'path': 'x'}}, got {tc.get('arguments')!r}"
    )
    assert tc.get("call_id") == "tool-1", (
        f"tool_call.call_id should be 'tool-1', got {tc.get('call_id')!r}"
    )

    # tool_result line (from TOOL_CALL_END)
    tr_lines = [l for l in lines if l.get("type") == "tool_result"]
    assert tr_lines, "no tool_result line found in transcript"
    tr = tr_lines[0]
    assert tr.get("status") == "ok", (
        f"tool_result.status should be 'ok', got {tr.get('status')!r}"
    )
    assert tr.get("result") == "ok", (
        f"tool_result.result should be 'ok', got {tr.get('result')!r}"
    )
    assert tr.get("call_id") == "tool-1", (
        f"tool_result.call_id should be 'tool-1', got {tr.get('call_id')!r}"
    )


def test_chain_end_terminal_fields(tmp_path):
    """chain_end line must carry stop_reason='completed', outcome='success', total_tokens."""
    config = TranscriptEmitterConfig(
        enabled=True, base_dir=tmp_path, project="testproj"
    )
    emitter = TranscriptEmitter(config=config)

    _drive(emitter, _success_events())

    _path, lines = _read_transcript(tmp_path)

    ce_lines = [l for l in lines if l.get("type") == "chain_end"]
    assert ce_lines, "no chain_end line found in transcript"

    ce = ce_lines[0]
    assert ce.get("stop_reason") == "completed", (
        f"chain_end.stop_reason should be 'completed', got {ce.get('stop_reason')!r}"
    )
    assert ce.get("outcome") == "success", (
        f"chain_end.outcome should be 'success', got {ce.get('outcome')!r}"
    )
    assert ce.get("total_tokens") == 3, (
        f"chain_end.total_tokens should be 3, got {ce.get('total_tokens')}"
    )


def test_chain_error_terminal_fields(tmp_path):
    """chain_error line: stop_reason in {error,limit}, outcome='error', error present.

    The transcript must also begin with a chain_start line (no partial-only file).
    """
    config = TranscriptEmitterConfig(
        enabled=True, base_dir=tmp_path, project="testproj"
    )
    emitter = TranscriptEmitter(config=config)

    error_events = [
        _ev(ExecutionEventType.CHAIN_START, chain_id="sess-e"),
        _ev(ExecutionEventType.MODEL_CALL_ERROR, error="boom"),
        _ev(ExecutionEventType.CHAIN_ERROR, error="boom"),
    ]
    _drive(emitter, error_events)

    _path, lines = _read_transcript(tmp_path)

    # Terminal must be chain_error
    last = lines[-1]
    assert last.get("type") == "chain_error", (
        f"last line should be 'chain_error', got {last.get('type')!r}"
    )
    assert last.get("stop_reason") in {"error", "limit"}, (
        f"chain_error.stop_reason should be 'error' or 'limit', got {last.get('stop_reason')!r}"
    )
    assert last.get("outcome") == "error", (
        f"chain_error.outcome should be 'error', got {last.get('outcome')!r}"
    )
    assert last.get("error") is not None, (
        "chain_error line must have a non-None 'error' field"
    )

    # First line must be chain_start (no partial-only file)
    assert lines[0].get("type") == "chain_start", (
        f"first line should be 'chain_start', got {lines[0].get('type')!r}"
    )
