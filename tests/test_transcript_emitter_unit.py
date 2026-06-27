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

from promptchain.observability._transcript_redaction import redact, truncate
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


# ---------------------------------------------------------------------------
# T013 — Secret redaction (redact is a NO-OP stub → MUST FAIL until Wave 8)
# ---------------------------------------------------------------------------

def test_redact_key_name_matches():
    """T013: sensitive key names must have values replaced with '***REDACTED***'.

    Keys: api_key, Authorization, token, password, secret (case-insensitive).
    Benign keys (e.g. 'ok') must be left unchanged.

    FAILS NOW: redact() is a no-op stub that returns value unchanged.
    """
    inp = {
        "api_key": "abc",
        "Authorization": "Bearer z",
        "token": "t",
        "password": "p",
        "secret": "s",
        "ok": "keep",
    }
    result = redact(inp)
    for key in ("api_key", "Authorization", "token", "password", "secret"):
        assert result[key] == "***REDACTED***", (
            f"expected key {key!r} to be '***REDACTED***', got {result[key]!r}"
        )
    assert result["ok"] == "keep", (
        f"benign key 'ok' should be unchanged, got {result['ok']!r}"
    )


def test_redact_pattern_matches():
    """T013: string VALUES that look like secrets are masked even under benign keys.

    Covers: sk-... API keys, Bearer tokens, long high-entropy hex runs (≥32 chars).

    FAILS NOW: redact() is a no-op stub.
    """
    # OpenAI-style key: sk-<long run>
    r1 = redact({"note": "my key sk-abcdefghijklmnopqrstuvwxyz0123"})
    v1 = r1["note"]
    assert "***REDACTED***" in v1 or v1 == "***REDACTED***", (
        f"sk-... value should be redacted under a benign key, got {v1!r}"
    )

    # Bearer JWT-ish token
    r2 = redact({"note": "Bearer eyJhbGciOiJIUzI1NiJ9.aaaa.bbbb"})
    v2 = r2["note"]
    assert "***REDACTED***" in v2 or v2 == "***REDACTED***", (
        f"Bearer token value should be redacted, got {v2!r}"
    )

    # Long high-entropy hex string (40 hex chars = 160 bits of entropy)
    r3 = redact({"note": "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"})
    v3 = r3["note"]
    assert "***REDACTED***" in v3 or v3 == "***REDACTED***", (
        f"long hex run (≥32 chars) should be redacted, got {v3!r}"
    )


def test_redact_recursive():
    """T013: redact must recurse into nested dicts and lists.

    FAILS NOW: redact() is a no-op stub.
    """
    result = redact({
        "outer": {"token": "x"},
        "list": [{"api_key": "y"}, "normal"],
    })
    assert result["outer"]["token"] == "***REDACTED***", (
        f"nested dict token should be redacted, got {result['outer']['token']!r}"
    )
    assert result["list"][0]["api_key"] == "***REDACTED***", (
        f"api_key inside a list element should be redacted, got {result['list'][0]['api_key']!r}"
    )
    assert result["list"][1] == "normal", (
        f"plain string in list should be unchanged, got {result['list'][1]!r}"
    )


def test_redact_preserves_nonsecrets():
    """T013 (guard): ints, bools, and short benign strings must NOT be redacted.

    PASSES with the no-op stub — this is the over-redaction guard; it verifies that
    correct implementations do not mask everything indiscriminately.
    """
    result = redact({"count": 5, "msg": "hello world", "flag": True})
    assert result["count"] == 5, (
        f"int value should be unchanged, got {result['count']!r}"
    )
    assert result["msg"] == "hello world", (
        f"short benign string should be unchanged, got {result['msg']!r}"
    )
    assert result["flag"] is True, (
        f"bool value should be unchanged, got {result['flag']!r}"
    )


# ---------------------------------------------------------------------------
# T014 — Value truncation (truncate is a NO-OP stub → MUST FAIL until Wave 8)
# ---------------------------------------------------------------------------

def test_truncate_long_string():
    """T014: string longer than max_len is capped; tail replaced with marker.

    Marker format: '…[truncated N chars]' where N = chars removed.
    Result must remain JSON-serializable.

    FAILS NOW: truncate() is a no-op stub that returns value unchanged.
    """
    s = "x" * 1000
    out = truncate(s, 100)
    assert len(out) < 1000, (
        f"truncated output must be shorter than 1000 chars, got len={len(out)}"
    )
    assert out[:100] == "x" * 100, (
        "kept prefix must be the first max_len chars of the original string"
    )
    assert "[truncated 900 chars]" in out, (
        f"truncation marker '[truncated 900 chars]' missing from output: {out!r}"
    )
    json.dumps(out)  # must be JSON-serializable after truncation


def test_truncate_short_string_untouched():
    """T014 (guard): string shorter than max_len must be returned unchanged.

    PASSES with the no-op stub — this is the identity-on-short-input guard.
    """
    assert truncate("short", 100) == "short", (
        "string shorter than max_len must be returned unchanged"
    )


def test_truncate_keeps_json_valid():
    """T014: truncated value in a dict round-trips through json.dumps/loads.

    Also asserts that truncation actually occurred (marker is present in the
    round-tripped value).

    FAILS NOW: truncate() is a no-op stub; '[truncated' marker will be absent.
    """
    long_val = "y" * 5000
    truncated = truncate(long_val, 50)
    roundtripped = json.loads(json.dumps({"r": truncated}))
    assert "[truncated" in roundtripped["r"], (
        f"truncation marker must be present in round-tripped value, "
        f"got prefix {roundtripped['r'][:80]!r}"
    )


# ---------------------------------------------------------------------------
# T015 — Whole-file rotation by mtime (NOT implemented → MUST FAIL)
# ---------------------------------------------------------------------------

def test_rotation_bounds_dir(tmp_path):
    """T015: transcript directory stays within max_files after many runs.

    Drives 120 runs through a single enabled emitter with max_files=5.
    After all runs the directory must contain ≤5 files (oldest deleted).

    FAILS NOW: rotation is not implemented — ~120 files will accumulate.
    """
    cfg = TranscriptEmitterConfig(
        enabled=True,
        base_dir=tmp_path,
        project="rot",
        max_files=5,
    )
    em = TranscriptEmitter(cfg)
    for i in range(120):
        run = [
            _ev(ExecutionEventType.CHAIN_START, chain_id=f"r{i}"),
            _ev(ExecutionEventType.CHAIN_END, total_tokens=0),
        ]
        _drive(em, run)
    files = list((tmp_path / "rot").glob("*.jsonl"))
    assert len(files) <= 5, (
        f"rotation should cap dir at max_files=5, found {len(files)} files"
    )
