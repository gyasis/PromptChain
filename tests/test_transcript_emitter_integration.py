"""T007 — Integration: real PromptChain.register_callback + CallbackManager.emit path.

Verifies that attaching the emitter via chain.register_callback and driving
events through chain.callback_manager.emit produces a valid transcript.
No LLM network calls are made — only construction + manual event emission.

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
from promptchain.utils.promptchaining import PromptChain


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

def test_register_callback_produces_transcript(tmp_path):
    """Attaching emitter via register_callback + emitting through callback_manager produces a valid transcript.

    Checks: ordered line types start with chain_start and end with chain_end,
    and the model_call line has a non-empty model string.
    """
    config = TranscriptEmitterConfig(
        enabled=True, base_dir=tmp_path, project="testproj"
    )
    emitter = TranscriptEmitter(config=config)

    # Construct a real PromptChain — no LLM calls, only construction
    chain = PromptChain(
        models=["openai/gpt-4o-mini"],
        instructions=["Echo: {input}"],
        store_steps=True,
    )
    chain.register_callback(emitter.handle_event)

    # Drive events through the REAL CallbackManager.emit path
    async def _run():
        for e in _success_events():
            await chain.callback_manager.emit(e)

    asyncio.run(_run())

    _path, lines = _read_transcript(tmp_path)

    assert lines[0]["type"] == "chain_start", (
        f"first line should be 'chain_start', got {lines[0]['type']!r}"
    )
    assert lines[-1]["type"] in {"chain_end", "chain_error"}, (
        f"last line should be terminal, got {lines[-1]['type']!r}"
    )
    assert lines[-1]["type"] == "chain_end", (
        f"success path must end with 'chain_end', got {lines[-1]['type']!r}"
    )

    model_lines = [l for l in lines if l.get("type") == "model_call"]
    assert model_lines, "no model_call line found in transcript"
    assert model_lines[0].get("model"), (
        "model_call line must have a non-empty 'model' field"
    )


def test_erroring_run_still_closes_with_terminal(tmp_path):
    """An error sequence must still produce a transcript with chain_start first and chain_error last.

    No partial-only file: even when events indicate failure, the terminal line
    must be present.
    """
    config = TranscriptEmitterConfig(
        enabled=True, base_dir=tmp_path, project="testproj"
    )
    emitter = TranscriptEmitter(config=config)

    chain = PromptChain(
        models=["openai/gpt-4o-mini"],
        instructions=["Echo: {input}"],
        store_steps=True,
    )
    chain.register_callback(emitter.handle_event)

    error_events = [
        _ev(ExecutionEventType.CHAIN_START, chain_id="sess-2"),
        _ev(ExecutionEventType.MODEL_CALL_ERROR, error="x"),
        _ev(ExecutionEventType.CHAIN_ERROR, error="x"),
    ]

    async def _run():
        for e in error_events:
            await chain.callback_manager.emit(e)

    asyncio.run(_run())

    # Transcript must exist even for erroring runs
    _path, lines = _read_transcript(tmp_path)

    assert lines[0]["type"] == "chain_start", (
        f"first line should be 'chain_start', got {lines[0]['type']!r}"
    )
    assert lines[-1]["type"] == "chain_error", (
        f"last line should be 'chain_error', got {lines[-1]['type']!r}"
    )
