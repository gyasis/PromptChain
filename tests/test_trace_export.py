"""Tests for TRL-shaped trace export (promptchain.utils.trace_export)."""

import json
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from promptchain import PromptChain
from promptchain.utils.execution_callback import CallbackManager
from promptchain.utils.execution_events import ExecutionEvent, ExecutionEventType
from promptchain.utils.trace_export import (
    TraceRecorder,
    records_from_agentic,
    write_jsonl,
)

SECRET = "sk-" + "a1B2c3D4e5F6g7H8"


def _chain(responses, **kw):
    chain = PromptChain(models=["openai/test"] * 2, instructions=["Summarize this {input}", "Shorten this {input}"], **kw)
    return chain, AsyncMock(side_effect=responses)


@pytest.mark.asyncio
async def test_two_steps_two_records_with_rendered_prompts():
    chain, mock = _chain(
        [
            {"role": "assistant", "content": "long summary", "usage": {"x": 1}},
            {"role": "assistant", "content": "short"},
        ]
    )
    rec = TraceRecorder()
    chain.register_callback(rec.handle_event)
    with patch.object(PromptChain, "run_model_async", mock):
        out = await chain.process_prompt_async("hello world")
    assert out == "short"
    records = rec.records()
    assert len(records) == 2
    assert records[0]["prompt"] == [{"role": "user", "content": "Summarize this\n\nhello world"}]
    assert records[0]["completion"] == [{"role": "assistant", "content": "long summary"}]
    assert records[1]["prompt"][-1]["content"] == "Shorten this\n\nlong summary"
    assert [r["step"] for r in records] == [1, 2]
    assert records[0]["model"] == "openai/test"
    assert "tools" not in records[0]


@pytest.mark.asyncio
async def test_tool_call_path_arguments_become_dict():
    def add(a: int, b: int) -> int:
        return a + b

    schema = {
        "type": "function",
        "function": {
            "name": "add",
            "description": "add",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            },
        },
    }
    chain = PromptChain(models=["openai/test"], instructions=["Add these {input}"])
    chain.add_tools([schema])
    chain.register_tool_function(add)
    tool_msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "add", "arguments": '{"a": 2, "b": 3}'},
            }
        ],
    }
    mock = AsyncMock(side_effect=[tool_msg, {"role": "assistant", "content": "5"}])
    rec = TraceRecorder()
    chain.register_callback(rec.handle_event)
    with patch.object(PromptChain, "run_model_async", mock):
        await chain.process_prompt_async("2+3")
    records = rec.records()
    assert len(records) == 2
    first = records[0]["completion"][0]
    assert first["tool_calls"][0]["function"]["arguments"] == {"a": 2, "b": 3}
    assert records[0]["tools"][0]["function"]["name"] == "add"
    followup = records[1]
    assert any(m.get("role") == "tool" for m in followup["prompt"])
    assert followup["completion"] == [{"role": "assistant", "content": "5"}]
    # assistant tool_calls inside the follow-up prompt are normalised too
    prior = [m for m in followup["prompt"] if m.get("tool_calls")][0]
    assert prior["tool_calls"][0]["function"]["arguments"] == {"a": 2, "b": 3}


@pytest.mark.asyncio
async def test_secret_redacted_and_opt_out():
    for redact_secrets, expect in ((True, "***REDACTED***"), (False, SECRET)):
        chain = PromptChain(models=["openai/test"], instructions=["{input}"])
        rec = TraceRecorder(redact_secrets=redact_secrets)
        chain.register_callback(rec.handle_event)
        mock = AsyncMock(return_value={"role": "assistant", "content": f"key {SECRET}"})
        with patch.object(PromptChain, "run_model_async", mock):
            await chain.process_prompt_async(f"use {SECRET}")
        r = rec.records()[0]
        if redact_secrets:
            assert r["prompt"][0]["content"] == expect
            assert r["completion"][0]["content"] == expect
        else:
            assert SECRET in r["prompt"][0]["content"]


@pytest.mark.asyncio
async def test_errored_call_is_dropped():
    chain = PromptChain(models=["openai/test"], instructions=["{input}"])
    rec = TraceRecorder()
    chain.register_callback(rec.handle_event)
    mock = AsyncMock(side_effect=RuntimeError("boom"))
    with patch.object(PromptChain, "run_model_async", mock):
        with pytest.raises(Exception):
            await chain.process_prompt_async("x")
    assert rec.records() == []
    assert rec._pending == {}


@pytest.mark.asyncio
async def test_recorder_copies_payload_at_event_time():
    rec = TraceRecorder()
    msgs = [{"role": "user", "content": "a"}]
    resp = {"role": "assistant", "content": "b"}
    await rec.handle_event(
        ExecutionEvent(
            event_type=ExecutionEventType.MODEL_CALL_START,
            step_number=1,
            model_name="m",
            metadata={"prompt_messages": msgs, "tools": None},
        )
    )
    await rec.handle_event(
        ExecutionEvent(
            event_type=ExecutionEventType.MODEL_CALL_END,
            step_number=1,
            model_name="m",
            metadata={"response_message": resp},
        )
    )
    msgs[0]["content"] = "mutated"
    resp["content"] = "mutated"
    r = rec.records()[0]
    assert r["prompt"][0]["content"] == "a"
    assert r["completion"][0]["content"] == "b"
    r["prompt"][0]["content"] = "caller edit"
    assert rec.records()[0]["prompt"][0]["content"] == "a"


@pytest.mark.asyncio
async def test_reward_callable_adds_column():
    chain, mock = _chain([{"role": "assistant", "content": "x"}, {"role": "assistant", "content": "yy"}])
    rec = TraceRecorder()
    chain.register_callback(rec.handle_event)
    with patch.object(PromptChain, "run_model_async", mock):
        await chain.process_prompt_async("q")
    recs = rec.records(reward=lambda r: float(len(r["completion"][0]["content"])) if r["step"] == 2 else None)
    assert "reward" not in recs[0]
    assert recs[1]["reward"] == 2.0
    assert "reward" not in rec.records()[1]


def test_records_from_agentic_history_and_processor():
    history = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "goal"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "1", "type": "function", "function": {"name": "t", "arguments": '{"q": "x"}'}}],
        },
        {"role": "tool", "tool_call_id": "1", "name": "t", "content": f"result {SECRET}"},
        {"role": "assistant", "content": "done"},
    ]
    proc = types.SimpleNamespace(internal_history=history)
    for src in (history, proc):
        (rec,) = records_from_agentic(src, tools=[{"type": "function"}], reward=1)
        assert rec["reward"] == 1.0
        assert rec["tools"] == [{"type": "function"}]
        assert rec["messages"][2]["tool_calls"][0]["function"]["arguments"] == {"q": "x"}
        assert rec["messages"][3]["content"] == "***REDACTED***"
    assert history[2]["tool_calls"][0]["function"]["arguments"] == '{"q": "x"}'
    (raw,) = records_from_agentic(history, redact_secrets=False)
    assert SECRET in raw["messages"][3]["content"]
    assert "tools" not in raw and "reward" not in raw


def test_write_jsonl_round_trip(tmp_path):
    recs = [{"prompt": [{"role": "user", "content": "é"}], "completion": [{"role": "assistant", "content": "ü"}]}]
    p = tmp_path / "sub" / "t.jsonl"
    assert write_jsonl(recs, p) == 1
    assert write_jsonl(recs, p, append=True) == 1
    lines = p.read_text(encoding="utf-8").splitlines()
    assert [json.loads(l) for l in lines] == recs + recs
    assert write_jsonl(recs, p) == 1
    assert len(p.read_text().splitlines()) == 1


@pytest.mark.asyncio
async def test_no_callbacks_means_no_new_metadata_keys():
    chain = PromptChain(models=["openai/test"], instructions=["{input}"])
    assert chain.callback_manager.callbacks == []
    seen = []
    mock = AsyncMock(return_value={"role": "assistant", "content": "ok"})
    # a callback registered after the fact proves the keys are absent beforehand
    with patch.object(PromptChain, "run_model_async", mock):
        orig = CallbackManager.emit

        async def spy(self, event):
            seen.append(event)
            return await orig(self, event)

        with patch.object(CallbackManager, "emit", spy):
            await chain.process_prompt_async("x")
    model_events = [e for e in seen if e.event_type.name.startswith("MODEL_CALL")]
    assert model_events
    for e in model_events:
        assert not {"prompt_messages", "tools", "response_message"} & set(e.metadata)


@pytest.mark.asyncio
async def test_transcript_model_call_lines_keep_their_keys(tmp_path):
    from promptchain.observability.transcript_emitter import TranscriptEmitter

    emitter = TranscriptEmitter.__new__(TranscriptEmitter)
    start = ExecutionEvent(
        event_type=ExecutionEventType.MODEL_CALL_START,
        model_name="m",
        metadata={"call_id": "c", "prompt_messages": [{"role": "user", "content": "p"}], "tools": []},
    )
    end = ExecutionEvent(
        event_type=ExecutionEventType.MODEL_CALL_END,
        model_name="m",
        metadata={"call_id": "c", "response_message": {"content": "r"}},
    )
    emitter._session_id, emitter._project = "s", "p"
    emitter.config = MagicMock(project="p")
    for ev in (start, end):
        line = emitter._build_line(ev)
        assert set(line) == {"type", "ts", "session_id", "phase", "model", "call_id", "usage", "execution_time_ms", "error"}


def test_mlflow_observer_logs_no_new_artifacts():
    from promptchain.observability.mlflow_observer import MLflowObserver

    with patch("promptchain.observability.mlflow_observer.mlflow") as mlf:
        obs = MLflowObserver.__new__(MLflowObserver)
        obs._active_run = object()
        obs._run_start_times = {}
        obs.auto_log_artifacts = True
        start = ExecutionEvent(
            event_type=ExecutionEventType.MODEL_CALL_START,
            model_name="m",
            metadata={
                "model_name": "m",
                "prompt_messages": [{"role": "user", "content": SECRET}],
                "tools": [{"type": "function"}],
            },
        )
        end = ExecutionEvent(
            event_type=ExecutionEventType.MODEL_CALL_END,
            model_name="m",
            metadata={"response_message": {"role": "assistant", "content": SECRET}},
        )
        obs._handle_model_call_start(start)
        obs._handle_model_call_end(end)
        mlf.log_text.assert_not_called()
        mlf.log_artifact.assert_not_called()
