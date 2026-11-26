"""latency_probe — TTFT / decode / prefill throughput probe through the Atelier governor.

Writes one raw measurement per call to a separate JSONL store; the persisted
CapabilityProfile is not touched. Thinking is turned off with the endpoint's own
parameter and the emitted reasoning is measured, so a flag is never taken as evidence.
"""
from __future__ import annotations

import statistics
from datetime import datetime, timezone
from typing import Any, Optional

from promptchain.profiler.governor import (
    DEFAULT_MEASUREMENTS, GateDecision, GovernorClient, StreamResult,
    append_measurement, consume_stream, is_resident, snapshot,
)

SCHEMA = "model_measurement/1"

# Filler sentence repeated to reach each input size; the task line is always the same.
_FILLER = "The quick brown fox jumps over the lazy dog near the quiet river bank. "
_TASK = "\n\nIn one word, is the text above about an animal (yes or no)?"
INPUT_SIZES = {"small": 0, "medium": 40, "large": 160}  # filler repeats


def build_prompt(size: str) -> str:
    return _FILLER * INPUT_SIZES[size] + (_TASK if INPUT_SIZES[size] else "Answer yes or no: is a fox an animal?")


def think_param(tag: str, capabilities: list[str]) -> Any:
    """The think-off value this endpoint takes. gpt-oss ignores booleans and needs a level."""
    if "thinking" not in capabilities:
        return None
    return "low" if tag.startswith("gpt-oss") else False


def rates(res: StreamResult) -> dict:
    decode = (res.output_tokens / res.decode_s) if res.output_tokens and res.decode_s else None
    prefill = (res.prompt_tokens / res.prefill_s) if res.prompt_tokens and res.prefill_s else None
    return {"decode_tok_s": decode, "prefill_tok_s": prefill}


def measurement_record(*, kind: str, tag: str, quant: Optional[str], endpoint: str, think: Any,
                       warm: bool, size: Optional[str], res: StreamResult, gov: dict,
                       gate: Optional[GateDecision], extra: Optional[dict] = None) -> dict:
    """One line of the measurements store. Thinking tokens are chunk counts (an estimate:
    ollama does not report them separately); output_tokens includes them."""
    rec = {
        "schema": SCHEMA, "kind": kind, "model": tag, "quant": quant, "endpoint": endpoint,
        "ts": datetime.now(timezone.utc).isoformat(), "warm": warm, "input_size": size, "think_param": think,
        "metrics": {
            "ttft_s": res.ttft_s, "ttft_content_s": res.ttft_content_s, "total_s": res.total_s,
            "prompt_tokens": res.prompt_tokens, "output_tokens": res.output_tokens,
            "thinking_tokens_est": res.thinking_chunks, "thinking_chars": res.thinking_chars,
            "load_s": res.load_s, "prefill_s": res.prefill_s, "decode_s": res.decode_s,
            "done_reason": res.done_reason, "error": res.error, **rates(res),
        },
        "governor": gov, "gate": gate.to_dict() if gate else None,
    }
    if extra:
        rec.update(extra)
    return rec


def run_latency_probe(client: GovernorClient, tag: str, *, quant: Optional[str], capabilities: list[str],
                      sizes: tuple[str, ...] = ("small", "medium", "large"),
                      gate: Optional[GateDecision] = None, store: str = DEFAULT_MEASUREMENTS) -> list[dict]:
    """Run one call per size, record each. Only the first call can be cold."""
    records = []
    think = think_param(tag, capabilities)
    for size in sizes:
        before_p = client.pressure()
        before_b = client.budget()
        warm = is_resident(tag, before_p)
        payload = {"model": tag, "messages": [{"role": "user", "content": build_prompt(size)}]}
        if think is not None:
            payload["think"] = think
        res = consume_stream(client.chat_stream(payload))
        rec = measurement_record(kind="latency", tag=tag, quant=quant, endpoint=client.base + "/llm/ollama/api/chat",
                                 think=think, warm=warm, size=size, res=res,
                                 gov=snapshot(before_b, before_p), gate=gate)
        append_measurement(store, rec)
        records.append(rec)
    return records


def summarize(records: list[dict]) -> dict:
    def col(key):
        return [r["metrics"][key] for r in records if r["metrics"].get(key) is not None]

    def med(xs):
        return statistics.median(xs) if xs else None

    return {
        "n": len(records), "cold_first": bool(records) and not records[0]["warm"],
        "ttft_p50_s": med(col("ttft_s")), "decode_tok_s_p50": med(col("decode_tok_s")),
        "prefill_tok_s_p50": med(col("prefill_tok_s")),
        "thinking_tokens_per_call": med(col("thinking_tokens_est")),
        "total_s": sum(col("total_s")),
    }
