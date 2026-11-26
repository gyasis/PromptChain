"""latency_probe — TTFT / decode / prefill throughput probe through the Atelier governor.

Writes one raw measurement per call to a separate JSONL store; the persisted
CapabilityProfile is not touched. Thinking is turned off with the endpoint's own
parameter and the emitted reasoning is measured, so a flag is never taken as evidence.

Rules that keep the numbers honest:
- Decode speed comes only from a call whose prompt asks for a long answer, and only when
  ``eval_count`` reaches ``MIN_DECODE_TOKENS``; the count is stored beside every rate.
- TTFT is the time to first CONTENT token, reported per input size and warm calls only.
  Cold load time is reported separately.
- A call that errored, never finished or hit the length limit is ``status="error"`` with
  no rates, and is excluded from every summary.
"""
from __future__ import annotations

import re
import statistics
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from promptchain.profiler.governor import (
    DEFAULT_MEASUREMENTS, GateDecision, GovernorClient, StreamResult,
    append_measurement, consume_stream, is_resident, snapshot,
)

SCHEMA = "model_measurement/2"

# Filler sentence repeated to reach each input size; the task line is always the same.
_FILLER = "The quick brown fox jumps over the lazy dog near the quiet river bank. "
_TASK = "\n\nIn one word, is the text above about an animal (yes or no)?"
_DECODE_PROMPT = ("Write a detailed explanation, of about 300 words, of how a hash table works, "
                  "covering hashing, collisions, load factor and resizing.")
COLD_LOAD_S = 0.5  # load_duration above this means the weights were (re)loaded for the call
MIN_DECODE_TOKENS = 128  # a decode rate over fewer emitted tokens is mostly startup, not decode
MIN_PREFILL_TOKENS = 100
INPUT_SIZES = {"small": 0, "medium": 40, "large": 160}  # filler repeats
# small runs twice so a warm small-input TTFT exists even when the first call loaded the model.
DEFAULT_SEQUENCE = ("small", "small", "medium", "large", "decode")


@dataclass
class RunContext:
    """Provenance stamped on every measurement line."""

    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    git_sha: Optional[str] = None
    bfcl_revision: Optional[str] = None
    digest: Optional[str] = None


@lru_cache(maxsize=1)
def repo_git_sha() -> Optional[str]:
    try:
        out = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


_QUANT_RE = re.compile(r"(?i)(?:^|[-_:])((?:UD-)?(?:I?Q\d(?:_[A-Z0-9]+)*|BF16|F16|FP16))(?:$|[-_])")


def quant_from_tag(tag: str) -> Optional[str]:
    """The quantisation named in the tag itself (``...:UD-Q4_K_XL``), else None."""
    m = _QUANT_RE.search(tag.rsplit("/", 1)[-1])
    return m.group(1) if m else None


def quant_from_show(show: dict) -> Optional[str]:
    q = (show.get("details") or {}).get("quantization_level")
    return None if q in (None, "", "unknown") else q


def build_prompt(size: str) -> str:
    if size == "decode":
        return _DECODE_PROMPT
    return _FILLER * INPUT_SIZES[size] + (_TASK if INPUT_SIZES[size] else "Answer yes or no: is a fox an animal?")


def think_param(tag: str, capabilities: list[str]) -> Any:
    """The think-off value this endpoint takes. gpt-oss ignores booleans and needs a level."""
    if "thinking" not in capabilities:
        return None
    return "low" if tag.startswith("gpt-oss") else False


def is_warm(warm_by_pressure: bool, load_s: Optional[float]) -> bool:
    """load_s, when reported, decides: the pressure poller lags a fresh load."""
    if load_s is not None:
        return load_s < COLD_LOAD_S
    return warm_by_pressure


def thinking_leak(think: Any, res: StreamResult) -> bool:
    """A think-off call that still reasoned (reasoning chunks, or ``<think>`` in the content)."""
    return think is False and (res.thinking_chunks > 0 or "<think>" in res.content)


def rates(res: StreamResult) -> dict:
    """Throughput, only where the token count makes it meaningful; None otherwise."""
    ok = not res.failed
    decode = prefill = None
    if ok and res.output_tokens and res.decode_s and res.output_tokens >= MIN_DECODE_TOKENS:
        decode = res.output_tokens / res.decode_s
    if ok and res.prompt_tokens and res.prefill_s and res.prompt_tokens >= MIN_PREFILL_TOKENS:
        prefill = res.prompt_tokens / res.prefill_s
    return {"decode_tok_s": decode, "decode_eval_count": res.output_tokens,
            "prefill_tok_s": prefill, "prefill_prompt_tokens": res.prompt_tokens}


def route_string(endpoint: str, tag: str) -> str:
    return f"{endpoint} model={tag}"


def measurement_record(*, kind: str, tag: str, show: Optional[dict] = None, endpoint: str, think: Any,
                       warm: bool, size: Optional[str], res: StreamResult, gov: dict,
                       gate: Optional[GateDecision], ctx: Optional[RunContext] = None,
                       extra: Optional[dict] = None) -> dict:
    """One line of the measurements store. Thinking tokens are chunk counts (an estimate:
    ollama does not report them separately); output_tokens includes them."""
    ctx = ctx or RunContext()
    rec = {
        "schema": SCHEMA, "kind": kind, "run_id": ctx.run_id, "git_sha": ctx.git_sha or repo_git_sha(),
        "bfcl_revision": ctx.bfcl_revision, "digest": ctx.digest, "model": tag,
        "quant_show": quant_from_show(show or {}), "quant_tag": quant_from_tag(tag),
        "endpoint": endpoint, "route": route_string(endpoint, tag),
        "ts": datetime.now(timezone.utc).isoformat(),
        "status": "error" if res.failed else "ok", "failure_reason": res.failure_reason,
        "warm": is_warm(warm, res.load_s), "warm_by_pressure": warm,
        "input_size": size, "think_param": think, "thinking_leak": thinking_leak(think, res),
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


def run_latency_probe(client: GovernorClient, tag: str, *, show: Optional[dict] = None, capabilities: list[str],
                      sequence: tuple[str, ...] = DEFAULT_SEQUENCE, ctx: Optional[RunContext] = None,
                      gate: Optional[GateDecision] = None, store: str = DEFAULT_MEASUREMENTS) -> list[dict]:
    """One call per entry of *sequence*; each is recorded. Only the first call can be cold."""
    records = []
    think = think_param(tag, capabilities)
    seen: dict[str, int] = {}
    for size in sequence:
        before_p, before_b = client.pressure(), client.budget()
        payload = {"model": tag, "messages": [{"role": "user", "content": build_prompt(size)}]}
        if think is not None:
            payload["think"] = think
        res = consume_stream(client.chat_stream(payload))
        seen[size] = seen.get(size, 0) + 1
        rec = measurement_record(kind="latency", tag=tag, show=show, endpoint=client.base + "/llm/ollama/api/chat",
                                 think=think, warm=is_resident(tag, before_p), size=size, res=res,
                                 gov=snapshot(before_b, before_p), gate=gate, ctx=ctx,
                                 extra={"repeat": seen[size]})
        append_measurement(store, rec)
        records.append(rec)
    return records


def _med(xs):
    return statistics.median(xs) if xs else None


def summarize(records: list[dict]) -> dict:
    """Per-model summary. Errored calls are excluded; TTFT is per input size, warm calls
    only, first content token; cold load time is reported on its own."""
    ok = [r for r in records if r.get("status") == "ok"]
    ttft = {}
    for size in INPUT_SIZES:
        xs = [r["metrics"]["ttft_content_s"] for r in ok
              if r["input_size"] == size and r["warm"] and r["metrics"]["ttft_content_s"] is not None]
        ttft[size] = _med(xs)
    col = lambda key, rows=ok: [r["metrics"][key] for r in rows if r["metrics"].get(key) is not None]  # noqa: E731
    cold = [r for r in ok if not r["warm"]]
    return {
        "n": len(records), "n_ok": len(ok), "n_error": len(records) - len(ok),
        "ttft_content_p50_s_by_size": ttft,
        "cold_load_s": _med(col("load_s", cold)), "n_cold": len(cold),
        "decode_tok_s_p50": _med(col("decode_tok_s")),
        "decode_eval_counts": col("decode_eval_count", [r for r in ok if r["metrics"].get("decode_tok_s")]),
        "prefill_tok_s_p50": _med(col("prefill_tok_s")),
        "thinking_tokens_per_call": _med([r["metrics"]["thinking_tokens_est"] for r in ok]),
        "think_param": records[0]["think_param"] if records else None,
        "thinking_leaks": sum(1 for r in ok if r["thinking_leak"]),
        "total_s": sum(r["metrics"]["total_s"] for r in records),
    }
