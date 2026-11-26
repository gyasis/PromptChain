"""governor — thin client for the Atelier governor plus the Layer-B availability gate.

Every local model call goes through the governor proxy (``POST /llm/ollama/api/chat``),
never raw ollama. The transport is injectable (``opener``) so tests run fully offline.
Sizes are never sent: no num_ctx, no num_predict.
"""
from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterator, Optional

DEFAULT_GOVERNOR = "http://192.168.0.159:8799"
CHAT_PATH = "/llm/ollama/api/chat"
SHOW_PATH = "/llm/ollama/api/show"

Opener = Callable[..., Any]


@dataclass(frozen=True)
class GateDecision:
    """Layer-B verdict for one model: run it, or drop it with the live numbers."""

    model: str
    admitted: bool
    reason: str
    resident: bool
    need_gb: Optional[float]
    free_budget_gb: Optional[float]
    level: Optional[str]
    queue_len: Optional[int]
    all_stop: Optional[bool]

    def to_dict(self) -> dict:
        return asdict(self)


def resident_tenants(pressure: dict) -> list[str]:
    """Names of the ollama tenants currently loaded."""
    return [t.get("name") for t in pressure.get("tenants", []) if t.get("tenant") == "ollama"]


def is_resident(tag: str, pressure: dict) -> bool:
    return tag in resident_tenants(pressure)


def gate_decision(tag: str, mem_gb_at_ctx: Optional[float], budget: dict, pressure: dict) -> GateDecision:
    """Admit a model only if it is resident, or fits the live budget with no queue/stop.

    Never evicts anything: a model that does not fit is DROPPED, not made room for.
    """
    free = budget.get("free_budget_gb")
    level = pressure.get("level")
    queue = budget.get("queue") or []
    all_stop = bool(budget.get("all_stop"))
    resident = is_resident(tag, pressure)

    def verdict(ok: bool, reason: str) -> GateDecision:
        return GateDecision(tag, ok, reason, resident, mem_gb_at_ctx, free, level, len(queue), all_stop)

    if resident:
        return verdict(True, "already resident")
    if mem_gb_at_ctx is None:
        return verdict(False, "need unknown: no mem_gb_at_ctx in catalog")
    if free is None or level is None:
        return verdict(False, "governor state unavailable")
    if all_stop:
        return verdict(False, "all_stop is set")
    if level != "ok":
        return verdict(False, f"pressure level {level!r} is not 'ok'")
    if queue:
        return verdict(False, f"governor queue not empty ({len(queue)})")
    if mem_gb_at_ctx > free:
        return verdict(False, f"needs {mem_gb_at_ctx} GB > free_budget {free} GB")
    return verdict(True, f"fits: needs {mem_gb_at_ctx} GB <= free_budget {free} GB")


def snapshot(budget: dict, pressure: dict) -> dict:
    """Compact governor state stored beside each measurement."""
    return {
        "free_budget_gb": budget.get("free_budget_gb"),
        "committed_gb": budget.get("committed_gb"),
        "queue_len": len(budget.get("queue") or []),
        "all_stop": budget.get("all_stop"),
        "active_leases": len(budget.get("active_leases") or []),
        "level": pressure.get("level"),
        "free_gb": pressure.get("free_gb"),
        "resident_gb": pressure.get("resident_gb"),
        "ollama_resident": resident_tenants(pressure),
    }


class GovernorClient:
    def __init__(self, base: str = DEFAULT_GOVERNOR, opener: Optional[Opener] = None,
                 connect_timeout: float = 5.0, read_timeout: float = 280.0):
        self.base = base.rstrip("/")
        self._open = opener or urllib.request.urlopen
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout

    def _request(self, path: str, payload: Optional[dict], timeout: float):
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(
            self.base + path, data=data,
            headers={"Content-Type": "application/json"} if data else {},
            method="POST" if data else "GET",
        )
        return self._open(req, timeout=timeout)

    def get_json(self, path: str) -> dict:
        with self._request(path, None, self.connect_timeout * 2) as r:
            return json.loads(r.read())

    def budget(self) -> dict:
        return self.get_json("/budget")

    def pressure(self) -> dict:
        return self.get_json("/pressure")

    def show(self, tag: str) -> dict:
        with self._request(SHOW_PATH, {"model": tag}, 30) as r:
            return json.loads(r.read())

    def chat_stream(self, payload: dict) -> Iterator[dict]:
        """Yield the NDJSON chunks of a streamed /api/chat reply."""
        body = dict(payload, stream=True)
        with self._request(CHAT_PATH, body, self.read_timeout) as r:
            for raw in r:
                raw = raw.strip()
                if raw:
                    yield json.loads(raw)


@dataclass
class StreamResult:
    """Everything measured from one streamed chat call."""

    ttft_s: Optional[float] = None
    ttft_content_s: Optional[float] = None
    total_s: float = 0.0
    prompt_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    thinking_chunks: int = 0
    thinking_chars: int = 0
    content: str = ""
    tool_calls: list = field(default_factory=list)
    prefill_s: Optional[float] = None
    decode_s: Optional[float] = None
    load_s: Optional[float] = None
    done_reason: Optional[str] = None
    error: Optional[str] = None


def consume_stream(chunks: Iterator[dict], clock: Callable[[], float] = time.monotonic) -> StreamResult:
    """Fold a chunk stream into a StreamResult. Ollama reports durations in nanoseconds."""
    res = StreamResult()
    t0 = clock()
    parts: list[str] = []
    for ch in chunks:
        now = clock() - t0
        if ch.get("error"):
            res.error = str(ch["error"])
            break
        msg = ch.get("message") or {}
        think = msg.get("thinking") or ""
        text = msg.get("content") or ""
        calls = msg.get("tool_calls") or []
        if (think or text or calls) and res.ttft_s is None:
            res.ttft_s = now
        if think:
            res.thinking_chunks += 1
            res.thinking_chars += len(think)
        if text:
            parts.append(text)
            if res.ttft_content_s is None:
                res.ttft_content_s = now
        if calls:
            res.tool_calls.extend(calls)
            if res.ttft_content_s is None:
                res.ttft_content_s = now
        if ch.get("done"):
            res.prompt_tokens = ch.get("prompt_eval_count")
            res.output_tokens = ch.get("eval_count")
            res.done_reason = ch.get("done_reason")
            for key, attr in (("prompt_eval_duration", "prefill_s"), ("eval_duration", "decode_s"),
                              ("load_duration", "load_s")):
                if ch.get(key) is not None:
                    setattr(res, attr, ch[key] / 1e9)
    res.total_s = clock() - t0
    res.content = "".join(parts)
    return res


def append_measurement(path, record: dict) -> None:
    """Append one JSON line to the measurements store (never the profile JSON)."""
    import os
    from pathlib import Path

    p = Path(os.path.expanduser(str(path)))
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


DEFAULT_MEASUREMENTS = "~/.promptchain/model_measurements.jsonl"
