"""governor — thin client for the Atelier governor plus the Layer-B availability gate.

Every local model call goes through the governor proxy (``POST /llm/ollama/api/chat``),
never raw ollama. The transport is injectable (``opener``) so tests run fully offline.
Sizes are never sent: no num_ctx, no num_predict.
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Iterator, Optional

DEFAULT_GOVERNOR = "http://192.168.0.159:8799"
CHAT_PATH = "/llm/ollama/api/chat"
SHOW_PATH = "/llm/ollama/api/show"
LLAMACPP_CHAT_PATH = "/llm/llamacpp/v1/chat/completions"
LLAMACPP_MODELS_PATH = "/llm/llamacpp/models"
LLAMACPP_LOAD_WAIT_S = 300.0  # the llamacpp sidecar's own bound on a model load (LLAMACPP_START_TIMEOUT)

Opener = Callable[..., Any]


class GovernorRefusal(RuntimeError):
    """The governor answered 503 (e.g. ``model_too_large``). Never retried: recorded as DROPPED."""

    def __init__(self, body: dict, raw: str):
        self.body, self.raw = body, raw
        self.code = body.get("code")
        text = f"{body.get('reason', '')} {body.get('error', '')}"
        m = re.search(r"est ([\d.]+)\s*GB", text)
        self.est_gb = float(m.group(1)) if m else None
        m = re.search(r"ceiling ([\d.]+)\s*GB", text)
        self.ceiling_gb = float(m.group(1)) if m else None
        self.retry_after_s = body.get("retry_after_s")
        super().__init__(f"governor 503 {self.code}: {body.get('error') or raw[:200]}")


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


def llamacpp_loaded(pressure: dict) -> Optional[str]:
    """The alias the llamacpp lane has loaded, from its ``atelier`` tenant in /pressure.

    None when the tenant is absent, cold, loading or errored, or names no model."""
    for t in pressure.get("tenants", []):
        if t.get("tenant") == "atelier" and t.get("name") == "llamacpp":
            if t.get("state") in ("cold", "loading") or t.get("error"):
                return None
            return t.get("model")
    return None


def gate_decision(tag: str, mem_gb_at_ctx: Optional[float], budget: dict, pressure: dict,
                  governor_est_gb: Optional[float] = None) -> GateDecision:
    """Admit a model only if the live state allows it. Resident models need 0 GB more,
    but every other check (all_stop, level, queue) still applies.

    A governor-supplied estimate (from a refusal) overrides the catalog's ``mem_gb_at_ctx``.
    Never evicts anything: a model that does not fit is DROPPED, not made room for.
    """
    free = budget.get("free_budget_gb")
    level = pressure.get("level")
    queue = budget.get("queue") or []
    all_stop = bool(budget.get("all_stop"))
    resident = is_resident(tag, pressure)
    catalog_need = governor_est_gb if governor_est_gb is not None else mem_gb_at_ctx
    need = 0.0 if resident else catalog_need

    def verdict(ok: bool, reason: str) -> GateDecision:
        return GateDecision(tag, ok, reason, resident, need, free, level, len(queue), all_stop)

    if all_stop:
        return verdict(False, "all_stop is set")
    if level is None or free is None:
        return verdict(False, "governor state unavailable")
    if level != "ok":
        return verdict(False, f"pressure level {level!r} is not 'ok'")
    if queue:
        return verdict(False, f"governor queue not empty ({len(queue)})")
    if need is None:
        return verdict(False, "need unknown: no mem_gb_at_ctx in catalog")
    if need > free:
        return verdict(False, f"needs {need} GB > free_budget {free} GB")
    if resident:
        return verdict(True, "already resident; all checks pass")
    src = "governor estimate" if governor_est_gb is not None else "catalog"
    return verdict(True, f"fits: needs {need} GB ({src}) <= free_budget {free} GB")


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
        "llamacpp_loaded": llamacpp_loaded(pressure),
    }


class GovernorClient:
    chat_path = CHAT_PATH

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
        try:
            return self._open(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            if e.code != 503:
                raise
            raw = e.read().decode("utf-8", "replace")
            try:
                body = json.loads(raw)
            except ValueError:
                body = {}
            raise GovernorRefusal(body if isinstance(body, dict) else {}, raw) from e

    def get_json(self, path: str) -> dict:
        with self._request(path, None, self.connect_timeout * 2) as r:
            return json.loads(r.read())

    def resident(self, tag: str, pressure: dict) -> bool:
        """Whether *tag* is already loaded on this client's lane, per a /pressure snapshot."""
        return is_resident(tag, pressure)

    def budget(self) -> dict:
        return self.get_json("/budget")

    def pressure(self) -> dict:
        return self.get_json("/pressure")

    def digest(self, tag: str) -> Optional[str]:
        """The model digest. /api/show carries none on this ollama, so it comes from /api/tags."""
        try:
            for m in self.get_json("/llm/ollama/api/tags").get("models", []):
                if tag in (m.get("name"), m.get("model")):
                    return m.get("digest")
        except Exception:
            return None
        return None

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


def _parse_args(text: str) -> Any:
    """Tool-call arguments arrive as a JSON string; Ollama-shaped chunks carry the decoded object."""
    try:
        return json.loads(text) if text else {}
    except ValueError:
        return text


def sse_to_chunks(lines: Iterable[bytes]) -> Iterator[dict]:
    """Translate an OpenAI-style SSE stream (llama-server) into Ollama-shaped chat chunks.

    Reasoning (``reasoning_content``) becomes ``message.thinking``. Tool-call fragments are
    accumulated per index and emitted once, with decoded arguments, on the final chunk.
    The final chunk carries llama-server's ``timings`` (ms) as Ollama's ns durations and
    token counts; ``usage`` fills the counts when no ``timings`` object is present.
    """
    calls: dict[int, dict] = {}
    finish: Optional[str] = None
    timings: dict = {}
    usage: dict = {}
    for raw in lines:
        raw = raw.strip()
        if not raw.startswith(b"data:"):
            continue
        body = raw[5:].strip()
        if body == b"[DONE]":
            break
        ev = json.loads(body)
        if ev.get("error"):
            yield {"error": ev["error"].get("message") if isinstance(ev["error"], dict) else ev["error"]}
            return
        timings = ev.get("timings") or timings
        usage = ev.get("usage") or usage
        for choice in ev.get("choices") or []:
            delta = choice.get("delta") or {}
            for frag in delta.get("tool_calls") or []:
                slot = calls.setdefault(frag.get("index", 0), {"name": "", "args": ""})
                fn = frag.get("function") or {}
                slot["name"] += fn.get("name") or ""
                slot["args"] += fn.get("arguments") or ""
            msg = {"role": "assistant", "content": delta.get("content") or "",
                   "thinking": delta.get("reasoning_content") or ""}
            if msg["content"] or msg["thinking"]:
                yield {"message": msg, "done": False}
            finish = choice.get("finish_reason") or finish
    final: dict = {"message": {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": c["name"], "arguments": _parse_args(c["args"])}}
        for _, c in sorted(calls.items())]}, "done": True,
        "done_reason": "length" if finish == "length" else (finish and "stop")}
    if timings:
        final.update(prompt_eval_count=timings.get("prompt_n"), eval_count=timings.get("predicted_n"),
                     prompt_eval_duration=int((timings.get("prompt_ms") or 0) * 1e6),
                     eval_duration=int((timings.get("predicted_ms") or 0) * 1e6))
    else:
        final.update(prompt_eval_count=usage.get("prompt_tokens"), eval_count=usage.get("completion_tokens"))
    if not final["message"]["tool_calls"]:
        del final["message"]["tool_calls"]
    if finish is None:
        final["done"] = False
    yield final


class LlamacppGovernorClient(GovernorClient):
    """The same interface over the governor's llamacpp lane (``/llm/llamacpp/v1``).

    Probes keep speaking Ollama: the request is translated to OpenAI chat-completions
    (``think:false`` becomes ``chat_template_kwargs.enable_thinking=false``; ``options`` are
    dropped, so no size is ever sent) and the SSE reply is translated back. llama-server
    reports no load time, so ``load_duration`` is absent and warmth comes from the lane's
    ``atelier`` tenant in /pressure (the alias it has loaded).

    The first call for an alias may wait on a model load, so its read timeout is
    ``read_timeout + load_wait``; later calls use ``read_timeout``.
    """

    chat_path = LLAMACPP_CHAT_PATH

    def __init__(self, *args: Any, load_wait: float = LLAMACPP_LOAD_WAIT_S, **kw: Any):
        super().__init__(*args, **kw)
        self.load_wait = load_wait
        self._called: set[str] = set()

    def resident(self, tag: str, pressure: dict) -> bool:
        return llamacpp_loaded(pressure) == tag

    def timeout_for(self, tag: str) -> float:
        return self.read_timeout if tag in self._called else self.read_timeout + self.load_wait

    def digest(self, tag: str) -> Optional[str]:
        """The llamacpp lane exposes no digest; the alias is the stable identity."""
        return f"llamacpp:{tag}"

    def show(self, tag: str) -> dict:
        """No /api/show on this lane. Capabilities are what the lane proves: a listed model
        completes and reasons; ``tools`` is advertised because llama-server parses tool calls
        from the chat template, and a model that fails to call scores 0 in the tool probe."""
        ids = [m.get("id") for m in self.get_json(LLAMACPP_MODELS_PATH).get("data", [])]
        return {"model": tag, "loaded": tag in ids, "digest": self.digest(tag),
                "capabilities": ["completion", "tools", "thinking"], "details": {}}

    def chat_stream(self, payload: dict) -> Iterator[dict]:
        body: dict = {"model": payload["model"], "messages": payload["messages"], "stream": True}
        if payload.get("tools"):
            body["tools"] = payload["tools"]
        if payload.get("think") is False:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        timeout = self.timeout_for(payload["model"])
        self._called.add(payload["model"])
        with self._request(self.chat_path, body, timeout) as r:
            yield from sse_to_chunks(r)


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
    done: bool = False

    @property
    def failed(self) -> bool:
        """A stream that errored, never finished, or hit the length limit is a failure, not a result."""
        return bool(self.error) or not self.done or self.done_reason == "length"

    @property
    def failure_reason(self) -> Optional[str]:
        if self.error:
            return f"error: {self.error}"
        if not self.done:
            return "no done chunk"
        if self.done_reason == "length":
            return "done_reason=length"
        return None


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
            res.done = True
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


ROTATE_BYTES = 10 * 1024 * 1024
ROTATE_KEEP = 3


def _rotate(p, max_bytes: int, keep: int) -> None:
    """store.jsonl -> store.jsonl.1 -> .2 -> .3; the oldest is overwritten."""
    if not p.exists() or p.stat().st_size < max_bytes:
        return
    for i in range(keep - 1, 0, -1):
        src = p.with_name(f"{p.name}.{i}")
        if src.exists():
            src.replace(p.with_name(f"{p.name}.{i + 1}"))
    p.replace(p.with_name(f"{p.name}.1"))


def append_measurement(path, record: dict, max_bytes: int = ROTATE_BYTES, keep: int = ROTATE_KEEP) -> None:
    """Append one JSON line to the measurements store (never the profile JSON); size-rotated."""
    import os
    from pathlib import Path

    p = Path(os.path.expanduser(str(path)))
    p.parent.mkdir(parents=True, exist_ok=True)
    _rotate(p, max_bytes, keep)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


DEFAULT_MEASUREMENTS = "~/.promptchain/model_measurements.jsonl"
