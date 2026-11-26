"""tool_probe — native tool-calling probe scored on a pulled BFCL subset.

Uses the model's native ``tools`` / ``tool_calls`` (not JSON-in-text). Items come from the
Berkeley Function Calling Leaderboard (Apache-2.0), cached under ``data/probes/bfcl`` with
its source recorded; there is no self-written fallback item set. Each item is scored on:
correct tool name, schema-valid args, correct arg values, and abstention when no tool fits.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

from promptchain.profiler.governor import (
    DEFAULT_MEASUREMENTS, GateDecision, GovernorClient, append_measurement, consume_stream, is_resident, snapshot,
)
from promptchain.profiler.latency_probe import RunContext, measurement_record, think_param

BFCL_REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
BFCL_LICENSE = "apache-2.0"
log = logging.getLogger(__name__)
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "probes" / "bfcl"

_TYPE_MAP = {"dict": "object", "float": "number", "integer": "integer", "string": "string",
             "boolean": "boolean", "array": "array", "tuple": "array", "number": "number",
             "object": "object", "list": "array", "bool": "boolean", "int": "integer"}


class BfclMissing(RuntimeError):
    """The cached BFCL files are absent: stop this probe, never substitute our own items."""


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", name)


def to_json_schema(node: Any) -> Any:
    """Convert BFCL's python-ish type names to JSON Schema, recursively."""
    if isinstance(node, list):
        return [to_json_schema(x) for x in node]
    if not isinstance(node, dict):
        return node
    out = {}
    for k, v in node.items():
        if k == "type" and isinstance(v, str):
            if v != "any":
                out[k] = _TYPE_MAP.get(v, v)
        elif k in ("properties",) and isinstance(v, dict):
            out[k] = {pk: to_json_schema(pv) for pk, pv in v.items()}
        else:
            out[k] = to_json_schema(v)
    return out


def to_ollama_tools(functions: list[dict]) -> list[dict]:
    return [{"type": "function", "function": {"name": sanitize_name(f["name"]),
             "description": f.get("description", ""), "parameters": to_json_schema(f["parameters"])}}
            for f in functions]


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_source(data_dir: Path = DEFAULT_DATA_DIR) -> dict:
    """Pinned dataset provenance (repo, revision, licence, file hashes), or {} if absent."""
    p = data_dir / "SOURCE.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _data_error(item: dict) -> Optional[str]:
    """Why a simple item cannot be scored (ground truth does not match its own function list)."""
    gt = item.get("ground_truth") or []
    if len(gt) != 1:
        return f"ground truth has {len(gt)} calls"
    name = next(iter(gt[0]))
    names = {sanitize_name(f["name"]) for f in item["function"]}
    if sanitize_name(name) not in names:
        return f"ground-truth function {name!r} not among {sorted(names)}"
    return None


def load_items(data_dir: Path = DEFAULT_DATA_DIR) -> dict:
    """Load simple (with ground truth) and irrelevance items from the cache.

    Items whose ground truth names a function the item does not define are data errors:
    skipped with a logged reason and listed under ``skipped``; the run continues."""
    simple_p = data_dir / "BFCL_v3_simple.json"
    gt_p = data_dir / "possible_answer_BFCL_v3_simple.json"
    irr_p = data_dir / "BFCL_v3_irrelevance.json"
    for p in (simple_p, gt_p, irr_p):
        if not p.exists():
            raise BfclMissing(f"{p} missing; pull it from {BFCL_REPO} (licence {BFCL_LICENSE})")
    gt = {g["id"]: g["ground_truth"] for g in _read_jsonl(gt_p)}
    simple, skipped = [], []
    for i in _read_jsonl(simple_p):
        if i["id"] not in gt:
            continue
        item = dict(i, ground_truth=gt[i["id"]], category="simple")
        err = _data_error(item)
        if err:
            log.warning("skipping BFCL item %s: %s", i["id"], err)
            skipped.append({"id": i["id"], "reason": err})
        else:
            simple.append(item)
    irrelevance = [dict(i, ground_truth=None, category="irrelevance") for i in _read_jsonl(irr_p)]
    return {"simple": simple, "irrelevance": irrelevance, "skipped": skipped}


def select_items(items: list[dict], n: int) -> list[dict]:
    """Deterministic, evenly spaced subset (no RNG, so every model sees the same items)."""
    if n >= len(items):
        return list(items)
    return [items[(i * len(items)) // n] for i in range(n)]


def has_dict_truth(item: dict) -> bool:
    """True when any ground-truth value is a dict or a list of dicts (BFCL dict/list checker)."""
    def nested(v):
        return isinstance(v, dict) or (isinstance(v, list) and any(nested(x) for x in v))
    gt = item.get("ground_truth") or []
    return bool(gt) and any(nested(c) for al in next(iter(gt[0].values())).values() for c in al)


def pick_pilot_items(bank: dict, n_simple: int, n_irrelevance: int, require_dict: bool = True) -> list[dict]:
    """Evenly spaced items; the last simple slot is swapped for the first dict-truth item
    (lowest id order) when none was picked, so the nested matcher is always exercised."""
    simple = select_items(bank["simple"], n_simple)
    if require_dict and n_simple and not any(has_dict_truth(i) for i in simple):
        extra = next((i for i in bank["simple"] if has_dict_truth(i)), None)
        if extra:
            simple[-1] = extra
    return simple + select_items(bank["irrelevance"], n_irrelevance)


def validator_name() -> str:
    try:
        import jsonschema  # noqa: F401
        return "jsonschema"
    except ImportError:
        return "fallback"


def validate_args(args: Any, schema: dict) -> list[str]:
    """Schema errors for tool-call args; jsonschema when installed, else a minimal check."""
    if not isinstance(args, dict):
        return ["arguments are not an object"]
    if validator_name() == "jsonschema":
        import jsonschema
        errs = [e.message for e in jsonschema.Draft7Validator(schema).iter_errors(args)]
        extra = [k for k in args if k not in schema.get("properties", {})]
        return errs + [f"unknown parameter {k!r}" for k in extra]
    errs = [f"missing required {r!r}" for r in schema.get("required", []) if r not in args]
    props = schema.get("properties", {})
    checks = {"string": str, "boolean": bool, "array": list, "object": dict,
              "integer": int, "number": (int, float)}
    for k, val in args.items():
        if k not in props:
            errs.append(f"unknown parameter {k!r}")
            continue
        t = props[k].get("type")
        if t in checks and (not isinstance(val, checks[t]) or (t != "boolean" and isinstance(val, bool))):
            errs.append(f"{k!r} should be {t}")
    return errs


def _norm_str(v: str) -> str:
    return re.sub(r"[\s,./\-_*^]", "", v.lower())


def _scalar_match(v: Any, exp: Any, ptype: Optional[str]) -> bool:
    if isinstance(exp, bool):
        return isinstance(v, bool) and v == exp
    if isinstance(exp, (int, float)):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return False
        strict_int = ptype == "integer" or (ptype is None and isinstance(exp, int))
        if strict_int and isinstance(v, float):
            return False  # BFCL: a float is not an integer, even 10.0
        return float(v) == float(exp)
    if isinstance(exp, str):
        return isinstance(v, str) and _norm_str(v) == _norm_str(exp)
    return v == exp


def _match_candidate(v: Any, cand: Any, ptype: Optional[str] = None) -> bool:
    """BFCL dict_checker / list_checker: a dict candidate maps each key to its own allowed
    list ("" = optional); a list candidate matches element-wise; scalars compare directly."""
    if isinstance(cand, dict):
        if not isinstance(v, dict) or any(k not in cand for k in v):
            return False
        return all((k in v and _match_allowed(v[k], al)) or (k not in v and "" in al) for k, al in cand.items())
    if isinstance(cand, list):
        return isinstance(v, list) and len(v) == len(cand) and all(_match_candidate(x, c) for x, c in zip(v, cand))
    return _scalar_match(v, cand, ptype)


def _match_allowed(v: Any, allowed: list, ptype: Optional[str] = None) -> bool:
    return any(_match_candidate(v, c, ptype) for c in allowed if c != "")


def values_match(args: dict, expected: dict, schema: Optional[dict] = None) -> list[str]:
    """Compare against BFCL ground truth: each param maps to a list of allowed values
    (scalars, dicts or lists, recursively); an allowed value of "" marks it optional."""
    props = (schema or {}).get("properties", {})
    errs = []
    for param, allowed in expected.items():
        if param not in args:
            if "" not in allowed:
                errs.append(f"missing {param!r}")
            continue
        if not _match_allowed(args[param], allowed, (props.get(param) or {}).get("type")):
            errs.append(f"{param!r}={args[param]!r} not in allowed {allowed!r}")
    for param in args:
        if param not in expected:
            errs.append(f"unexpected {param!r}")
    return errs


def _row(item: dict, **kw) -> dict:
    base = {"id": item["id"], "category": item["category"], "status": "ok", "abstained": None, "name_ok": None,
            "schema_ok": None, "values_ok": None, "count_ok": None, "passed": None, "errors": [],
            "validator": validator_name()}
    base.update(kw)
    return base


def score_item(item: dict, tool_calls: list[dict]) -> dict:
    """Score one item. Irrelevance items pass only on abstention; simple items need exactly
    one call, with the right name, schema-valid args and the right values."""
    if item["category"] == "irrelevance":
        return _row(item, abstained=not tool_calls, passed=not tool_calls)
    err = _data_error(item)
    if err:
        return _row(item, status="data_error", errors=[err])
    expected_name, expected_args = next(iter(item["ground_truth"][0].items()))
    schema = {sanitize_name(f["name"]): to_json_schema(f["parameters"]) for f in item["function"]}
    want = sanitize_name(expected_name)
    if not tool_calls:
        return _row(item, abstained=True, name_ok=False, schema_ok=False, values_ok=False, count_ok=False,
                    passed=False, errors=["no tool call"])
    fn = (tool_calls[0].get("function") or {})
    name, args = fn.get("name"), fn.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            args = None
    name_ok = name == want
    sch = schema.get(name) if name in schema else schema[want]
    schema_errs = validate_args(args, sch)
    value_errs = values_match(args, expected_args, schema[want]) if isinstance(args, dict) else ["no args"]
    count_ok = len(tool_calls) == 1
    errors = schema_errs + value_errs + ([] if count_ok else [f"expected 1 call, got {len(tool_calls)}"])
    return _row(item, abstained=False, name_ok=name_ok, schema_ok=not schema_errs,
                values_ok=name_ok and not value_errs, count_ok=count_ok,
                passed=name_ok and not schema_errs and not value_errs and count_ok, errors=errors)


def error_row(item: dict, reason: str) -> dict:
    """A failed call is a failure, never a result: no pass/fail, excluded from every rate."""
    return _row(item, status="error", errors=[reason])


def summarize_scores(scores: list[dict]) -> dict:
    ok = [s for s in scores if s["status"] == "ok"]
    simple = [s for s in ok if s["category"] == "simple"]
    irr = [s for s in ok if s["category"] == "irrelevance"]

    def rate(rows, key):
        return sum(1 for r in rows if r[key]) / len(rows) if rows else None

    return {"n": len(scores), "n_scored": len(ok), "n_error": sum(1 for s in scores if s["status"] == "error"),
            "n_data_error": sum(1 for s in scores if s["status"] == "data_error"),
            "n_simple": len(simple), "n_irrelevance": len(irr),
            "name_rate": rate(simple, "name_ok"), "schema_rate": rate(simple, "schema_ok"),
            "values_rate": rate(simple, "values_ok"), "abstention_rate": rate(irr, "abstained"),
            "pass_rate": rate(ok, "passed"), "passed": sum(1 for s in ok if s["passed"]),
            "validator": validator_name()}


def run_tool_probe(client: GovernorClient, tag: str, *, show: Optional[dict] = None, capabilities: list[str],
                   n_simple: int = 3, n_irrelevance: int = 2, data_dir: Path = DEFAULT_DATA_DIR,
                   ctx: Optional[RunContext] = None, gate: Optional[GateDecision] = None,
                   store: str = DEFAULT_MEASUREMENTS) -> dict:
    """Run the BFCL subset against *tag*; one measurement line per item."""
    if "tools" not in capabilities:
        return {"skipped": "model does not advertise the tools capability", "scores": [], "summary": None}
    bank = load_items(data_dir)
    items = pick_pilot_items(bank, n_simple, n_irrelevance)
    think = think_param(tag, capabilities)
    scores = []
    for item in items:
        before_p, before_b = client.pressure(), client.budget()
        payload = {"model": tag, "messages": item["question"][0], "tools": to_ollama_tools(item["function"])}
        if think is not None:
            payload["think"] = think
        res = consume_stream(client.chat_stream(payload))
        score = error_row(item, res.failure_reason) if res.failed else score_item(item, res.tool_calls)
        scores.append(score)
        append_measurement(store, measurement_record(
            kind="tool", tag=tag, show=show, endpoint=client.base + "/llm/ollama/api/chat", think=think,
            warm=is_resident(tag, before_p), size=None, res=res, gov=snapshot(before_b, before_p), gate=gate, ctx=ctx,
            extra={"item": {"id": item["id"], "category": item["category"], "source": BFCL_REPO,
                            "license": BFCL_LICENSE}, "score": score, "validator": validator_name()}))
    return {"scores": scores, "summary": summarize_scores(scores), "skipped_items": bank["skipped"]}
