"""tool_probe — native tool-calling probe scored on a pulled BFCL subset.

Uses the model's native ``tools`` / ``tool_calls`` (not JSON-in-text). Items come from the
Berkeley Function Calling Leaderboard (Apache-2.0), cached under ``data/probes/bfcl`` with
its source recorded; there is no self-written fallback item set. Each item is scored on:
correct tool name, schema-valid args, correct arg values, and abstention when no tool fits.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from promptchain.profiler.governor import (
    DEFAULT_MEASUREMENTS, GateDecision, GovernorClient, append_measurement, consume_stream, is_resident, snapshot,
)
from promptchain.profiler.latency_probe import measurement_record, think_param

BFCL_REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
BFCL_LICENSE = "apache-2.0"
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


def load_items(data_dir: Path = DEFAULT_DATA_DIR) -> dict[str, list[dict]]:
    """Load simple (with ground truth) and irrelevance items from the cache."""
    simple_p = data_dir / "BFCL_v3_simple.json"
    gt_p = data_dir / "possible_answer_BFCL_v3_simple.json"
    irr_p = data_dir / "BFCL_v3_irrelevance.json"
    for p in (simple_p, gt_p, irr_p):
        if not p.exists():
            raise BfclMissing(f"{p} missing; pull it from {BFCL_REPO} (licence {BFCL_LICENSE})")
    gt = {g["id"]: g["ground_truth"] for g in _read_jsonl(gt_p)}
    simple = [dict(i, ground_truth=gt[i["id"]], category="simple") for i in _read_jsonl(simple_p) if i["id"] in gt]
    irrelevance = [dict(i, ground_truth=None, category="irrelevance") for i in _read_jsonl(irr_p)]
    return {"simple": simple, "irrelevance": irrelevance}


def select_items(items: list[dict], n: int) -> list[dict]:
    """Deterministic, evenly spaced subset (no RNG, so every model sees the same items)."""
    if n >= len(items):
        return list(items)
    return [items[(i * len(items)) // n] for i in range(n)]


def validate_args(args: Any, schema: dict) -> list[str]:
    """Schema errors for tool-call args; jsonschema when installed, else a minimal check."""
    if not isinstance(args, dict):
        return ["arguments are not an object"]
    try:
        import jsonschema
        v = jsonschema.Draft7Validator(schema)
        errs = [e.message for e in v.iter_errors(args)]
        extra = [k for k in args if k not in schema.get("properties", {})]
        return errs + [f"unknown parameter {k!r}" for k in extra]
    except ImportError:
        pass
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


def _norm(v: Any) -> Any:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        return re.sub(r"[\s,./\-_*^]", "", v.lower())
    if isinstance(v, list):
        return [_norm(x) for x in v]
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in v.items()}
    return v


def values_match(args: dict, expected: dict) -> list[str]:
    """Compare against BFCL ground truth: each param maps to a list of allowed values;
    an allowed value of "" marks the parameter optional."""
    errs = []
    for param, allowed in expected.items():
        optional = "" in allowed
        if param not in args:
            if not optional:
                errs.append(f"missing {param!r}")
            continue
        if _norm(args[param]) not in [_norm(a) for a in allowed if a != ""]:
            errs.append(f"{param!r}={args[param]!r} not in allowed {allowed!r}")
    for param in args:
        if param not in expected:
            errs.append(f"unexpected {param!r}")
    return errs


def score_item(item: dict, tool_calls: list[dict]) -> dict:
    """Score one item. Irrelevance items pass only on abstention."""
    if item["category"] == "irrelevance":
        return {"id": item["id"], "category": "irrelevance", "abstained": not tool_calls,
                "name_ok": None, "schema_ok": None, "values_ok": None, "passed": not tool_calls, "errors": []}
    expected_name, expected_args = next(iter(item["ground_truth"][0].items()))
    schema = {sanitize_name(f["name"]): to_json_schema(f["parameters"]) for f in item["function"]}
    if not tool_calls:
        return {"id": item["id"], "category": "simple", "abstained": True, "name_ok": False,
                "schema_ok": False, "values_ok": False, "passed": False, "errors": ["no tool call"]}
    fn = (tool_calls[0].get("function") or {})
    name, args = fn.get("name"), fn.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            args = None
    name_ok = name == sanitize_name(expected_name)
    schema_errs = validate_args(args, schema.get(name, schema[sanitize_name(expected_name)]))
    value_errs = values_match(args, expected_args) if isinstance(args, dict) else ["no args"]
    return {"id": item["id"], "category": "simple", "abstained": False, "name_ok": name_ok,
            "schema_ok": not schema_errs, "values_ok": name_ok and not value_errs,
            "passed": name_ok and not schema_errs and not value_errs, "errors": schema_errs + value_errs}


def summarize_scores(scores: list[dict]) -> dict:
    simple = [s for s in scores if s["category"] == "simple"]
    irr = [s for s in scores if s["category"] == "irrelevance"]

    def rate(rows, key):
        return sum(1 for r in rows if r[key]) / len(rows) if rows else None

    return {"n": len(scores), "n_simple": len(simple), "n_irrelevance": len(irr),
            "name_rate": rate(simple, "name_ok"), "schema_rate": rate(simple, "schema_ok"),
            "values_rate": rate(simple, "values_ok"), "abstention_rate": rate(irr, "abstained"),
            "pass_rate": rate(scores, "passed"), "passed": sum(1 for s in scores if s["passed"])}


def run_tool_probe(client: GovernorClient, tag: str, *, quant: Optional[str], capabilities: list[str],
                   n_simple: int = 3, n_irrelevance: int = 2, data_dir: Path = DEFAULT_DATA_DIR,
                   gate: Optional[GateDecision] = None, store: str = DEFAULT_MEASUREMENTS) -> dict:
    """Run the BFCL subset against *tag*; one measurement line per item."""
    if "tools" not in capabilities:
        return {"skipped": "model does not advertise the tools capability", "scores": [], "summary": None}
    bank = load_items(data_dir)
    items = select_items(bank["simple"], n_simple) + select_items(bank["irrelevance"], n_irrelevance)
    think = think_param(tag, capabilities)
    scores = []
    for item in items:
        before_p, before_b = client.pressure(), client.budget()
        payload = {"model": tag, "messages": item["question"][0], "tools": to_ollama_tools(item["function"])}
        if think is not None:
            payload["think"] = think
        res = consume_stream(client.chat_stream(payload))
        score = score_item(item, res.tool_calls)
        score["error"] = res.error
        scores.append(score)
        append_measurement(store, measurement_record(
            kind="tool", tag=tag, quant=quant, endpoint=client.base + "/llm/ollama/api/chat", think=think,
            warm=is_resident(tag, before_p), size=None, res=res, gov=snapshot(before_b, before_p), gate=gate,
            extra={"item": {"id": item["id"], "category": item["category"], "source": BFCL_REPO,
                            "license": BFCL_LICENSE}, "score": score}))
    return {"scores": scores, "summary": summarize_scores(scores)}
