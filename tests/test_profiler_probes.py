"""Offline tests for the latency and tool-calling probes (mocked HTTP, no network)."""
import io
import json

import pytest

from promptchain.profiler import governor, latency_probe, tool_probe
from promptchain.profiler.governor import GovernorClient, consume_stream, gate_decision


class FakeResp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def make_opener(routes):
    calls = []

    def opener(req, timeout=None):
        calls.append((req.full_url, req.data))
        for suffix, body in routes.items():
            if req.full_url.endswith(suffix):
                data = body if isinstance(body, bytes) else (
                    "\n".join(json.dumps(c) for c in body) + "\n").encode() if isinstance(body, list) else json.dumps(body).encode()
                return FakeResp(data)
        raise AssertionError(f"unexpected url {req.full_url}")

    opener.calls = calls
    return opener


BUDGET = {"free_budget_gb": 28.1, "committed_gb": 13.0, "queue": [], "all_stop": False, "active_leases": []}
PRESSURE = {"level": "ok", "free_gb": 35.1, "resident_gb": 28.9,
            "tenants": [{"tenant": "ollama", "name": "bge-m3:latest"}, {"tenant": "atelier", "name": "whisper"}]}


# --- gate ---------------------------------------------------------------
def test_gate_admits_fit():
    g = gate_decision("m:1", 20.0, BUDGET, PRESSURE)
    assert g.admitted and not g.resident


def test_gate_drops_too_big_with_numbers():
    g = gate_decision("m:1", 30.0, BUDGET, PRESSURE)
    assert not g.admitted and "30.0" in g.reason and g.free_budget_gb == 28.1


def test_gate_resident_always_admitted_even_if_big():
    p = dict(PRESSURE, tenants=[{"tenant": "ollama", "name": "m:1"}])
    assert gate_decision("m:1", 99, BUDGET, p).admitted


@pytest.mark.parametrize("budget,pressure,word", [
    (dict(BUDGET, all_stop=True), PRESSURE, "all_stop"),
    (dict(BUDGET, queue=[{"x": 1}]), PRESSURE, "queue"),
    (BUDGET, dict(PRESSURE, level="warn"), "level"),
])
def test_gate_drops_on_state(budget, pressure, word):
    g = gate_decision("m:1", 1.0, budget, pressure)
    assert not g.admitted and word in g.reason


def test_gate_unknown_need_dropped():
    assert not gate_decision("m:1", None, BUDGET, PRESSURE).admitted


# --- stream folding -----------------------------------------------------
STREAM = [
    {"message": {"thinking": "hm"}},
    {"message": {"thinking": "ok"}},
    {"message": {"content": "yes"}},
    {"done": True, "done_reason": "stop", "prompt_eval_count": 100, "eval_count": 50,
     "prompt_eval_duration": 500_000_000, "eval_duration": 2_000_000_000, "load_duration": 1_000_000_000,
     "message": {"content": ""}},
]


def test_consume_stream_metrics():
    ticks = iter([0.0, 0.1, 0.2, 0.3, 0.4, 1.0])
    res = consume_stream(iter(STREAM), clock=lambda: next(ticks))
    assert res.ttft_s == pytest.approx(0.1) and res.ttft_content_s == pytest.approx(0.3)
    assert res.thinking_chunks == 2 and res.content == "yes"
    assert res.prompt_tokens == 100 and res.output_tokens == 50
    r = latency_probe.rates(res)
    assert r["decode_tok_s"] == pytest.approx(25) and r["prefill_tok_s"] == pytest.approx(200)


def test_stream_error_chunk():
    res = consume_stream(iter([{"error": "boom"}]))
    assert res.error == "boom"


# --- latency probe ------------------------------------------------------
def test_think_param():
    assert latency_probe.think_param("x", ["tools"]) is None
    assert latency_probe.think_param("gpt-oss:20b", ["thinking"]) == "low"
    assert latency_probe.think_param("ornith:1", ["thinking"]) is False


def test_latency_probe_records_shape(tmp_path):
    op = make_opener({"/budget": BUDGET, "/pressure": PRESSURE, "/api/chat": STREAM})
    store = tmp_path / "m.jsonl"
    recs = latency_probe.run_latency_probe(GovernorClient("http://gov", op), "ornith:1", quant="Q5_K_M",
                                           capabilities=["thinking", "tools"], sizes=("small", "medium"),
                                           store=str(store))
    assert len(recs) == 2
    lines = [json.loads(x) for x in store.read_text().splitlines()]
    assert len(lines) == 2
    r = lines[0]
    assert r["schema"] == "model_measurement/1" and r["kind"] == "latency"
    assert r["model"] == "ornith:1" and r["quant"] == "Q5_K_M" and r["warm"] is False
    assert r["endpoint"].endswith("/llm/ollama/api/chat") and r["think_param"] is False
    assert {"ttft_s", "decode_tok_s", "prefill_tok_s", "thinking_tokens_est", "output_tokens"} <= set(r["metrics"])
    assert r["governor"]["free_budget_gb"] == 28.1
    chat_bodies = [json.loads(d) for u, d in op.calls if u.endswith("/api/chat")]
    assert all("num_ctx" not in b and "options" not in b for b in chat_bodies)
    assert chat_bodies[0]["think"] is False and chat_bodies[0]["stream"] is True


def test_summarize():
    op = make_opener({"/budget": BUDGET, "/pressure": PRESSURE, "/api/chat": STREAM})
    recs = latency_probe.run_latency_probe(GovernorClient("http://gov", op), "m", quant=None,
                                           capabilities=[], sizes=("small",), store="/dev/null")
    s = latency_probe.summarize(recs)
    assert s["n"] == 1 and s["cold_first"] is True


# --- tool probe ---------------------------------------------------------
SIMPLE = {"id": "simple_0", "category": "simple",
          "question": [[{"role": "user", "content": "area of triangle base 10 height 5"}]],
          "function": [{"name": "calc.triangle_area", "description": "d",
                        "parameters": {"type": "dict", "properties": {
                            "base": {"type": "integer"}, "height": {"type": "integer"},
                            "unit": {"type": "string"}}, "required": ["base", "height"]}}],
          "ground_truth": [{"calc.triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]}}]}
IRR = {"id": "irrelevance_0", "category": "irrelevance", "ground_truth": None,
       "question": [[{"role": "user", "content": "q"}]], "function": SIMPLE["function"]}


def call(name, args):
    return [{"function": {"name": name, "arguments": args}}]


def test_to_json_schema_types():
    s = tool_probe.to_json_schema({"type": "dict", "properties": {"a": {"type": "float"}, "b": {"type": "any"},
                                                                    "c": {"type": "array", "items": {"type": "integer"}}}})
    assert s["type"] == "object" and s["properties"]["a"]["type"] == "number"
    assert "type" not in s["properties"]["b"] and s["properties"]["c"]["items"]["type"] == "integer"


def test_sanitize_and_tools():
    t = tool_probe.to_ollama_tools(SIMPLE["function"])
    assert t[0]["function"]["name"] == "calc_triangle_area"


def test_score_pass():
    s = tool_probe.score_item(SIMPLE, call("calc_triangle_area", {"base": 10, "height": 5}))
    assert s["passed"] and s["name_ok"] and s["schema_ok"] and s["values_ok"]


def test_score_optional_and_float_equal():
    s = tool_probe.score_item(SIMPLE, call("calc_triangle_area", {"base": 10.0, "height": 5, "unit": "Units"}))
    assert s["passed"]


def test_score_wrong_name():
    s = tool_probe.score_item(SIMPLE, call("other", {"base": 10, "height": 5}))
    assert not s["name_ok"] and not s["passed"]


def test_score_schema_invalid_type():
    s = tool_probe.score_item(SIMPLE, call("calc_triangle_area", {"base": "ten", "height": 5}))
    assert s["name_ok"] and not s["schema_ok"] and not s["passed"]


def test_score_wrong_value_schema_valid():
    s = tool_probe.score_item(SIMPLE, call("calc_triangle_area", {"base": 11, "height": 5}))
    assert s["schema_ok"] and not s["values_ok"] and not s["passed"]


def test_score_missing_required_and_extra():
    assert not tool_probe.score_item(SIMPLE, call("calc_triangle_area", {"base": 10}))["schema_ok"]
    assert not tool_probe.score_item(SIMPLE, call("calc_triangle_area", {"base": 10, "height": 5, "zz": 1}))["schema_ok"]


def test_score_no_call_on_simple_fails():
    s = tool_probe.score_item(SIMPLE, [])
    assert not s["passed"] and s["errors"] == ["no tool call"]


def test_irrelevance_abstain():
    assert tool_probe.score_item(IRR, [])["passed"]
    assert not tool_probe.score_item(IRR, call("calc_triangle_area", {}))["passed"]


def test_arguments_as_json_string():
    s = tool_probe.score_item(SIMPLE, call("calc_triangle_area", json.dumps({"base": 10, "height": 5})))
    assert s["passed"]


def test_minimal_validator_without_jsonschema(monkeypatch):
    import builtins
    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "jsonschema":
            raise ImportError
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    sch = tool_probe.to_json_schema(SIMPLE["function"][0]["parameters"])
    assert tool_probe.validate_args({"base": 1, "height": 2}, sch) == []
    assert tool_probe.validate_args({"base": True, "height": 2}, sch)


def test_select_items_deterministic():
    items = list(range(100))
    assert tool_probe.select_items(items, 4) == [0, 25, 50, 75]
    assert tool_probe.select_items(items, 500) == items


def test_summarize_scores():
    sc = [tool_probe.score_item(SIMPLE, call("calc_triangle_area", {"base": 10, "height": 5})),
          tool_probe.score_item(IRR, [])]
    s = tool_probe.summarize_scores(sc)
    assert s["pass_rate"] == 1.0 and s["n_simple"] == 1 and s["abstention_rate"] == 1.0


def test_missing_bfcl_stops(tmp_path):
    with pytest.raises(tool_probe.BfclMissing):
        tool_probe.load_items(tmp_path)


def test_cached_bfcl_loads():
    bank = tool_probe.load_items()
    assert len(bank["simple"]) >= 100 and len(bank["irrelevance"]) >= 100
    assert bank["simple"][0]["ground_truth"]


def test_run_tool_probe_records(tmp_path):
    (tmp_path / "BFCL_v3_simple.json").write_text(json.dumps({k: v for k, v in SIMPLE.items() if k not in ("ground_truth", "category")}) + "\n")
    (tmp_path / "possible_answer_BFCL_v3_simple.json").write_text(json.dumps({"id": "simple_0", "ground_truth": SIMPLE["ground_truth"]}) + "\n")
    (tmp_path / "BFCL_v3_irrelevance.json").write_text(json.dumps({k: v for k, v in IRR.items() if k not in ("ground_truth", "category")}) + "\n")
    stream = [{"message": {"tool_calls": call("calc_triangle_area", {"base": 10, "height": 5})}},
              {"done": True, "prompt_eval_count": 9, "eval_count": 7, "eval_duration": 1_000_000_000,
               "prompt_eval_duration": 100_000_000, "message": {"content": ""}}]
    op = make_opener({"/budget": BUDGET, "/pressure": PRESSURE, "/api/chat": stream})
    store = tmp_path / "m.jsonl"
    out = tool_probe.run_tool_probe(GovernorClient("http://gov", op), "m:1", quant="q4", capabilities=["tools"],
                                    n_simple=1, n_irrelevance=1, data_dir=tmp_path, store=str(store))
    assert out["summary"]["n"] == 2 and out["scores"][0]["passed"] and not out["scores"][1]["passed"]
    rec = json.loads(store.read_text().splitlines()[0])
    assert rec["kind"] == "tool" and rec["item"]["license"] == "apache-2.0" and rec["score"]["passed"]
    body = json.loads([d for u, d in op.calls if u.endswith("/api/chat")][0])
    assert "tools" in body and "options" not in body


def test_tool_probe_skips_without_capability(tmp_path):
    out = tool_probe.run_tool_probe(GovernorClient("http://gov", make_opener({})), "m", quant=None,
                                    capabilities=["completion"], store=str(tmp_path / "x"))
    assert out["skipped"]
