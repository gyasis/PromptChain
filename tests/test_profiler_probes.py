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


RESIDENT = dict(PRESSURE, tenants=[{"tenant": "ollama", "name": "m:1"}])


def test_gate_resident_needs_zero_even_if_catalog_big():
    g = gate_decision("m:1", 99, BUDGET, RESIDENT)
    assert g.admitted and g.resident and g.need_gb == 0.0


@pytest.mark.parametrize("budget,pressure,word", [
    (dict(BUDGET, all_stop=True), RESIDENT, "all_stop"),
    (dict(BUDGET, queue=[{"x": 1}]), RESIDENT, "queue"),
    (BUDGET, dict(RESIDENT, level="critical"), "level"),
])
def test_gate_resident_still_obeys_all_checks(budget, pressure, word):
    g = gate_decision("m:1", 1.0, budget, pressure)
    assert not g.admitted and word in g.reason


def test_gate_governor_estimate_overrides_catalog():
    g = gate_decision("m:1", 10.0, BUDGET, PRESSURE, governor_est_gb=57.1)
    assert not g.admitted and g.need_gb == 57.1 and "57.1" in g.reason


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
    assert r["decode_tok_s"] is None and r["decode_eval_count"] == 50  # too few tokens for a decode rate
    assert r["prefill_tok_s"] == pytest.approx(200)


def test_stream_error_chunk():
    res = consume_stream(iter([{"error": "boom"}]))
    assert res.error == "boom"


# --- latency probe ------------------------------------------------------
LONG = [{"message": {"content": "word "}} for _ in range(3)] + [
    {"done": True, "done_reason": "stop", "prompt_eval_count": 40, "eval_count": 300,
     "prompt_eval_duration": 100_000_000, "eval_duration": 3_000_000_000, "load_duration": 10_000_000,
     "message": {"content": ""}}]


def test_think_param():
    assert latency_probe.think_param("x", ["tools"]) is None
    assert latency_probe.think_param("gpt-oss:20b", ["thinking"]) == "low"
    assert latency_probe.think_param("ornith:1", ["thinking"]) is False


def test_decode_rate_requires_min_eval_count():
    short = consume_stream(iter(STREAM))  # 50 tokens: below MIN_DECODE_TOKENS
    r = latency_probe.rates(short)
    assert r["decode_tok_s"] is None and r["decode_eval_count"] == 50
    long = consume_stream(iter(LONG))
    r = latency_probe.rates(long)
    assert r["decode_tok_s"] == pytest.approx(100) and r["decode_eval_count"] == 300
    assert r["prefill_tok_s"] is None  # 40 prompt tokens: below MIN_PREFILL_TOKENS


def test_decode_prompt_asks_for_long_answer_and_sets_no_limits():
    assert "300 words" in latency_probe.build_prompt("decode")


def test_errored_latency_call_has_no_rates_and_status_error(tmp_path):
    res = consume_stream(iter([{"error": "runner stopped"}]))
    rec = latency_probe.measurement_record(kind="latency", tag="m", endpoint="e", think=False, warm=True,
                                           size="small", res=res, gov={}, gate=None)
    assert rec["status"] == "error" and rec["metrics"]["decode_tok_s"] is None
    assert "runner stopped" in rec["failure_reason"]


@pytest.mark.parametrize("chunks,reason", [
    ([{"message": {"content": "x"}}], "no done chunk"),
    ([{"message": {"content": "x"}}, {"done": True, "done_reason": "length"}], "length"),
])
def test_unfinished_or_truncated_stream_is_failure(chunks, reason):
    res = consume_stream(iter(chunks))
    assert res.failed and reason in res.failure_reason


def test_latency_probe_records_shape_and_sequence(tmp_path):
    op = make_opener({"/budget": BUDGET, "/pressure": PRESSURE, "/api/chat": LONG})
    store = tmp_path / "m.jsonl"
    ctx = latency_probe.RunContext(run_id="r1", git_sha="abc", bfcl_revision="rev", digest="dg")
    show = {"details": {"quantization_level": "Q5_K_M"}}
    recs = latency_probe.run_latency_probe(GovernorClient("http://gov", op), "hf.co/x/y:UD-Q4_K_XL", show=show,
                                           capabilities=["thinking", "tools"], ctx=ctx, store=str(store))
    assert [r["input_size"] for r in recs] == ["small", "small", "medium", "large", "decode"]
    assert [r["repeat"] for r in recs][:2] == [1, 2]
    r = [json.loads(x) for x in store.read_text().splitlines()][0]
    assert r["schema"] == "model_measurement/2" and r["kind"] == "latency" and r["status"] == "ok"
    assert (r["run_id"], r["git_sha"], r["bfcl_revision"], r["digest"]) == ("r1", "abc", "rev", "dg")
    assert r["quant_show"] == "Q5_K_M" and r["quant_tag"] == "UD-Q4_K_XL"
    assert r["route"].endswith("/llm/ollama/api/chat model=hf.co/x/y:UD-Q4_K_XL")
    assert r["think_param"] is False and r["warm"] is True  # load_s overrides pressure
    assert {"ttft_content_s", "decode_tok_s", "decode_eval_count", "thinking_tokens_est"} <= set(r["metrics"])
    bodies = [json.loads(d) for u, d in op.calls if u.endswith("/api/chat")]
    assert all("num_ctx" not in b and "options" not in b and "num_predict" not in b for b in bodies)
    assert bodies[0]["think"] is False and bodies[0]["stream"] is True


def test_quant_helpers():
    assert latency_probe.quant_from_tag("ornith-1.5-9b:latest") is None
    assert latency_probe.quant_from_tag("qwen3.8:27b-mtp-q4_K_M") == "q4_K_M"
    assert latency_probe.quant_from_show({"details": {"quantization_level": "unknown"}}) is None
    assert latency_probe.quant_from_show({}) is None


def test_warm_load_s_overrides_pressure():
    assert latency_probe.is_warm(False, 0.01) is True
    assert latency_probe.is_warm(True, 4.0) is False
    assert latency_probe.is_warm(True, None) is True


def test_thinking_leak_flag():
    res = consume_stream(iter([{"message": {"thinking": "hm"}}, {"done": True, "done_reason": "stop"}]))
    assert latency_probe.thinking_leak(False, res) and not latency_probe.thinking_leak("low", res)
    res2 = consume_stream(iter([{"message": {"content": "<think>x</think>y"}}, {"done": True}]))
    assert latency_probe.thinking_leak(False, res2)


def _rec(size, ttft_c, load_s, warm_p, out=4, status_chunks=None):
    chunks = status_chunks or [{"message": {"content": "y"}}, {"done": True, "done_reason": "stop",
                               "prompt_eval_count": 500, "eval_count": out, "load_duration": int(load_s * 1e9),
                               "eval_duration": 1_000_000_000, "prompt_eval_duration": 1_000_000_000}]
    res = consume_stream(iter(chunks))
    res.ttft_content_s = ttft_c
    return latency_probe.measurement_record(kind="latency", tag="m", endpoint="e", think=False, warm=warm_p,
                                            size=size, res=res, gov={}, gate=None)


def test_summarize_is_per_size_warm_only_and_cold_separate():
    recs = [_rec("small", 9.0, 5.0, False), _rec("small", 0.5, 0.0, True), _rec("medium", 1.5, 0.0, True),
            _rec("large", 6.0, 0.0, True), _rec("decode", 0.4, 0.0, True, out=300),
            _rec("large", 99.0, 0.0, True, status_chunks=[{"error": "x"}])]
    s = latency_probe.summarize(recs)
    assert s["ttft_content_p50_s_by_size"] == {"small": 0.5, "medium": 1.5, "large": 6.0}
    assert s["cold_load_s"] == 5.0 and s["n_error"] == 1 and s["n_ok"] == 5
    assert s["decode_tok_s_p50"] == pytest.approx(300) and s["decode_eval_counts"] == [300]
    assert s["think_param"] is False and s["thinking_leaks"] == 0


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


def test_score_optional_and_case_insensitive_string():
    s = tool_probe.score_item(SIMPLE, call("calc_triangle_area", {"base": 10, "height": 5, "unit": "Units"}))
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


def test_cached_bfcl_loads_and_skips_data_error():
    bank = tool_probe.load_items()
    assert len(bank["simple"]) == 399 and len(bank["irrelevance"]) >= 100
    assert [x["id"] for x in bank["skipped"]] == ["simple_363"]


def test_source_pins_revision_and_hashes():
    import hashlib
    src = tool_probe.load_source()
    assert len(src["revision"]) == 40 and src["license"] == "apache-2.0"
    for name, digest in src["files"].items():
        assert hashlib.sha256((tool_probe.DEFAULT_DATA_DIR / name).read_bytes()).hexdigest() == digest


# --- BFCL oracle: the exact ground truth must pass (dict/list checker semantics) -------------
def _first(al):
    return next(c for c in al if c != "")


def _resolve_cand(c):
    if isinstance(c, dict):
        return {k: _resolve_allowed(v) for k, v in c.items() if any(x != "" for x in v)}
    if isinstance(c, list):
        return [_resolve_cand(x) for x in c]
    return c


def _resolve_allowed(al):
    return _resolve_cand(_first(al))


# simple_307 allows venue=true although its schema says string: the data conflicts with itself.
SCHEMA_CONFLICTS = {"simple_307"}


def test_oracle_every_ground_truth_passes():
    bank = tool_probe.load_items()
    bad = []
    for it in bank["simple"]:
        fn, args = next(iter(it["ground_truth"][0].items()))
        ideal = {p: _resolve_allowed(al) for p, al in args.items() if any(x != "" for x in al)}
        s = tool_probe.score_item(it, [{"function": {"name": tool_probe.sanitize_name(fn), "arguments": ideal}}])
        if not s["values_ok"] or (not s["passed"] and it["id"] not in SCHEMA_CONFLICTS):
            bad.append((it["id"], s["errors"][:2]))
    assert not bad, bad


@pytest.mark.parametrize("iid", ["simple_89", "simple_94", "simple_96", "simple_260", "simple_337"])
def test_oracle_dict_and_list_items_have_dict_truth(iid):
    item = next(i for i in tool_probe.load_items()["simple"] if i["id"] == iid)
    assert tool_probe.has_dict_truth(item)


def test_dict_truth_rejects_wrong_nested_value_and_extra_key():
    item = next(i for i in tool_probe.load_items()["simple"] if i["id"] == "simple_94")
    good = {"user_id": 43523, "update_info": {"name": "John Doe", "email": "johndoe@email.com"}}
    name = tool_probe.sanitize_name(next(iter(item["ground_truth"][0])))
    assert tool_probe.score_item(item, call(name, good))["passed"]
    bad = dict(good, update_info={"name": "Jane", "email": "johndoe@email.com"})
    assert not tool_probe.score_item(item, call(name, bad))["values_ok"]
    extra = dict(good, update_info={"name": "John Doe", "email": "johndoe@email.com", "zz": 1})
    assert not tool_probe.score_item(item, call(name, extra))["values_ok"]


def test_integer_rejects_float():
    assert not tool_probe.score_item(SIMPLE, call("calc_triangle_area", {"base": 10.0, "height": 5}))["passed"]
    # a float-typed parameter accepts an integer value
    sch = {"properties": {"x": {"type": "number"}}}
    assert tool_probe.values_match({"x": 5}, {"x": [5.0]}, sch) == []


def test_wrong_call_count_fails():
    two = call("calc_triangle_area", {"base": 10, "height": 5}) * 2
    s = tool_probe.score_item(SIMPLE, two)
    assert s["count_ok"] is False and s["passed"] is False


def test_ground_truth_name_mismatch_is_data_error_not_keyerror():
    bad = dict(SIMPLE, ground_truth=[{"find_closest": {"base": [1]}}])
    s = tool_probe.score_item(bad, call("calc_triangle_area", {"base": 1}))
    assert s["status"] == "data_error" and s["passed"] is None


def test_load_items_skips_mismatch_and_continues(tmp_path):
    bad = {k: v for k, v in SIMPLE.items() if k not in ("ground_truth", "category")}
    (tmp_path / "BFCL_v3_simple.json").write_text(json.dumps(bad) + "\n")
    (tmp_path / "possible_answer_BFCL_v3_simple.json").write_text(
        json.dumps({"id": "simple_0", "ground_truth": [{"nope": {"base": [1]}}]}) + "\n")
    (tmp_path / "BFCL_v3_irrelevance.json").write_text(json.dumps({k: v for k, v in IRR.items() if k not in ("ground_truth", "category")}) + "\n")
    bank = tool_probe.load_items(tmp_path)
    assert bank["simple"] == [] and bank["skipped"][0]["id"] == "simple_0"


def test_pick_pilot_items_includes_dict_truth():
    bank = tool_probe.load_items()
    items = tool_probe.pick_pilot_items(bank, 3, 2)
    assert len(items) == 5 and any(tool_probe.has_dict_truth(i) for i in items if i["category"] == "simple")
    assert items == tool_probe.pick_pilot_items(bank, 3, 2)  # deterministic


def test_error_chunk_on_irrelevance_is_not_an_abstention(tmp_path):
    (tmp_path / "BFCL_v3_simple.json").write_text(json.dumps({k: v for k, v in SIMPLE.items() if k not in ("ground_truth", "category")}) + "\n")
    (tmp_path / "possible_answer_BFCL_v3_simple.json").write_text(json.dumps({"id": "simple_0", "ground_truth": SIMPLE["ground_truth"]}) + "\n")
    (tmp_path / "BFCL_v3_irrelevance.json").write_text(json.dumps({k: v for k, v in IRR.items() if k not in ("ground_truth", "category")}) + "\n")
    op = make_opener({"/budget": BUDGET, "/pressure": PRESSURE, "/api/chat": [{"error": "runner stopped"}]})
    out = tool_probe.run_tool_probe(GovernorClient("http://gov", op), "m:1", capabilities=["tools"], n_simple=1,
                                    n_irrelevance=1, data_dir=tmp_path, store=str(tmp_path / "m.jsonl"))
    irr = [s for s in out["scores"] if s["category"] == "irrelevance"][0]
    assert irr["status"] == "error" and irr["passed"] is None and irr["abstained"] is None
    sm = out["summary"]
    assert sm["n_error"] == 2 and sm["n_scored"] == 0 and sm["abstention_rate"] is None and sm["pass_rate"] is None


def test_run_tool_probe_records(tmp_path):
    (tmp_path / "BFCL_v3_simple.json").write_text(json.dumps({k: v for k, v in SIMPLE.items() if k not in ("ground_truth", "category")}) + "\n")
    (tmp_path / "possible_answer_BFCL_v3_simple.json").write_text(json.dumps({"id": "simple_0", "ground_truth": SIMPLE["ground_truth"]}) + "\n")
    (tmp_path / "BFCL_v3_irrelevance.json").write_text(json.dumps({k: v for k, v in IRR.items() if k not in ("ground_truth", "category")}) + "\n")
    stream = [{"message": {"tool_calls": call("calc_triangle_area", {"base": 10, "height": 5})}},
              {"done": True, "done_reason": "stop", "prompt_eval_count": 9, "eval_count": 7,
               "eval_duration": 1_000_000_000, "prompt_eval_duration": 100_000_000, "message": {"content": ""}}]
    op = make_opener({"/budget": BUDGET, "/pressure": PRESSURE, "/api/chat": stream})
    store = tmp_path / "m.jsonl"
    out = tool_probe.run_tool_probe(GovernorClient("http://gov", op), "m:1", show={"details": {"quantization_level": "q4"}},
                                    capabilities=["tools"], n_simple=1, n_irrelevance=1, data_dir=tmp_path,
                                    store=str(store))
    assert out["summary"]["n"] == 2 and out["scores"][0]["passed"] and out["scores"][1]["passed"] is False
    rec = json.loads(store.read_text().splitlines()[0])
    assert rec["kind"] == "tool" and rec["item"]["license"] == "apache-2.0" and rec["score"]["passed"]
    assert rec["validator"] in ("jsonschema", "fallback") and rec["quant_show"] == "q4"
    body = json.loads([d for u, d in op.calls if u.endswith("/api/chat")][0])
    assert "tools" in body and "options" not in body


def test_tool_probe_skips_without_capability(tmp_path):
    out = tool_probe.run_tool_probe(GovernorClient("http://gov", make_opener({})), "m",
                                    capabilities=["completion"], store=str(tmp_path / "x"))
    assert out["skipped"]


def test_fallback_validator_is_marked(monkeypatch):
    monkeypatch.setattr(tool_probe, "validator_name", lambda: "fallback")
    s = tool_probe.score_item(SIMPLE, call("calc_triangle_area", {"base": 10, "height": 5}))
    assert s["validator"] == "fallback" and s["passed"]
    assert tool_probe.validate_args({"base": True, "height": 2}, tool_probe.to_json_schema(SIMPLE["function"][0]["parameters"]))


def test_bool_is_not_an_int_and_list_length_matters():
    assert tool_probe.values_match({"f": 1}, {"f": [True]}) != []
    assert tool_probe.values_match({"f": True}, {"f": [True]}) == []
    assert tool_probe.values_match({"l": ["a", "b", "c"]}, {"l": [["a", "b"]]}) != []
    assert tool_probe.values_match({"l": ["a"]}, {"l": [["a", "b"]]}) != []
    assert tool_probe.values_match({"l": ["a", "b"]}, {"l": [["a", "b"]]}) == []


# --- governor 503, rotation ------------------------------------------------
def test_governor_503_becomes_refusal_with_estimate():
    import urllib.error
    body = json.dumps({"error": "too big", "code": "model_too_large", "retry_after_s": 120,
                       "reason": "won't fit on this box: est 57.1GB > LLM ceiling 42.9GB (cliff"}).encode()

    def opener(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 503, "Service Unavailable", {}, io.BytesIO(body))

    c = GovernorClient("http://gov", opener)
    with pytest.raises(governor.GovernorRefusal) as e:
        list(c.chat_stream({"model": "m"}))
    assert e.value.est_gb == 57.1 and e.value.ceiling_gb == 42.9 and e.value.code == "model_too_large"
    with pytest.raises(governor.GovernorRefusal):
        c.show("m")


def test_governor_other_http_errors_propagate():
    import urllib.error

    def opener(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 500, "boom", {}, io.BytesIO(b"{}"))

    with pytest.raises(urllib.error.HTTPError):
        GovernorClient("http://gov", opener).budget()


def test_store_rotates_by_size(tmp_path):
    p = tmp_path / "s.jsonl"
    for i in range(12):
        governor.append_measurement(p, {"i": i, "pad": "x" * 100}, max_bytes=500, keep=3)
    names = sorted(x.name for x in tmp_path.iterdir())
    assert names == ["s.jsonl", "s.jsonl.1", "s.jsonl.2", "s.jsonl.3"] or names == ["s.jsonl", "s.jsonl.1", "s.jsonl.2"]
    assert not (tmp_path / "s.jsonl.4").exists()


def test_digest_from_tags():
    op = make_opener({"/api/tags": {"models": [{"name": "m:1", "digest": "abc"}]}})
    assert GovernorClient("http://gov", op).digest("m:1") == "abc"
    assert GovernorClient("http://gov", op).digest("zz") is None


# --- netguard (run in a clean subprocess; it patches the process-wide socket) ---------------
RUN_DIR = __import__("pathlib").Path(__file__).resolve().parents[1] / "scripts" / "runs" / "2026-11-26_probe-pilot"


def _guarded(code):
    import subprocess, sys
    return subprocess.run([sys.executable, "-I", "-c", f"import sys; sys.path.insert(0, {str(RUN_DIR)!r}); import netguard\n{code}"],
                          capture_output=True, text=True, timeout=60)


@pytest.mark.parametrize("addr", ["('127.0.0.1', 37400)", "('192.168.0.159', 11434)", "('140.82.112.3', 443)",
                                  "('192.168.0.159', 8801)"])
def test_netguard_blocks_everything_but_governor(addr):
    r = _guarded(f"import socket\ns = socket.socket()\n"
                 f"try:\n    s.connect({addr}); print('CONNECTED')\nexcept RuntimeError as e: print('B1')\n"
                 f"try:\n    s.connect_ex({addr}); print('CONNECTED')\nexcept RuntimeError: print('B2')\n")
    assert r.stdout.split() == ["B1", "B2"], r.stdout + r.stderr


def test_netguard_blocks_dns_and_allows_governor_pair():
    r = _guarded("import socket, netguard\n"
                 "try: socket.getaddrinfo('raw.githubusercontent.com', 443)\nexcept RuntimeError: print('DNS-BLOCKED')\n"
                 "import socket as s; sk = s.socket(); sk.settimeout(0.2)\n"
                 "try: sk.connect(('192.168.0.159', 8799)); print('GOV-OK')\n"
                 "except RuntimeError: print('GOV-BLOCKED')\n"
                 "except OSError: print('GOV-OK')\n")
    assert "DNS-BLOCKED" in r.stdout and "GOV-OK" in r.stdout, r.stdout + r.stderr


def test_netguard_precedes_litellm_and_strips_keys():
    import os, subprocess, sys
    env = dict(os.environ, OPENAI_API_KEY="k", SOME_SERVICE_SECRET="s", AWS_ACCESS_KEY_ID="a")
    code = ("import sys, os; sys.path.insert(0, %r); import netguard\n"
            "print(os.environ.get('LITELLM_LOCAL_MODEL_COST_MAP'), os.environ.get('OPENAI_API_KEY'),"
            " os.environ.get('SOME_SERVICE_SECRET'), os.environ.get('AWS_ACCESS_KEY_ID'))\n"
            "import litellm\nprint('BLOCKED', netguard.blocked)\n") % str(RUN_DIR)
    r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=120, env=env)
    assert r.stdout.splitlines()[0] == "True None None None", r.stdout + r.stderr
    assert "BLOCKED []" in r.stdout, r.stdout + r.stderr


# ---- llamacpp lane: SSE -> Ollama-shaped chunks ----

def _sse(*events):
    return [b"data: " + json.dumps(e).encode() + b"\n\n" for e in events] + [b"data: [DONE]\n\n"]


def _delta(**d):
    return {"choices": [{"index": 0, "delta": d, "finish_reason": None}]}


def _final(reason, **extra):
    return {"choices": [{"index": 0, "delta": {}, "finish_reason": reason}], **extra}


TIMINGS = {"prompt_n": 58, "prompt_ms": 1000.0, "predicted_n": 75, "predicted_ms": 5000.0}


def test_sse_content_reasoning_and_timings():
    from promptchain.profiler.governor import consume_stream, sse_to_chunks
    lines = _sse(_delta(role="assistant", content=None), _delta(reasoning_content="think"),
                 _delta(reasoning_content="ing"), _delta(content="Hel"), _delta(content="lo"),
                 _final("stop", timings=TIMINGS))
    res = consume_stream(sse_to_chunks(lines))
    assert res.content == "Hello" and res.thinking_chunks == 2 and res.thinking_chars == 8
    assert res.done and not res.failed
    assert (res.prompt_tokens, res.output_tokens) == (58, 75)
    assert res.prefill_s == 1.0 and res.decode_s == 5.0 and res.load_s is None


def test_sse_fragmented_tool_call_arguments():
    from promptchain.profiler.governor import consume_stream, sse_to_chunks
    frag = lambda **fn: _delta(tool_calls=[{"index": 0, **({"id": "x", "type": "function"} if "name" in fn else {}),
                                            "function": fn}])
    lines = _sse(frag(name="get_weather", arguments="{"), frag(arguments='"city":"'), frag(arguments="Paris"),
                 frag(arguments='"}'), _final("tool_calls", timings=TIMINGS))
    res = consume_stream(sse_to_chunks(lines))
    assert res.tool_calls == [{"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}]
    assert res.done and not res.failed


def test_sse_length_and_missing_finish_are_failures():
    from promptchain.profiler.governor import consume_stream, sse_to_chunks
    assert consume_stream(sse_to_chunks(_sse(_delta(content="a"), _final("length", timings=TIMINGS)))).failed
    assert consume_stream(sse_to_chunks([b"data: " + json.dumps(_delta(content="a")).encode() + b"\n\n"])).failed


def test_sse_usage_fallback_and_done_stops_stream():
    from promptchain.profiler.governor import sse_to_chunks
    lines = _sse(_delta(content="a"), _final("stop", usage={"prompt_tokens": 3, "completion_tokens": 4}))
    last = list(sse_to_chunks(lines + [b"data: " + b"not json\n\n"]))[-1]
    assert last["prompt_eval_count"] == 3 and last["eval_count"] == 4


def test_llamacpp_client_translates_request_and_raises_refusal():
    import urllib.error
    from promptchain.profiler.governor import GovernorRefusal, LlamacppGovernorClient
    seen = {}

    def opener(req, timeout=None):
        seen["url"], seen["body"] = req.full_url, json.loads(req.data)
        return FakeResp(b"".join(_sse(_delta(content="ok"), _final("stop", timings=TIMINGS))))

    c = LlamacppGovernorClient(opener=opener)
    chunks = list(c.chat_stream({"model": "m", "messages": [], "think": False, "options": {"num_ctx": 1}}))
    assert seen["url"].endswith("/llm/llamacpp/v1/chat/completions")
    assert seen["body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "options" not in seen["body"] and "think" not in seen["body"] and chunks[-1]["done"]
    assert c.digest("m") == "llamacpp:m"

    def refuse(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 503, "x", {}, io.BytesIO(b'{"code":"model_too_large"}'))

    with pytest.raises(GovernorRefusal):
        list(LlamacppGovernorClient(opener=refuse).chat_stream({"model": "m", "messages": []}))
