"""Tests for RawCaller (issue #31) — the point is to PROVE strict passthrough: no injection,
no hidden keys, no output parsing. All offline (payload builder + response accessors)."""
from promptchain.utils.raw_caller import RawCaller, RawResponse


def test_openai_payload_is_exact_passthrough():
    rc = RawCaller("http://x/v1", model="m", api_style="openai")
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "hi"}]
    p = rc.build_payload(msgs, params={"temperature": 0.0, "seed": 7})
    # EXACTLY model + messages + the params we gave — nothing injected
    assert p == {"model": "m", "messages": msgs, "temperature": 0.0, "seed": 7}
    assert p["messages"] is msgs  # messages passed through verbatim, not rewritten


def test_openai_no_hidden_keys_when_no_params():
    rc = RawCaller("http://x/v1", model="m")
    p = rc.build_payload([{"role": "user", "content": "hi"}])
    assert set(p.keys()) == {"model", "messages"}  # no temperature/stop/tools invented


def test_ollama_payload_shape_and_passthrough():
    rc = RawCaller("http://h:11434", model="m", api_style="ollama", default_params={"temperature": 0.2})
    msgs = [{"role": "user", "content": "hi"}]
    p = rc.build_payload(msgs, params={"seed": 1})
    assert p == {"model": "m", "messages": msgs, "stream": False, "options": {"temperature": 0.2, "seed": 1}}


def test_endpoint_selection():
    assert RawCaller("http://x/v1", api_style="openai")._endpoint() == "http://x/v1/chat/completions"
    assert RawCaller("http://x", api_style="ollama")._endpoint() == "http://x/api/chat"


def test_bad_api_style_rejected():
    try:
        RawCaller("http://x", api_style="anthropic-magic")
        assert False, "should have raised"
    except ValueError:
        pass


def test_response_accessors_openai():
    r = RawResponse({"choices": [{"message": {"content": "hello"}}],
                     "usage": {"prompt_tokens": 8, "completion_tokens": 3}}, "openai")
    assert r.text == "hello" and r.usage["prompt_tokens"] == 8


def test_response_accessors_ollama():
    r = RawResponse({"message": {"content": "hi"}, "prompt_eval_count": 5, "eval_count": 2}, "ollama")
    assert r.text == "hi"
    assert r.usage == {"prompt_tokens": 5, "completion_tokens": 2}
