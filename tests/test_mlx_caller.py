"""Tests for MLXCaller (issue #35) — prompt building (template vs raw), param merge, lazy load.
Offline: a fake tokenizer is injected so build_prompt is testable without mlx-lm or a model."""
from promptchain.utils.mlx_caller import MLXCaller, MLXResponse


class _FakeTok:
    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        assert add_generation_prompt is True and tokenize is False
        return "TEMPLATED:" + "|".join(m["content"] for m in messages)


MSGS = [{"role": "system", "content": "S"}, {"role": "user", "content": "hi"}]


def test_build_prompt_uses_chat_template():
    c = MLXCaller("m", use_chat_template=True)
    assert c.build_prompt(MSGS, tokenizer=_FakeTok()) == "TEMPLATED:S|hi"


def test_build_prompt_raw_when_disabled():
    c = MLXCaller("m", use_chat_template=False)
    assert c.build_prompt(MSGS, tokenizer=_FakeTok()) == "S\nhi"


def test_build_prompt_raw_when_no_template_support():
    c = MLXCaller("m", use_chat_template=True)
    assert c.build_prompt(MSGS, tokenizer=object()) == "S\nhi"  # tokenizer lacks apply_chat_template


def test_resolve_params_call_overrides_instance():
    c = MLXCaller("m", default_params={"max_tokens": 256, "temp": 0.2})
    assert c.resolve_params({"temp": 0.0}) == {"max_tokens": 256, "temp": 0.0}


def test_lazy_no_load_on_construct():
    c = MLXCaller("/nonexistent")
    assert c._model is None and c._tok is None


def test_response_defaults():
    r = MLXResponse("hello")
    assert r.text == "hello" and r.usage == {}


def test_response_none_text_safe():
    assert MLXResponse(None).text == ""
