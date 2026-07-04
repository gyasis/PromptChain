"""Tests for LlamaCppCaller (issue #33) — grammar-toggle + param resolution + response parsing.
All offline: no model load, no llama-cpp-python required (lazy import)."""
from promptchain.utils.llama_cpp_caller import LlamaCppCaller, LlamaCppResponse, _UNSET


def test_grammar_unset_uses_instance_default():
    c = LlamaCppCaller("m.gguf", grammar="root ::= object")
    _, g = c.resolve(grammar=_UNSET)
    assert g == "root ::= object"


def test_grammar_none_overrides_to_raw():
    # explicit None must beat the instance default → RAW/malformable output
    c = LlamaCppCaller("m.gguf", grammar="root ::= object")
    _, g = c.resolve(grammar=None)
    assert g is None


def test_grammar_string_override():
    c = LlamaCppCaller("m.gguf", grammar=None)
    _, g = c.resolve(grammar="root ::= json")
    assert g == "root ::= json"


def test_default_is_raw():
    # no grammar given at all → raw (None) by default
    c = LlamaCppCaller("m.gguf")
    _, g = c.resolve()
    assert g is None


def test_params_merge_call_overrides_instance():
    c = LlamaCppCaller("m.gguf", default_params={"temperature": 0.2, "top_p": 0.9})
    merged, _ = c.resolve(params={"temperature": 0.0})
    assert merged == {"temperature": 0.0, "top_p": 0.9}


def test_lazy_no_model_load_on_construct():
    c = LlamaCppCaller("/nonexistent.gguf")
    assert c._llm is None  # constructing must not load the model


def test_response_accessors():
    r = LlamaCppResponse({"choices": [{"message": {"content": "hi"}}],
                          "usage": {"prompt_tokens": 4, "completion_tokens": 1}})
    assert r.text == "hi" and r.usage["prompt_tokens"] == 4


def test_response_empty_safe():
    r = LlamaCppResponse({})
    assert r.text == "" and r.usage == {}
