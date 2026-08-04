"""Token usage must be recorded for the shape PromptChain actually returns.

Regression: `track_llm_call` extracted usage via `hasattr(result, "usage")`, but it
decorates `PromptChain.run_model_async`, which returns the MESSAGE dict
(`response["choices"][0]["message"]`). `usage` lives on the RESPONSE, one level up,
so the check was ALWAYS False and no consumer ever got token metrics — measured:
3,163 MLflow runs logged, zero token metrics. It failed silently: init_mlflow()
reports success and runs appear in the UI, only the numbers are missing.
"""

import pytest

from promptchain.observability.decorators import _log_token_usage


class _Usage:
    def __init__(self, p, c, t=None):
        self.prompt_tokens = p
        self.completion_tokens = c
        if t is not None:
            self.total_tokens = t


class _Response:
    """What a raw LiteLLM/OpenAI response looks like."""
    def __init__(self, p, c, t=None):
        self.usage = _Usage(p, c, t)


class _Chain:
    """What run_model_async's bound instance carries after a call."""
    def __init__(self, p, c):
        self.last_prompt_tokens = p
        self.last_completion_tokens = c


@pytest.fixture
def logged(monkeypatch):
    """Capture queued metrics instead of shipping them to MLflow."""
    out = {}
    monkeypatch.setattr(
        "promptchain.observability.decorators.queue_log_metric",
        lambda k, v, step=None: out.__setitem__(k, v),
    )
    return out


def test_response_object_with_usage(logged):
    assert _log_token_usage(_Response(11, 22, 33), ()) is True
    assert logged == {"prompt_tokens": 11.0, "completion_tokens": 22.0, "total_tokens": 33.0}


def test_mapping_with_usage_key(logged):
    payload = {"usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}}
    assert _log_token_usage(payload, ()) is True
    assert logged == {"prompt_tokens": 5.0, "completion_tokens": 7.0, "total_tokens": 12.0}


def test_message_dict_falls_back_to_instance(logged):
    """THE REAL PATH: run_model_async returns a message dict with no usage anywhere."""
    message = {"role": "assistant", "content": "hi"}
    chain = _Chain(101, 202)
    assert _log_token_usage(message, (chain,)) is True
    assert logged["prompt_tokens"] == 101.0
    assert logged["completion_tokens"] == 202.0
    assert logged["total_tokens"] == 303.0        # derived when absent


def test_total_is_derived_when_missing(logged):
    assert _log_token_usage(_Response(3, 4), ()) is True   # no total_tokens on usage
    assert logged["total_tokens"] == 7.0


def test_no_usage_anywhere_is_silent_not_fatal(logged):
    assert _log_token_usage({"role": "assistant", "content": "x"}, (object(),)) is False
    assert logged == {}


def test_none_result_does_not_raise(logged):
    assert _log_token_usage(None, ()) is False
    assert logged == {}


def test_non_numeric_usage_is_ignored(logged):
    payload = {"usage": {"prompt_tokens": "n/a", "completion_tokens": None}}
    assert _log_token_usage(payload, ()) is False
    assert logged == {}


def test_explicit_usage_wins_over_instance(logged):
    """A real response's own numbers must not be shadowed by stale instance state."""
    chain = _Chain(999, 999)
    assert _log_token_usage(_Response(1, 2, 3), (chain,)) is True
    assert logged == {"prompt_tokens": 1.0, "completion_tokens": 2.0, "total_tokens": 3.0}


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
