"""Models config file: precedence, endpoints, errors, show/init. Offline."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from promptchain.utils import model_roles as mr
from promptchain.utils.model_roles import (ModelConfigError,
                                           ModelNotConfiguredError,
                                           resolve_model, resolve_model_spec)

BASE = "http://<host>:1/v1"


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    def write(text):
        path = tmp_path / "models.yaml"
        path.write_text(text)
        monkeypatch.setenv(mr.ENV_VAR, str(path))
        mr._cache = None
        return path

    return write


def test_missing_file_is_empty_and_required_errors(tmp_path, monkeypatch):
    monkeypatch.setenv(mr.ENV_VAR, str(tmp_path / "none.yaml"))
    mr.reload()
    assert resolve_model("judge", required=False, param="judge_model") is None
    with pytest.raises(ModelNotConfiguredError) as e:
        resolve_model("supervisor", required=True, param="supervisor_model")
    msg = str(e.value)
    assert "supervisor_model" in msg and "roles.supervisor" in msg and "none.yaml" in msg


def test_precedence_explicit_role_default(cfg):
    cfg("default: d/m\nroles:\n  supervisor: r/m\n")
    assert resolve_model("supervisor", "x/m", required=True, param="p") == "x/m"
    assert resolve_model("supervisor", required=True, param="p") == "r/m"
    assert resolve_model("agent", required=True, param="p") == "d/m"
    assert resolve_model_spec("agent", required=True, param="p").source == "default"
    assert resolve_model_spec("supervisor", required=True, param="p").source == "role"


def test_optional_ignores_default(cfg):
    cfg("default: d/m\nroles:\n  summarizer: s/m\n")
    assert resolve_model("judge", required=False, param="p") is None
    assert resolve_model("summarizer", required=False, param="p") == "s/m"
    assert resolve_model("judge", "e/m", required=False, param="p") == "e/m"


@pytest.mark.parametrize(
    "text, needle",
    [
        ("roles: [a]\n", "roles"),
        ("roles:\n  nope: a/b\n", "unknown role"),
        ("bogus: 1\n", "unknown key"),
        ("default: [1]\n", "default"),
        ("a: [unclosed\n", "Cannot read"),
        ("endpoints:\n  e: {api_base: x, num_ctx: 9}\n", "unsupported"),
        ("endpoints:\n  e: {provider: openai}\n", "api_base"),
    ],
)
def test_malformed_file_raises_with_path(cfg, text, needle):
    path = cfg(text)
    with pytest.raises(ModelConfigError) as e:
        mr.reload()
    assert needle in str(e.value) and str(path) in str(e.value)


def test_endpoint_resolution_and_mapping_roles(cfg):
    cfg(
        f"""
endpoints:
  lane-ollama: {{api_base: "{BASE}"}}
  lane-llama: {{api_base: "{BASE}", provider: openai}}
default: {{model: dm, endpoint: lane-ollama}}
roles:
  judge: {{model: jm, endpoint: lane-llama}}
  agent: plain/model
"""
)
    j = resolve_model_spec("judge", required=False, param="p")
    assert (j.model, j.api_base, j.endpoint) == ("jm", BASE, "lane-llama")
    assert j.params == {"api_base": BASE, "custom_llm_provider": "openai"}
    assert j.as_model() == {"name": "jm", "params": j.params}
    assert resolve_model_spec("agent", required=True, param="p").as_model() == "plain/model"
    d = resolve_model_spec("router", required=True, param="p")
    assert d.params == {"api_base": BASE} and d.source == "default"
    assert resolve_model("judge", required=False, param="p") == "jm"
    # an explicit model carries no endpoint
    assert resolve_model_spec("judge", "x", required=False, param="p").params == {}


def test_unknown_endpoint_names_role_and_endpoint(cfg):
    cfg("endpoints:\n  a: {api_base: u}\nroles:\n  judge: {model: m, endpoint: nope}\n")
    with pytest.raises(ModelConfigError) as e:
        mr.reload()
    assert "roles.judge" in str(e.value) and "'nope'" in str(e.value) and "a" in str(e.value)


def test_show_and_init(cfg, tmp_path, monkeypatch, capsys):
    cfg(f"default: d/m\nendpoints:\n  e: {{api_base: \"{BASE}\"}}\nroles:\n  judge: {{model: jm, endpoint: e}}\n")
    text = mr.show_text()
    assert "judge" in text and BASE in text and "role" in text and "default" in text
    for name in mr.ROLE_NAMES:
        assert name in text
    target = tmp_path / "new" / "m.yaml"
    monkeypatch.setenv(mr.ENV_VAR, str(target))
    mr.reload()
    assert mr.main(["init"]) == 0
    body = target.read_text()
    assert all(f"  {n}:" in body for n in mr.ROLE_NAMES)
    assert "<host>" in body and "endpoints:" in body
    # no model name: every non-comment line carries no value
    for line in body.splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            assert line.rstrip().endswith(":"), line
    assert mr.main(["init"]) == 1  # refuses to overwrite
    mr.reload()
    assert resolve_model("judge", required=False, param="p") is None


# ---- wiring ---------------------------------------------------------------

def test_wiring_required_sites_read_the_file(cfg):
    cfg("default: d/m\nroles:\n  history_summarizer: h/m\n")
    from promptchain.utils.history_summarizer import HistorySummarizer
    from promptchain.patterns.agent_comms import agent

    assert HistorySummarizer().model == "h/m"
    assert agent("A", "p").asp.model_name == "d/m"
    assert HistorySummarizer(model="x/m").model == "x/m"


def test_missing_required_error_names_param_key_path(cfg):
    path = cfg("")
    from promptchain.utils.history_summarizer import HistorySummarizer

    with pytest.raises(ValueError) as e:
        HistorySummarizer()
    assert "`model`" in str(e.value) and "roles.history_summarizer" in str(e.value)
    assert str(path) in str(e.value)


def test_api_base_reaches_litellm_call_summarizer(cfg):
    cfg(f"endpoints:\n  e: {{api_base: \"{BASE}\", provider: openai}}\nroles:\n  history_summarizer: {{model: sm, endpoint: e}}\n")
    from promptchain.utils.history_summarizer import HistorySummarizer

    s = HistorySummarizer()
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content="sum"))]
    with patch("litellm.acompletion", new=AsyncMock(return_value=resp)) as call:
        asyncio.run(
            s.summarize_history(
                [{"role": "user", "content": f"turn {i}"} for i in range(6)], preserve_last_n=1
            )
        )
    kw = call.call_args.kwargs
    assert kw["model"] == "sm" and kw["api_base"] == BASE and kw["custom_llm_provider"] == "openai"


def test_api_base_reaches_litellm_call_judge(cfg):
    cfg(f"endpoints:\n  e: {{api_base: \"{BASE}\"}}\nroles:\n  judge: {{model: jm, endpoint: e}}\n")
    from promptchain.integrations.lightrag.branching import (
        BranchingConfig, Hypothesis, LightRAGBranchingThoughts)

    with patch("promptchain.integrations.lightrag.branching.LIGHTRAG_AVAILABLE", True):
        b = LightRAGBranchingThoughts(MagicMock(), BranchingConfig())
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content="[]"))]
    b._litellm = MagicMock(acompletion=AsyncMock(return_value=resp))
    hyp = [Hypothesis(hypothesis_id="h1", approach="a", reasoning="r", confidence=0.5, mode="local")]
    asyncio.run(b._judge_hypotheses("q", hyp))
    kw = b._litellm.acompletion.call_args.kwargs
    assert kw["model"] == "jm" and kw["api_base"] == BASE


def test_optional_judge_off_with_only_default(cfg):
    cfg("default: d/m\n")
    from promptchain.integrations.lightrag.branching import BranchingConfig

    assert resolve_model_spec("judge", BranchingConfig().judge_model, required=False, param="p") is None
