"""Repo-root pytest config: tests run against a MOCK model by default and never spend money.

* Autouse fixture replaces litellm.completion / litellm.acompletion (and every module-level
  alias made by `from litellm import ...`) with a deterministic mock, unless PC_LIVE_TESTS=1.
* `live` marker: tests that need a real model are skipped unless PC_LIVE_TESTS=1.
* Socket guard: any attempt to resolve/connect to a real provider host while the mock is
  active is recorded and fails the test loudly (even if the code swallows the exception).

Inspect calls:   def test_x(mock_llm): ...; mock_llm.calls, mock_llm.sync_calls, mock_llm.async_calls
Custom reply:    mock_llm.reply = "text"   or   mock_llm.reply = lambda kwargs: "text"
"""
import os
import socket
import sys
import time
import uuid

import pytest

LIVE = os.environ.get("PC_LIVE_TESTS") == "1"
MOCK_REPLY = "MOCK_RESPONSE"
PROVIDER_HOSTS = ("api.openai.com", "api.anthropic.com", "generativelanguage.googleapis.com")


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "live: real model call; set PC_LIVE_TESTS=1 to run"
    )


def pytest_collection_modifyitems(config, items):
    if LIVE:
        return
    skip = pytest.mark.skip(reason="real model call; set PC_LIVE_TESTS=1 to run")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)


def _build_response(kwargs, content):
    from litellm import ModelResponse

    msgs = kwargs.get("messages") or []
    ptoks = max(1, sum(len(str(m.get("content", ""))) for m in msgs if isinstance(m, dict)) // 4)
    ctoks = max(1, len(content) // 4)
    return ModelResponse(
        id=f"mock-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=kwargs.get("model", "mock-model"),
        object="chat.completion",
        choices=[
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content, "tool_calls": None},
            }
        ],
        usage={
            "prompt_tokens": ptoks,
            "completion_tokens": ctoks,
            "total_tokens": ptoks + ctoks,
        },
    )


class MockLLM:
    def __init__(self):
        self.reply = MOCK_REPLY  # str, or callable(kwargs) -> str
        self.sync_calls = []
        self.async_calls = []

    @property
    def calls(self):
        return self.sync_calls + self.async_calls

    def _content(self, kwargs):
        return self.reply(kwargs) if callable(self.reply) else self.reply

    def completion(self, *args, **kwargs):
        self.sync_calls.append(kwargs)
        if kwargs.get("stream"):
            return iter([_build_chunk(kwargs, self._content(kwargs))])
        return _build_response(kwargs, self._content(kwargs))

    async def acompletion(self, *args, **kwargs):
        self.async_calls.append(kwargs)
        content = self._content(kwargs)
        if kwargs.get("stream"):
            chunk = _build_chunk(kwargs, content)

            async def agen():
                yield chunk

            return agen()
        return _build_response(kwargs, content)


def _build_chunk(kwargs, content):
    from litellm import ModelResponse

    return ModelResponse(
        id=f"mock-{uuid.uuid4().hex[:12]}",
        model=kwargs.get("model", "mock-model"),
        stream=True,
        choices=[{"index": 0, "finish_reason": "stop", "delta": {"role": "assistant", "content": content}}],
    )


def _is_provider(host):
    h = host.decode() if isinstance(host, bytes) else str(host)
    return any(p in h for p in PROVIDER_HOSTS)


@pytest.fixture(autouse=True)
def mock_llm(monkeypatch, request):
    """Mock litellm unless PC_LIVE_TESTS=1 (live tests only run in that mode anyway)."""
    if LIVE:
        yield None
        return

    import litellm

    mock = MockLLM()
    orig_c, orig_a = litellm.completion, litellm.acompletion
    monkeypatch.setattr(litellm, "completion", mock.completion)
    monkeypatch.setattr(litellm, "acompletion", mock.acompletion)
    # `from litellm import completion` copies the name into each module; rebind those too.
    for mod in list(sys.modules.values()):
        d = getattr(mod, "__dict__", None)
        if not d or mod is litellm:
            continue
        try:
            for name, val in list(d.items()):
                if val is orig_c:
                    monkeypatch.setattr(mod, name, mock.completion)
                elif val is orig_a:
                    monkeypatch.setattr(mod, name, mock.acompletion)
        except Exception:
            continue

    # Socket guard: no real provider traffic, ever, while mocked.
    violations = []
    real_gai, real_cc = socket.getaddrinfo, socket.create_connection

    def guard_gai(host, *a, **k):
        if _is_provider(host):
            violations.append(f"getaddrinfo({host!r})")
            raise RuntimeError(f"BLOCKED real provider access: {host}. Mark test @pytest.mark.live.")
        return real_gai(host, *a, **k)

    def guard_cc(address, *a, **k):
        if _is_provider(address[0]):
            violations.append(f"create_connection({address!r})")
            raise RuntimeError(f"BLOCKED real provider access: {address}. Mark test @pytest.mark.live.")
        return real_cc(address, *a, **k)

    monkeypatch.setattr(socket, "getaddrinfo", guard_gai)
    monkeypatch.setattr(socket, "create_connection", guard_cc)

    yield mock

    if violations:
        pytest.fail(f"Test attempted real provider network access: {violations}", pytrace=False)


@pytest.fixture(autouse=True)
def _isolated_models_config(tmp_path_factory, monkeypatch):
    """Point the models file at a path that does not exist, so a developer's own
    ~/.promptchain/models.yaml never changes a test's result."""
    from promptchain.utils import model_roles

    monkeypatch.setenv(
        model_roles.ENV_VAR, str(tmp_path_factory.getbasetemp() / "no-models.yaml")
    )
    model_roles._cache = None
    yield
    model_roles._cache = None
