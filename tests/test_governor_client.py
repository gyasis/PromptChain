"""Tests for GovernorClient + governed callers (issue #37, dev branch).
Offline: GovernorClient with a mocked _req; callers with a FAKE duck-typed governor (no network,
no model, no heavy deps)."""
from promptchain.utils.governor_client import GovernorClient, _Lease
from promptchain.utils.llama_cpp_caller import LlamaCppCaller
from promptchain.utils.mlx_caller import MLXCaller


# ---- fake duck-typed governor (what the callers actually depend on) ----
class _FakeLease:
    def __init__(self):
        self.released = False

    def release(self):
        self.released = True


class _FakeGovernor:
    def __init__(self):
        self.calls = []
        self.last = None

    def lease(self, model, est_gb):
        self.last = _FakeLease()
        self.calls.append((model, est_gb))
        return self.last


# ---- GovernorClient (mocked HTTP) ----
def test_governor_lease_admits_then_release(monkeypatch):
    gc = GovernorClient()
    seen = []

    def fake_req(method, path, payload=None):
        seen.append((method, path, payload))
        if path == "/admit":
            return {"ok": True, "grant": True, "lease_id": "L1", "reserved_gb": payload["est_gb"], "job_id": payload["job_id"]}
        return {"ok": True, "released": 1}

    monkeypatch.setattr(gc, "_req", fake_req)
    lease = gc.lease("m", 2.0, job_id="j1")
    assert isinstance(lease, _Lease) and lease.lease_id == "L1" and lease.granted
    lease.release()
    assert seen[0][1] == "/admit" and seen[0][2]["est_gb"] == 2.0
    assert seen[-1][1] == "/release" and seen[-1][2] == {"job_id": "j1"}


def test_governor_admit_queues_then_grants(monkeypatch):
    gc = GovernorClient(poll_s=0)  # no real sleep
    calls = {"n": 0}

    def fake_req(method, path, payload=None):
        calls["n"] += 1
        granted = calls["n"] >= 3  # queued twice, then granted
        return {"ok": True, "grant": granted, "job_id": payload.get("job_id"), "lease_id": "L9"}

    monkeypatch.setattr(gc, "_req", fake_req)
    r = gc.admit("m", 1.0, job_id="jq")
    assert r["grant"] is True and calls["n"] == 3


# ---- governed callers: admit BEFORE load, release on close ----
def test_llamacpp_admits_before_load_and_releases():
    g = _FakeGovernor()
    c = LlamaCppCaller("/nonexistent.gguf", governor=g, est_gb=1.5)
    try:
        c._ensure_loaded()  # llama_cpp import (or model load) fails AFTER admit — that's fine
    except Exception:
        pass
    assert g.calls == [("/nonexistent.gguf", 1.5)]  # governor asked BEFORE the load
    assert c._lease is g.last
    c.close()
    assert g.last.released is True and c._lease is None


def test_mlx_admits_before_load():
    g = _FakeGovernor()
    c = MLXCaller("some/model-id", governor=g, est_gb=2.0)
    try:
        c._ensure_loaded()
    except Exception:
        pass
    assert g.calls == [("some/model-id", 2.0)]


def test_llamacpp_estimate_gb_override_and_fallback():
    assert LlamaCppCaller("/x.gguf", est_gb=3.5)._estimate_gb() == 3.5
    assert LlamaCppCaller("/nonexistent.gguf")._estimate_gb() == 4.0  # missing file -> safe default


def test_callers_without_governor_are_ungoverned():
    # close() with no governor/lease must be safe (runs anywhere)
    LlamaCppCaller("/x.gguf").close()
    MLXCaller("m").close()
