"""RED (test-first) integration tests for the ModelProfiler harness (NOT YET IMPLEMENTED).

Pins the harness contract from specs/013-model-profiler/contracts/profiler-api.md (D7):
    from promptchain.profiler import ModelProfiler
    ModelProfiler(*, store_path=None, transcript_dir=None, item_bank=None)
    .run_probe(model_id, *, bank=None, model_runner=None, scorer=None,
               dimensions=None, persist=True) -> CapabilityProfile

Each probe trial is written as a schema-valid F1 transcript (per
specs/012-sio-output-integration/contracts/transcript-schema.md) whose model_call
lines carry the probed model id. A deterministic fake model_runner + scorer are
injected so NO live model is called.

pytest-asyncio is NOT installed; we drive only the sync run_probe wrapper.
MUST fail now (ModelProfiler / probe.py are stubs).
"""
from __future__ import annotations

import glob
import json
import os

import pytest

from promptchain.profiler import ModelProfiler
from promptchain.profiler.item_bank import ItemBank, CAPABILITY_DIMENSIONS

MODEL_ID = "ollama/test-model:1b"


def _fake_runner(item):
    """Deterministic canned reply (no live call)."""
    return {
        "text": f"reply-{item.item_id}",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "latency_ms": 12.0,
    }


def _fake_scorer(item, reply):
    """Deterministic outcome driven purely by item difficulty."""
    return (item.b <= 1.0, {"latency_ms": reply["latency_ms"]})


def _profiler(tmp_path, sub):
    store = os.path.join(str(tmp_path), sub, "model_profiles.json")
    tdir = os.path.join(str(tmp_path), sub, "transcripts")
    return ModelProfiler(store_path=store, transcript_dir=tdir), store, tdir


def _synthetic_bank(n_per_dim=8):
    return ItemBank.synthetic(dimensions=CAPABILITY_DIMENSIONS, n_per_dim=n_per_dim, seed=0)


# --------------------------------------------------------------------------- #
# 1. persists profile
# --------------------------------------------------------------------------- #
def test_run_probe_persists_profile(tmp_path):
    profiler, store, _ = _profiler(tmp_path, "run1")
    bank = _synthetic_bank()
    profile = profiler.run_probe(
        MODEL_ID, bank=bank, model_runner=_fake_runner, scorer=_fake_scorer, persist=True
    )

    assert profile.model_id == MODEL_ID
    assert profile.skills, "skills must be non-empty"
    for est in profile.skills.values():
        assert hasattr(est, "theta")
        assert hasattr(est, "se")
        assert hasattr(est, "capability")
    assert isinstance(profile.recommended_tier, str) and profile.recommended_tier
    assert isinstance(profile.budget_tokens, int) and profile.budget_tokens > 0

    # persisted to disk, keyed by model id under "profiles"
    assert os.path.exists(store)
    with open(store) as fh:
        data = json.load(fh)
    assert MODEL_ID in data["profiles"]


# --------------------------------------------------------------------------- #
# 2. reproducible (SC-002)
# --------------------------------------------------------------------------- #
def test_run_probe_reproducible(tmp_path):
    p1, _, _ = _profiler(tmp_path, "rep_a")
    p2, _, _ = _profiler(tmp_path, "rep_b")
    bank = _synthetic_bank()

    prof1 = p1.run_probe(MODEL_ID, bank=bank, model_runner=_fake_runner, scorer=_fake_scorer)
    prof2 = p2.run_probe(MODEL_ID, bank=bank, model_runner=_fake_runner, scorer=_fake_scorer)

    common = set(prof1.skills) & set(prof2.skills)
    assert common
    for dim in common:
        assert prof1.skills[dim].theta == pytest.approx(prof2.skills[dim].theta, abs=1e-6)


# --------------------------------------------------------------------------- #
# 3. bounded item count (SC-003 — the CAT stop fired)
# --------------------------------------------------------------------------- #
def test_probe_bounded_item_count(tmp_path):
    profiler, _, _ = _profiler(tmp_path, "bounded")
    bank = _synthetic_bank(n_per_dim=60)  # far larger than needed
    profile = profiler.run_probe(
        MODEL_ID, bank=bank, model_runner=_fake_runner, scorer=_fake_scorer
    )

    total = sum(est.n_items for est in profile.skills.values())
    assert total <= len(bank.items)
    for est in profile.skills.values():
        assert est.n_items <= 30  # hard CAT cap


# --------------------------------------------------------------------------- #
# 4. trials emitted as valid F1 transcripts (SC-005 / FR-007)
# --------------------------------------------------------------------------- #
def test_trials_emitted_as_f1_transcripts(tmp_path):
    profiler, _, tdir = _profiler(tmp_path, "transcripts")
    bank = _synthetic_bank()
    profiler.run_probe(MODEL_ID, bank=bank, model_runner=_fake_runner, scorer=_fake_scorer)

    files = glob.glob(os.path.join(tdir, "**", "*.jsonl"), recursive=True)
    assert len(files) >= 1

    for path in files:
        with open(path) as fh:
            lines = [ln for ln in fh.read().splitlines() if ln.strip()]
        assert lines, f"transcript {path} is empty"

        parsed = [json.loads(ln) for ln in lines]  # every line is standalone JSON

        first = parsed[0]
        assert first["type"] == "chain_start"
        assert "schema_version" in first

        last = parsed[-1]
        assert last["type"] in ("chain_end", "chain_error")

        model_calls = [p for p in parsed if p.get("type") == "model_call"]
        assert any(mc.get("model") == MODEL_ID for mc in model_calls)


# --------------------------------------------------------------------------- #
# 5. uncalibrated / empty bank raises
# --------------------------------------------------------------------------- #
def test_uncalibrated_bank_raises(tmp_path):
    profiler, _, _ = _profiler(tmp_path, "uncal")
    with pytest.raises((ValueError, RuntimeError)):
        profiler.run_probe(
            MODEL_ID, bank=ItemBank(items=[]), model_runner=_fake_runner, scorer=_fake_scorer
        )
