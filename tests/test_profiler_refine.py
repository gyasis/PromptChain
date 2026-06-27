"""US3 tests (FR-012, FR-013) — EWMA refine + two-sided model×jacket fit + the DSPy
jacket generator (test-first, red until W10).

All deterministic: a fake model_runner + scorer, no live model, no SIO/DSPy LM (the
generator must degrade gracefully).
"""
from __future__ import annotations

import os

import pytest

from promptchain.profiler import ModelProfiler
from promptchain.profiler.item_bank import ItemBank, CAPABILITY_DIMENSIONS
from promptchain.profiler.jacket import Jacket

MODEL = "ollama/test:1b"


def _fake_runner(item):
    return {"text": "x", "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}, "latency_ms": 5.0}


def _fake_scorer(item, reply):
    return (item.b <= 1.0, {})


def _bank():
    return ItemBank.synthetic(dimensions=CAPABILITY_DIMENSIONS, n_per_dim=8, seed=0)


def _profiler(tmp_path):
    return ModelProfiler(
        store_path=os.path.join(str(tmp_path), "m.json"),
        transcript_dir=os.path.join(str(tmp_path), "t"),
    )


# --------------------------------------------------------------------------- #
# EWMA refine (FR-013)
# --------------------------------------------------------------------------- #
def test_refine_ewma_moves_toward_evidence(tmp_path):
    prof = _profiler(tmp_path)
    prof.run_probe(MODEL, bank=_bank(), model_runner=_fake_runner, scorer=_fake_scorer)
    old = prof.get_profile(MODEL).skills["reasoning_depth"].capability

    updated = prof.refine(MODEL, {"reasoning_depth": 0.95}, lam=0.5)
    new = updated.skills["reasoning_depth"].capability
    assert new == pytest.approx(0.5 * old + 0.5 * 0.95, abs=1e-9)
    # persisted
    assert prof.get_profile(MODEL).skills["reasoning_depth"].capability == pytest.approx(new, abs=1e-9)


def test_refine_empty_metrics_is_noop(tmp_path):
    prof = _profiler(tmp_path)
    prof.run_probe(MODEL, bank=_bank(), model_runner=_fake_runner, scorer=_fake_scorer)
    before = prof.get_profile(MODEL).skills["reasoning_depth"].capability
    same = prof.refine(MODEL, {})
    assert same.skills["reasoning_depth"].capability == pytest.approx(before, abs=1e-9)


def test_refine_increments_observation_count(tmp_path):
    prof = _profiler(tmp_path)
    prof.run_probe(MODEL, bank=_bank(), model_runner=_fake_runner, scorer=_fake_scorer)
    n0 = prof.get_profile(MODEL).n_observations
    updated = prof.refine(MODEL, {"reasoning_depth": 0.9})
    assert updated.n_observations == n0 + 1


# --------------------------------------------------------------------------- #
# two-sided model×jacket fit (FR-012)
# --------------------------------------------------------------------------- #
def test_jacket_fit_selects_max_lift(tmp_path):
    prof = _profiler(tmp_path)
    bank = _bank()

    good = Jacket(tier="rich", budget_tokens=8000, mode="heavy-loop", spawn_temp=0.5,
                  compress_at=0.6, max_turns=8, role="both")
    bad = Jacket(tier="lean", budget_tokens=2000, mode="single-shot", spawn_temp=0.2,
                 compress_at=0.85, max_turns=10, role="planner")

    def runner_for(jacket):
        # a "good" jacket raises the model's effective ability (more items pass)
        boost = 1.5 if jacket.tier == "rich" else -1.5

        def runner(item):
            ok = item.b <= boost
            return {"text": "yes" if ok else "no",
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    "latency_ms": 5.0}
        return runner

    scorer = lambda item, reply: (reply["text"] == "yes", {})

    result = prof.jacket_fit(MODEL, [bad, good], bank=bank, runner_for=runner_for,
                             scorer=scorer, baseline=bad)
    assert result["best"] == good            # the higher-lift jacket
    lifts = result["lifts"]
    assert lifts[1] > lifts[0]               # good lift > bad (baseline) lift
    assert lifts[0] == pytest.approx(0.0, abs=1e-9)  # baseline lift is 0 vs itself


# --------------------------------------------------------------------------- #
# DSPy jacket generator — degrades gracefully with no LM configured
# --------------------------------------------------------------------------- #
def test_prompt_generator_degrades_gracefully(tmp_path):
    from promptchain.profiler.prompt_generator import ModelPromptGenerator

    prof = _profiler(tmp_path)
    prof.run_probe(MODEL, bank=_bank(), model_runner=_fake_runner, scorer=_fake_scorer)
    profile = prof.get_profile(MODEL)

    gen = ModelPromptGenerator()
    jacket = gen.generate(profile)           # no DSPy LM available → returns the probe jacket
    assert isinstance(jacket, Jacket)
