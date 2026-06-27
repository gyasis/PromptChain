"""TEST-FIRST (red) for F3 / US3 — weak-model longevity via Document-&-Clear.

Covers FR-011 (<turn-context> goal re-injection), FR-012 (compress threshold +
document_and_clear progress doc), FR-013 (stall detection + jacket-respecting
escalation + ≥10 task-turns before reset), and FR-014 (lossy fallback when the
working dir is not writable).

These import from `promptchain.prompts.longevity`, which is currently an empty
stub — so this module FAILS at collection (ImportError) until US3 is built.

pytest-asyncio is NOT installed → plain `def test_*`. Run with:
    python -m pytest tests/test_dynamic_prompt_longevity.py -q
"""
from __future__ import annotations

from pathlib import Path

from promptchain.prompts.longevity import build_turn_context, DocumentAndClear
from promptchain.profiler.jacket import Jacket


GOAL = "Refactor the payment module to remove the global currency singleton."


def _jacket(*, escalate: bool) -> Jacket:
    """Minimal valid Jacket with the escalate flag set as requested."""
    return Jacket(
        tier="tiny",
        budget_tokens=800,
        mode="heavy-loop",
        spawn_temp=0.4,
        compress_at=0.60,
        max_turns=30,
        role="executor",
        escalate=escalate,
    )


# --------------------------------------------------------------------------- #
# FR-011 — build_turn_context: <turn-context> goal re-injection
# --------------------------------------------------------------------------- #
def test_build_turn_context_reinjects_goal_and_turn():
    out = build_turn_context(GOAL, turn=4)
    assert isinstance(out, str)
    assert "<turn-context" in out
    assert "</turn-context>" in out
    assert GOAL in out  # goal re-injected verbatim
    assert "4" in out  # turn number present


def test_build_turn_context_includes_extra_when_given():
    extra = "Last result: tests for currency conversion now pass."
    out = build_turn_context(GOAL, turn=7, extra=extra)
    assert GOAL in out
    assert "7" in out
    assert extra in out


def test_build_turn_context_omits_extra_when_absent():
    out = build_turn_context(GOAL, turn=2)
    assert "Last result:" not in out


# --------------------------------------------------------------------------- #
# FR-012 — should_compress: usage >= compress_at
# --------------------------------------------------------------------------- #
def test_should_compress_threshold():
    dac = DocumentAndClear(compress_at=0.60)
    assert dac.should_compress(0.62) is True
    assert dac.should_compress(0.59) is False
    assert dac.should_compress(0.60) is True  # boundary inclusive


# --------------------------------------------------------------------------- #
# FR-012 — document_and_clear: writes a progress doc + returns reset history
# --------------------------------------------------------------------------- #
def _state():
    return {
        "goal": GOAL,
        "plan": ["audit singleton usages", "introduce a Currency context object"],
        "decisions": ["keep the public API stable for one release"],
        "progress": ["removed 3 of 7 singleton call-sites"],
        "old": "turn-3-noise",  # arbitrary old turn content that MUST be dropped
    }


def test_document_and_clear_writes_progress_doc(tmp_path):
    dac = DocumentAndClear()
    dac.document_and_clear(str(tmp_path), _state())

    md_files = list(tmp_path.glob("*.md"))
    assert md_files, "expected a progress doc (.md) written under working_dir"

    text = "\n".join(p.read_text(encoding="utf-8") for p in md_files)
    assert GOAL in text  # goal recorded
    # at least one plan/decision/progress item recorded
    assert (
        "audit singleton usages" in text
        or "keep the public API stable for one release" in text
        or "removed 3 of 7 singleton call-sites" in text
    )


def test_document_and_clear_returns_reset_history(tmp_path):
    dac = DocumentAndClear()
    resumed = dac.document_and_clear(str(tmp_path), _state())

    assert isinstance(resumed, list)
    assert len(resumed) >= 1
    assert all(isinstance(m, dict) for m in resumed)

    blob = repr(resumed)
    assert GOAL in blob  # resumes from the doc/goal, not cleared context
    # the cleared old turn content must NOT survive into the resumed history
    assert "turn-3-noise" not in blob


# --------------------------------------------------------------------------- #
# FR-013 — is_stalled: no measurable progress across the window
# --------------------------------------------------------------------------- #
def test_is_stalled_true_on_identical_signals():
    dac = DocumentAndClear()
    assert dac.is_stalled([5, 5, 5]) is True


def test_is_stalled_false_on_changing_signals():
    dac = DocumentAndClear()
    assert dac.is_stalled([1, 2, 3]) is False


# --------------------------------------------------------------------------- #
# FR-013 — should_escalate: stalled AND jacket.escalate
# --------------------------------------------------------------------------- #
def test_should_escalate_when_jacket_allows():
    dac = DocumentAndClear(jacket=_jacket(escalate=True))
    assert dac.should_escalate(stalled=True) is True
    assert dac.should_escalate(stalled=False) is False


def test_should_not_escalate_when_jacket_forbids():
    dac = DocumentAndClear(jacket=_jacket(escalate=False))
    # stalled but the jacket forbids escalation → never escalate
    assert dac.should_escalate(stalled=True) is False


def test_should_not_escalate_without_jacket_when_not_stalled():
    dac = DocumentAndClear(jacket=None)
    assert dac.should_escalate(stalled=False) is False


# --------------------------------------------------------------------------- #
# FR-013 — ≥10 task-turns before reset (min_turns threshold)
# --------------------------------------------------------------------------- #
def test_min_turns_threshold_exposed():
    dac = DocumentAndClear(min_turns=10)
    assert dac.min_turns == 10


# --------------------------------------------------------------------------- #
# FR-014 — lossy fallback: working_dir not writable → no raise, still list[dict]
# --------------------------------------------------------------------------- #
def test_document_and_clear_falls_back_when_dir_unwritable(tmp_path):
    # Create a FILE at the path so it cannot be used as a writable directory.
    blocker = tmp_path / "nope"
    blocker.write_text("not a directory", encoding="utf-8")
    assert blocker.is_file()

    dac = DocumentAndClear()
    resumed = dac.document_and_clear(str(blocker), _state())  # must NOT raise

    assert isinstance(resumed, list)
