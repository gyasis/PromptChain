"""TEST-FIRST (Constitution III) failing tests for F3 budget measurement + fitting.

Targets ``promptchain.prompts.budget`` (empty stub → RED). Asserts the contract in
``contracts/prompt-layout.md`` (D5 budget enforcement) and ``generator-api.md``:
base is NEVER dropped/truncated; optionals dropped in ascending drop_priority.
"""
from __future__ import annotations

from promptchain.prompts.budget import fit_to_budget, measure
from promptchain.prompts.tiers import OptionalModule, PromptTier


# --------------------------------------------------------------------------- #
# measure
# --------------------------------------------------------------------------- #
def test_measure_non_negative():
    assert measure("hello world") >= 0


def test_measure_empty_is_zero():
    assert measure("") == 0


def test_measure_monotonic_with_length():
    short = "one two three"
    long = "one two three " * 100
    assert measure(long) > measure(short)


# --------------------------------------------------------------------------- #
# Fixtures: optional modules with distinct drop_priority + sizeable text
# --------------------------------------------------------------------------- #
def _modules():
    # examples = lowest priority (dropped FIRST), extended_guidance next, extras highest.
    return [
        OptionalModule(
            key="examples",
            text="EXAMPLES: " + ("example block alpha " * 20),
            drop_priority=1,
            tiers={PromptTier.EXTENDED},
        ),
        OptionalModule(
            key="extended_guidance",
            text="GUIDANCE: " + ("extended guidance bravo " * 20),
            drop_priority=2,
            tiers={PromptTier.EXTENDED},
        ),
        OptionalModule(
            key="extra_notes",
            text="NOTES: " + ("supplementary notes charlie " * 20),
            drop_priority=3,
            tiers={PromptTier.EXTENDED},
        ),
    ]


BASE = "STATIC BASE: parity floor text that must always survive."


# --------------------------------------------------------------------------- #
# fit_to_budget
# --------------------------------------------------------------------------- #
def test_base_always_present_generous_budget():
    assembled, dropped = fit_to_budget(
        BASE, _modules(), target_max=100_000, hard_cap=200_000
    )
    assert BASE in assembled
    assert dropped == []
    for m in _modules():
        assert m.text in assembled


def test_drops_in_ascending_drop_priority_order():
    # Tight budget: only the base (plus maybe nothing) fits → force drops.
    assembled, dropped = fit_to_budget(
        BASE, _modules(), target_max=measure(BASE) + 5, hard_cap=10_000
    )
    assert BASE in assembled
    # examples (priority 1) dropped first, then extended_guidance (2), then extra_notes (3).
    assert dropped == ["examples", "extended_guidance", "extra_notes"]


def test_partial_drop_keeps_highest_priority_module():
    mods = _modules()
    # Budget large enough for base + the highest-priority (extra_notes) module only.
    target = measure(BASE) + measure(mods[2].text) + 20
    assembled, dropped = fit_to_budget(BASE, mods, target_max=target, hard_cap=10_000)
    assert BASE in assembled
    # examples + extended_guidance dropped (ascending), extra_notes retained.
    assert dropped == ["examples", "extended_guidance"]
    assert mods[2].text in assembled
    assert mods[0].text not in assembled


def test_base_over_hard_cap_drops_all_optionals_keeps_base():
    big_base = "X" * 5000
    assembled, dropped = fit_to_budget(
        big_base, _modules(), target_max=10, hard_cap=50
    )
    # Parity floor never truncated, even over the hard cap.
    assert big_base in assembled
    assert dropped == ["examples", "extended_guidance", "extra_notes"]


def test_returns_tuple_str_and_list():
    assembled, dropped = fit_to_budget(BASE, _modules(), target_max=100_000, hard_cap=200_000)
    assert isinstance(assembled, str)
    assert isinstance(dropped, list)
