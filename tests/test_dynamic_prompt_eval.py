"""F3 A/B eval harness — TEST-FIRST (red).

Pins SC-007 / D7: a tiny (N=5) set of programmatically-checkable coding tasks run
through an A/B harness over two arms — ``f3`` (``DynamicModelPromptGenerator``) vs
``static_base`` (``DynamicTUIPromptGenerator``) — scored deterministically with a
FAKE, injected model (no live calls). This test proves the harness MEASURES a
directional win deterministically; the REAL "weak model beats static base" claim
is the separate OFFLINE LIVE SMOKE (LAN ollama), not this test.

This file MUST FAIL today: ``promptchain.prompts.eval_ab`` does not exist yet, so
collection errors with ImportError. Do NOT implement to make it pass here.

Run: ``python -m pytest tests/test_dynamic_prompt_eval.py -q``
"""
from __future__ import annotations

from promptchain.prompts.eval_ab import (  # noqa: E402  (import-time red expected)
    EVAL_TASKS,
    EvalReport,
    EvalTask,
    run_ab,
)
from promptchain.prompts.tiers import TINY_DIRECTIVE
from promptchain.profiler.jacket import CapabilityProfile, Jacket

# A genuinely weak model id, profiled TINY, so the f3 generator injects the TINY
# focusing directive (the real, non-harmful F3 differentiator) into the floor.
# Discriminator (see _scripted_fake): the f3 arm carries TINY_DIRECTIVE; the bare
# static base never does. (Earlier this keyed off a per-family "FORMAT:" preamble,
# but that preamble was removed for MEASURABLY HARMING weak models — 2026-06-27.)
_MODEL = "ollama/llama3.2:1b"
_F3_MARKER = TINY_DIRECTIVE
_WRONG = "<<<deliberately-wrong-output: this never satisfies a task check>>>"


class _TinyStore:
    """Profile store seeding a TINY capability profile for the weak model."""

    def get_profile(self, model_id):
        if model_id != _MODEL:
            return None
        jacket = Jacket(
            tier="tiny", budget_tokens=600, mode="single-shot+retry",
            spawn_temp=0.8, compress_at=0.6, max_turns=10, role="executor",
            escalate=False,
        )
        return CapabilityProfile(
            model_id=_MODEL, capability=0.2, recommended_tier="tiny",
            budget_tokens=600, jacket=jacket,
        )


def _store() -> _TinyStore:
    return _TinyStore()


def _scripted_fake(prompt: str, task: EvalTask) -> str:
    """Injected fake model: PASS on the f3 arm, FAIL on the static_base arm.

    Arm is detected purely from the assembled ``prompt`` — the f3 arm (TINY tier)
    carries the TINY focusing directive; the static base does not. On the f3 arm we
    return the task's KNOWN-CORRECT output (``task.check`` passes); on static_base a
    constant wrong output (``task.check`` fails).
    """
    is_f3 = _F3_MARKER in prompt
    correct = getattr(task, "answer", None)
    assert correct is not None, "EvalTask must expose its known-correct output"
    return correct if is_f3 else _WRONG


def test_eval_tasks_shape() -> None:
    """``EVAL_TASKS`` is exactly 5 tasks, each with the contracted fields."""
    assert isinstance(EVAL_TASKS, list)
    assert len(EVAL_TASKS) == 5
    for task in EVAL_TASKS:
        assert isinstance(task.id, str) and task.id
        assert isinstance(task.objective, str) and task.objective
        assert isinstance(task.tools, list)
        assert callable(task.check)
        # The fake needs a known-correct output to score the f3 arm green.
        assert getattr(task, "answer", None) is not None


def test_run_ab_measures_directional_win() -> None:
    """f3 beats static_base on the scripted fake; report is well-formed."""
    report = run_ab(_scripted_fake, model=_MODEL, store=_store(), budgets=(1000, 300))

    assert isinstance(report, EvalReport)

    rates = report.per_arm_completion_rate
    assert set(rates) == {"f3", "static_base"}

    # completion rates are proportions in [0, 1].
    for arm, rate in rates.items():
        assert 0.0 <= rate <= 1.0, f"{arm} rate {rate} out of [0,1]"

    # the directional win: f3 > static_base, and delta = f3 - static_base > 0.
    assert rates["f3"] > rates["static_base"]
    assert report.delta > 0
    assert abs(report.delta - (rates["f3"] - rates["static_base"])) < 1e-9


def test_run_ab_is_deterministic() -> None:
    """Same fake + same inputs → identical aggregate (deterministic aggregation)."""
    r1 = run_ab(_scripted_fake, model=_MODEL, store=_store(), budgets=(1000, 300))
    r2 = run_ab(_scripted_fake, model=_MODEL, store=_store(), budgets=(1000, 300))
    assert r1.per_arm_completion_rate == r2.per_arm_completion_rate
    assert r1.delta == r2.delta
