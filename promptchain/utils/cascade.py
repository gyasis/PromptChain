"""Verify-then-escalate cascade: try tiers in order, keep the first output that passes a check.

A cascade is one plain Callable chain step. It runs the cheapest tier on the step input, asks
``check(step_input, output)`` whether the output is acceptable, and only on a failure runs the
next tier on the SAME original input. It consumes no model slot of its own::

    from promptchain import PromptChain
    from promptchain.utils.cascade import cascade, decision_check
    from promptchain.utils.decision_step import SystemOneBackend

    step = cascade(
        [cheap_chain, frontier_chain],
        check=decision_check(SystemOneBackend(model="tev1:4b"),
                             instructions="Does the output fully answer the input?",
                             accept=0.9),
        name="answer",
    )
    chain = PromptChain(models=[], instructions=[step])

A tier is a ``PromptChain`` or a Callable step, exactly like a ``route_on`` branch. A tier that
raises is recorded on its attempt and the cascade escalates. A check that raises propagates:
an unusable verifier is never read as a pass or a fail. When every tier fails the check the
cascade runs ``on_exhausted`` on the original input, or raises ``CascadeExhausted`` carrying
every attempt.
"""

from __future__ import annotations

import inspect
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Awaitable, Callable, List, Optional, Sequence, Tuple, Union

from .decision_step import (
    Branch,
    DecisionBackend,
    _run_branch,
    band,
    normalize_answer,
    validate_questions,
)

logger = logging.getLogger(__name__)

CheckResult = Union[bool, Tuple[bool, str]]
# (step_input, output) -> bool or (bool, reason); sync or async.
Check = Callable[[str, str], Union[CheckResult, Awaitable[CheckResult]]]


@dataclass
class TierAttempt:
    """One tier's run: whether its output passed the check, and why."""

    tier: int
    label: str
    passed: bool
    reason: str
    elapsed_s: float
    output: Optional[str]
    error: Optional[str] = None


@dataclass
class CascadeResult:
    """Every attempt of one cascade run and which tier (if any) produced the answer."""

    name: str
    output: Optional[str]
    answered_by: Optional[str]
    tier: Optional[int]
    attempts: List[TierAttempt] = field(default_factory=list)

    @property
    def escalated(self) -> bool:
        """True when the answer did not come from the first tier."""
        return len(self.attempts) > 1

    def to_dict(self) -> dict:
        d = asdict(self)
        d["escalated"] = self.escalated
        return d


class CascadeExhausted(RuntimeError):
    """Every tier failed the check. ``result`` holds all attempts, including tier errors."""

    def __init__(self, result: CascadeResult):
        self.result = result
        detail = "; ".join(
            f"[{a.tier}] {a.label}: {a.error or a.reason}" for a in result.attempts
        )
        super().__init__(f"cascade {result.name!r} exhausted: {detail}")


def _label_of(tier: Branch, index: int) -> str:
    models = getattr(tier, "models", None)
    if models:
        first = models[0]
        if isinstance(first, dict):
            return str(first.get("name", f"tier{index}"))
        return str(first)
    return getattr(tier, "__name__", None) or f"tier{index}"


async def _run_check(check: Check, step_input: str, output: str) -> Tuple[bool, str]:
    res = check(step_input, output)
    if inspect.isawaitable(res):
        res = await res
    if isinstance(res, tuple):
        passed, reason = res
        return bool(passed), str(reason)
    return bool(res), ""


def cascade(
    tiers: Sequence[Branch],
    *,
    check: Check,
    name: str,
    labels: Optional[Sequence[str]] = None,
    on_exhausted: Optional[Branch] = None,
    on_result: Optional[Callable[[CascadeResult], None]] = None,
) -> Callable[[str], Awaitable[str]]:
    """A Callable chain step: run ``tiers`` in order until ``check`` accepts an output.

    ``check(step_input, output)`` returns ``bool`` or ``(bool, reason)`` and may be async.
    Returns the accepted output as plain text. The returned step exposes the latest run as
    ``step.last_result``; under concurrent calls that attribute holds whichever run finished
    last, so concurrent callers read ``on_result`` instead.
    """
    tiers = list(tiers)
    if not tiers:
        raise ValueError("cascade needs at least one tier")
    if labels is not None and len(labels) != len(tiers):
        raise ValueError(f"labels ({len(labels)}) must match tiers ({len(tiers)})")
    if not callable(check):
        raise ValueError("cascade needs a check callable")
    names = list(labels) if labels is not None else [_label_of(t, i) for i, t in enumerate(tiers)]

    async def _cascade_step(state: str) -> str:
        attempts: List[TierAttempt] = []

        def finish(output: Optional[str], index: Optional[int]) -> CascadeResult:
            result = CascadeResult(
                name=name,
                output=output,
                answered_by=names[index] if index is not None else None,
                tier=index,
                attempts=attempts,
            )
            _cascade_step.last_result = result
            if on_result is not None:
                on_result(result)
            return result

        for i, tier in enumerate(tiers):
            t0 = time.time()
            try:
                output = await _run_branch(tier, state)
            except Exception as e:  # a failing tier escalates; the error stays on the attempt
                attempt = TierAttempt(
                    i, names[i], False, "tier raised", round(time.time() - t0, 3), None,
                    error=f"{type(e).__name__}: {e}",
                )
            else:
                passed, reason = await _run_check(check, state, output)
                attempt = TierAttempt(
                    i, names[i], passed, reason, round(time.time() - t0, 3), output
                )
            attempts.append(attempt)
            logger.info(
                "cascade %s tier %d (%s): passed=%s reason=%r error=%r",
                name, i, names[i], attempt.passed, attempt.reason, attempt.error,
            )
            if attempt.passed:
                return finish(attempt.output, i).output

        result = finish(None, None)
        if on_exhausted is not None:
            return await _run_branch(on_exhausted, state)
        raise CascadeExhausted(result)

    _cascade_step.last_result = None
    _cascade_step.__name__ = f"cascade_{name}"
    return _cascade_step


def decision_check(backend: DecisionBackend, *, instructions: str, accept: float) -> Check:
    """A check that asks one boolean question through a decision backend.

    The backend sees ``<input>…</input><output>…</output>``. The output passes when the answer
    is True and its measured confidence is at least ``accept``.
    """
    if not 0 <= accept <= 1:
        raise ValueError(f"accept ({accept}) must be within 0..1")
    questions = {"ok": {"type": "boolean", "instructions": instructions}}
    validate_questions(questions)

    async def _check(step_input: str, output: str) -> Tuple[bool, str]:
        state = f"<input>{step_input}</input><output>{output}</output>"
        reply = await backend.decide(state, questions)
        ans = normalize_answer("ok", questions["ok"], (reply.get("answers") or {}).get("ok"))
        which = band(ans["confidence"], accept, accept)
        passed = ans["value"] is True and which == "accept"
        return passed, f"value={ans['value']} confidence={ans['confidence']} band={which}"

    return _check
