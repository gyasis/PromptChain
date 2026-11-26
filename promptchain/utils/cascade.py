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
from dataclasses import dataclass, field, fields
from typing import Any, Awaitable, Callable, List, Optional, Sequence, Tuple, Union

from .decision_step import (
    Branch,
    DecisionBackend,
    DecisionError,
    _run_branch,
    band,
    normalize_answer,
    validate_questions,
)

logger = logging.getLogger(__name__)

CheckResult = Union[bool, Tuple[bool, str]]
# (step_input, output) -> bool or (bool, reason); sync or async.
Check = Callable[[str, str], Union[CheckResult, Awaitable[CheckResult]]]


def _fields_dict(obj: Any) -> dict:
    """Dataclass fields as a dict without deep-copying values (exceptions stay as they are)."""
    return {f.name: getattr(obj, f.name) for f in fields(obj)}


@dataclass
class TierAttempt:
    """One tier's run: whether its output passed the check, and why.

    ``exception`` keeps the raised exception object (tier or check); ``to_dict`` omits it.
    """

    tier: int
    label: str
    passed: bool
    reason: str
    elapsed_s: float
    output: Optional[str]
    error: Optional[str] = None
    exception: Optional[BaseException] = field(default=None, repr=False, compare=False)


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
        attempts = [
            {k: v for k, v in _fields_dict(a).items() if k != "exception"}
            for a in self.attempts
        ]
        return {
            "name": self.name,
            "output": self.output,
            "answered_by": self.answered_by,
            "tier": self.tier,
            "attempts": attempts,
            "escalated": self.escalated,
        }


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
    if isinstance(res, bool):
        return res, ""
    if isinstance(res, (tuple, list)) and len(res) == 2 and isinstance(res[0], bool):
        return res[0], str(res[1])
    raise TypeError(
        f"check must return bool or (bool, reason); got {type(res).__name__}: {res!r}"
    )


async def _call_tier(tier: Branch, state: str) -> str:
    """Run one tier and return its text. A tier that returns None is an error, not "None"."""
    if hasattr(tier, "process_prompt_async"):
        out = await tier.process_prompt_async(state)
    else:
        out = tier(state)
        if inspect.isawaitable(out):
            out = await out
    if out is None:
        raise ValueError("tier returned None")
    return str(out)


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

    ``check(step_input, output)`` returns ``bool`` or ``(bool, reason)`` and may be async; any
    other return type raises ``TypeError``. Returns the accepted output as plain text.

    Every run ends with one ``CascadeResult``: it is stored on ``step.last_result`` and passed
    to ``on_result``. A run where the check raised ends with no answer before the exception
    propagates. A run answered by ``on_exhausted`` has ``answered_by="on_exhausted"``. Under
    concurrent calls ``last_result`` holds whichever run finished last, so concurrent callers
    read ``on_result``. An ``on_result`` exception is logged and never discards the answer.
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
        _cascade_step.last_result = None
        attempts: List[TierAttempt] = []

        def finish(output: Optional[str], index: Optional[int], by: Optional[str] = None) -> None:
            result = CascadeResult(
                name=name,
                output=output,
                answered_by=by if by is not None else (names[index] if index is not None else None),
                tier=index,
                attempts=attempts,
            )
            _cascade_step.last_result = result
            if on_result is not None:
                try:
                    on_result(result)
                except Exception:
                    logger.exception("cascade %s: on_result callback raised", name)

        last_exc: Optional[BaseException] = None
        for i, tier in enumerate(tiers):
            t0 = time.time()
            try:
                output = await _call_tier(tier, state)
            except Exception as e:  # a failing tier escalates; the exception stays on the attempt
                last_exc = e
                attempt = TierAttempt(
                    i, names[i], False, "tier raised", round(time.time() - t0, 3), None,
                    error=f"{type(e).__name__}: {e}", exception=e,
                )
                logger.warning(
                    "cascade %s tier %d (%s) raised: %s", name, i, names[i], attempt.error,
                    exc_info=True,
                )
                attempts.append(attempt)
                continue
            try:
                passed, reason = await _run_check(check, state, output)
            except Exception as e:
                attempts.append(TierAttempt(
                    i, names[i], False, "check raised", round(time.time() - t0, 3), output,
                    error=f"check raised: {type(e).__name__}: {e}", exception=e,
                ))
                logger.error("cascade %s tier %d (%s): check raised", name, i, names[i],
                             exc_info=True)
                finish(None, None)
                raise
            attempt = TierAttempt(i, names[i], passed, reason, round(time.time() - t0, 3), output)
            attempts.append(attempt)
            logger.info(
                "cascade %s tier %d (%s): passed=%s reason=%r",
                name, i, names[i], passed, reason,
            )
            if passed:
                finish(output, i)
                return output

        if on_exhausted is not None:
            try:
                fallback = await _run_branch(on_exhausted, state)
            except Exception:
                finish(None, None)
                raise
            finish(fallback, None, by="on_exhausted")
            return fallback
        finish(None, None)
        raise CascadeExhausted(_cascade_step.last_result) from last_exc

    _cascade_step.last_result = None
    _cascade_step.__name__ = f"cascade_{name}"
    return _cascade_step


def _html_escape(text: str) -> str:
    return text.replace("<", "&lt;").replace(">", "&gt;")


def _rate(value: float, what: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise ValueError(f"{what} ({value!r}) must be a number within 0..1")


def _p_true(ans: dict) -> float:
    probs = ans.get("probabilities") or {}
    for key in ("True", True, "true"):
        if key in probs:
            return float(probs[key])
    raise DecisionError(f"min_probability needs a probability for True; answer has {ans}")


def decision_check(
    backend: DecisionBackend,
    *,
    instructions: str,
    accept: float,
    min_probability: Optional[float] = None,
) -> Check:
    """A check that asks one boolean question through a decision backend.

    The backend sees ``<input>…</input><output>…</output>``, with ``<`` and ``>`` in both
    fields escaped. Without ``min_probability`` the output passes when the answer is True and
    the measured confidence is at least ``accept``. A backend that reports only a probability
    gets an entropy confidence (p=0.97 -> 0.81), so ``accept=0.9`` then needs p >= 0.988.
    With ``min_probability`` the output passes when the answer is True and the probability of
    True is at least ``min_probability``; ``accept`` is not consulted.
    """
    _rate(accept, "accept")
    if min_probability is not None:
        _rate(min_probability, "min_probability")
    questions = {"ok": {"type": "boolean", "instructions": instructions}}
    validate_questions(questions)

    async def _check(step_input: str, output: str) -> Tuple[bool, str]:
        state = f"<input>{_html_escape(step_input)}</input><output>{_html_escape(output)}</output>"
        reply = await backend.decide(state, questions)
        if not isinstance(reply, dict) or not isinstance(reply.get("answers"), dict):
            raise DecisionError(f"backend reply has no answers dict: {reply!r}")
        ans = normalize_answer("ok", questions["ok"], reply["answers"].get("ok"))
        if min_probability is not None:
            p = _p_true(ans)
            passed = ans["value"] is True and p >= min_probability
            return passed, f"value={ans['value']} p_true={p} min_probability={min_probability}"
        which = band(ans["confidence"], accept, accept)
        passed = ans["value"] is True and which == "accept"
        return passed, f"value={ans['value']} confidence={ans['confidence']} band={which}"

    return _check
