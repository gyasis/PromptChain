"""Model router (experimental, unvalidated): choose a model per task from task difficulty and model ability.

A difficulty source scores a task in [0, 1]. A fitted link maps (model ability theta, task
difficulty d) to P(pass). The router picks the first model, in the caller's cheapest-first
order, whose P(pass) reaches a threshold. It returns the rest of the list as the escalation order.

**Unvalidated: `link` must be fit on outcome data; there is no default link.** Without a link
the router has no basis to choose, so it uses the most able model and states why in the plan.
No data shows that it beats a fixed model. The validation experiment in
``docs/llms/recipes/recipe-model-router.md`` gates its use.

The router never blocks a chain. The difficulty call runs under ``asyncio.wait_for(timeout_s)``.
A failed call logs ONE warning, and the plan uses the most able model. ``plan`` never raises.
``asyncio.wait_for`` cannot stop a blocking call that already runs in an executor thread
(``SystemOneBackend`` is one), so give the backend its own short ``timeout`` as well.

The module does not import the profiler until a stored profile or a probability is needed.
It imports numpy inside ``DifficultyLink.fit``. ``import promptchain`` already loads numpy
and scipy, so those two are not lazy.

``asyncio.CancelledError`` from the caller's task propagates. The router does not swallow it.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import (
    Any, Awaitable, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple,
)

from promptchain.utils.decision_step import (
    DecisionBackend,
    DecisionError,
    normalize_answer,
    validate_questions,
)

logger = logging.getLogger(__name__)

DIFFICULTY_QUESTION = "difficulty"
MIN_FIT_OUTCOMES = 30


# --------------------------------------------------------------------------- difficulty


@dataclass
class Difficulty:
    """A task difficulty in [0, 1] (0 = easiest) and where it came from."""

    value: float
    source: str
    model: Optional[str] = None
    distribution: Optional[dict] = None

    def __post_init__(self) -> None:
        if not isinstance(self.value, (int, float)) or not math.isfinite(self.value):
            raise ValueError(f"Difficulty.value must be a finite number, got {self.value!r}")
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"Difficulty.value must be in [0, 1], got {self.value}")
        self.value = float(self.value)


DifficultyFn = Callable[[str], Awaitable[Difficulty]]


def _level_index(key: Any, levels: Sequence[str]) -> Optional[int]:
    if key in levels:
        return list(levels).index(key)
    try:
        i = int(key)
    except (TypeError, ValueError):
        return None
    return i if 0 <= i < len(levels) else None


def decision_difficulty(
    backend: DecisionBackend,
    *,
    instructions: str,
    levels: Sequence[str],
    prepare_state: Optional[Callable[[str], str]] = None,
) -> DifficultyFn:
    """A DifficultyFn backed by one decision-backend ``score`` question.

    ``levels`` are the ordered answer levels, easiest first. The difficulty is the expected
    level index divided by ``len(levels) - 1``: from the answer's ``probabilities`` (keyed by
    level label or by index) when it has them, else from the numeric score read as a level
    index. ``prepare_state`` trims or reshapes the task before it is sent (tev1 has a
    2,050-token limit). Backend errors propagate as ``DecisionError`` for ``ModelRouter`` to
    handle.
    """
    levels = list(levels)
    questions = {
        DIFFICULTY_QUESTION: {"type": "score", "instructions": instructions, "criteria": levels}
    }
    validate_questions(questions)
    if backend is None:
        raise DecisionError("decision_difficulty needs an explicit backend")
    top = len(levels) - 1

    async def _difficulty(task: str) -> Difficulty:
        state = prepare_state(task) if prepare_state is not None else task
        reply = await backend.decide(state, questions)
        raw = (reply.get("answers") or {}).get(DIFFICULTY_QUESTION)
        ans = normalize_answer(DIFFICULTY_QUESTION, questions[DIFFICULTY_QUESTION], raw)
        probs = ans.get("probabilities")
        dist: Optional[dict] = None
        index: Optional[float] = None
        if isinstance(probs, dict) and probs:
            mass = 0.0
            weighted = 0.0
            for key, p in probs.items():
                i = _level_index(key, levels)
                if i is None or not isinstance(p, (int, float)) or not math.isfinite(p) or p < 0:
                    continue
                mass += p
                weighted += p * i
            if mass > 0:
                index = weighted / mass
                dist = dict(probs)
            else:
                logger.debug("model_router: no probability key maps to a level in %s; "
                             "using the score", list(probs))
        if index is None:
            index = float(ans["value"])
            if not math.isfinite(index):
                raise DecisionError(f"difficulty score {ans['value']!r} is not finite")
        index = min(max(index, 0.0), float(top))
        return Difficulty(
            value=index / top,
            source=f"decision:{backend.name}",
            model=reply.get("model", backend.model),
            distribution=dist,
        )

    # ModelRouter reads this to compare the backend timeout with its own bound.
    _difficulty.backend = backend  # type: ignore[attr-defined]
    return _difficulty


# --------------------------------------------------------------------------- link


@dataclass
class RoutingOutcome:
    """One labelled routing record: a task run on a model, and whether it passed."""

    task_id: str
    task: str
    difficulty: float
    difficulty_source: str
    model: str
    theta: float
    passed: bool
    source: str
    reason: str = ""


@dataclass
class DifficultyLink:
    """P(pass | theta, d) = sigmoid(a * (theta - (alpha + beta * d))), fit on outcomes.

    ``auroc`` is measured on the data the link was fit on (in-sample); hold out a split for
    an honest number.
    """

    a: float
    alpha: float
    beta: float
    n: int
    auroc: Optional[float] = None

    def p_success(self, theta: float, d: float) -> float:
        from promptchain.profiler.irt import prob_3pl

        return prob_3pl(theta, self.a, self.alpha + self.beta * d, 0.0)

    @classmethod
    def fit(cls, outcomes: Iterable[Any]) -> "DifficultyLink":
        """Fit by logistic regression (IRLS) on ``(theta, difficulty, passed)`` records.

        Accepts ``RoutingOutcome`` or any object with those three attributes; ``passed`` must be
        a bool. Raises ``ValueError`` on fewer than ``MIN_FIT_OUTCOMES`` records, a single outcome
        class, fewer than two distinct thetas or difficulties, no convergence, perfectly
        separable data (the parameters are then unidentified), a degenerate result
        (``a < 1e-6`` or ``|alpha|`` or ``|beta|`` above 1e3), or a non-positive discrimination
        ``a`` (stronger models must pass more often). Logs a warning when ``beta <= 0``: harder
        tasks then do not lower the pass probability, so difficulty carries no signal.
        """
        import numpy as np

        rows = []
        for o in outcomes:
            if not isinstance(o.passed, (bool, np.bool_)):
                raise ValueError(f"fit: passed must be a bool, got {o.passed!r}")
            rows.append((float(o.theta), float(o.difficulty), 1.0 if o.passed else 0.0))
        if len(rows) < MIN_FIT_OUTCOMES:
            raise ValueError(
                f"fit needs at least {MIN_FIT_OUTCOMES} outcomes, got {len(rows)}"
            )
        data = np.asarray(rows, dtype=float)
        theta, d, y = data[:, 0], data[:, 1], data[:, 2]
        if not np.all(np.isfinite(data)):
            raise ValueError("fit: outcomes contain a non-finite theta or difficulty")
        if y.min() == y.max():
            raise ValueError("fit: all outcomes have the same result; need both passes and fails")
        if np.ptp(theta) == 0 or np.ptp(d) == 0:
            raise ValueError("fit: need at least two distinct thetas and two distinct difficulties")

        x = np.column_stack([np.ones_like(theta), theta, d])
        w = np.zeros(3)
        ridge = 1e-6
        converged = False
        for _ in range(100):
            z = np.clip(x @ w, -30.0, 30.0)
            p = 1.0 / (1.0 + np.exp(-z))
            grad = x.T @ (y - p) - ridge * w
            hess = (x * (p * (1.0 - p))[:, None]).T @ x + ridge * np.eye(3)
            step = np.linalg.solve(hess, grad)
            w = w + step
            if np.max(np.abs(step)) < 1e-9:
                converged = True
                break
        if not converged:
            raise ValueError("fit: IRLS did not converge (data are likely separable)")
        z = x @ w
        if np.all((z > 0) == (y == 1.0)):
            raise ValueError(
                "fit: the outcomes are perfectly separable, so a, alpha and beta are not "
                "identified; collect more mixed outcomes"
            )
        w0, a, w2 = (float(v) for v in w)
        if not (math.isfinite(a) and a > 0):
            raise ValueError(
                f"fit: discrimination a={a:.4g} is not positive; stronger models are not "
                "passing more often in this data, so no usable link exists"
            )
        alpha, beta = -w0 / a, -w2 / a
        if a < 1e-6 or abs(alpha) > 1e3 or abs(beta) > 1e3:
            raise ValueError(
                f"fit: degenerate parameters a={a:.3g}, alpha={alpha:.3g}, beta={beta:.3g}; "
                "the data are separable or carry no signal"
            )
        if beta <= 0:
            logger.warning(
                "model_router: fitted beta=%.3g <= 0; harder tasks do not lower P(pass), "
                "so the difficulty score carries no signal for this data", beta
            )
        return cls(a=a, alpha=alpha, beta=beta, n=len(rows), auroc=_auroc(z, y))


def _auroc(score: Any, y: Any) -> Optional[float]:
    """Rank-based AUROC (ties averaged); None if a class is empty."""
    import numpy as np

    pos = y == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    ranks = np.empty(len(score), dtype=float)
    i = 0
    while i < len(score):
        j = i
        while j + 1 < len(score) and sorted_score[j + 1] == sorted_score[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


# --------------------------------------------------------------------------- router


@dataclass
class RoutePlan:
    """The router's decision for one task."""

    chosen: str
    order: List[str]
    p_success: Dict[str, Optional[float]]
    difficulty: Optional[Difficulty]
    degraded: Optional[str]
    reason: str

    def to_dict(self) -> dict:
        return {
            "chosen": self.chosen,
            "order": list(self.order),
            "p_success": dict(self.p_success),
            "difficulty": asdict(self.difficulty) if self.difficulty is not None else None,
            "degraded": self.degraded,
            "reason": self.reason,
        }


class ModelRouter:
    """Choose a model per task. ``models`` run cheapest first, most able last.

    ``threshold`` is the P(pass) a model must reach to be chosen; set it per task family from
    an evaluation, not a global value. ``timeout_s`` bounds the difficulty call and must be
    finite and positive. Abilities come from ``abilities[model]``, else from a stored profile
    (``store_path``) using ``dimension`` or, when ``dimension`` is None, the mean theta over the
    capability dimensions.

    With ``link=None`` the router does not call the difficulty source: without a link the
    difficulty cannot change the choice. To collect difficulty values before a link exists,
    call the difficulty function yourself when you record outcomes.

    If the difficulty function carries a ``backend`` (``decision_difficulty`` does) whose
    ``timeout`` exceeds ``timeout_s``, the constructor logs a warning. A hung backend then keeps
    its executor threads busy after the router has moved on; see the recipe.
    """

    def __init__(
        self,
        models: Sequence[str],
        *,
        threshold: float,
        timeout_s: float,
        difficulty: Optional[DifficultyFn] = None,
        link: Optional[DifficultyLink] = None,
        abilities: Optional[Mapping[str, float]] = None,
        store_path: Optional[Any] = None,
        dimension: Optional[str] = None,
    ) -> None:
        models = list(models)
        if not models:
            raise ValueError("ModelRouter needs at least one model")
        if len(set(models)) != len(models):
            raise ValueError(f"ModelRouter models must be unique: {models}")
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"threshold must be in (0, 1], got {threshold}")
        if not (isinstance(timeout_s, (int, float)) and math.isfinite(timeout_s) and timeout_s > 0):
            raise ValueError(f"timeout_s must be a finite number > 0, got {timeout_s!r}")
        self.models = models
        self.threshold = threshold
        self.timeout_s = timeout_s
        self.difficulty = difficulty
        self.link = link
        self.abilities = dict(abilities or {})
        self.store_path = store_path
        self.dimension = dimension
        self._warned: set = set()
        backend_timeout = getattr(getattr(difficulty, "backend", None), "timeout", None)
        if isinstance(backend_timeout, (int, float)) and backend_timeout > timeout_s:
            logger.warning(
                "model_router: backend timeout %ss exceeds timeout_s %ss; a hung backend "
                "keeps executor threads busy and can delay asyncio.run exit up to %ss. "
                "Set the backend timeout at or below timeout_s.",
                backend_timeout, timeout_s, backend_timeout,
            )

    def _warn_once(self, key: str, msg: str, *args: Any) -> None:
        if key not in self._warned:
            self._warned.add(key)
            logger.warning(msg, *args)

    # ---- abilities

    def _read_store(self) -> dict:
        """The stored profiles keyed by model id; {} when absent. Warns once on a corrupt file."""
        if self.store_path is None:
            return {}
        try:
            from promptchain.profiler.store import ProfileStore

            store = ProfileStore(self.store_path)
            profiles = store.load().get("profiles") or {}
            if not profiles and store.path.exists() and store.path.stat().st_size > 0:
                self._warn_once(
                    "store", "model_router: profile store %s exists but holds no readable "
                    "profiles (corrupt or empty); abilities fall back to explicit values",
                    store.path,
                )
            return profiles
        except Exception as e:  # a bad store never blocks routing
            self._warn_once("store-err", "model_router: profile store unreadable: %s", e)
            return {}

    def _ability(self, model: str, profiles: dict) -> "Tuple[Optional[float], Optional[str]]":
        """(theta, note) for one model. A note is attached to the plan reason."""
        if model in self.abilities:
            try:
                value = float(self.abilities[model])
            except (TypeError, ValueError):
                return None, "invalid ability"
            return (value, None) if math.isfinite(value) else (None, "invalid ability")
        raw = profiles.get(model)
        if raw is None:
            return None, None
        try:
            from promptchain.profiler.jacket import CapabilityProfile

            profile = CapabilityProfile.from_dict(raw)
            if not profile.skills:
                return None, None
            if self.dimension is not None:
                skill = profile.skills.get(self.dimension)
                skills = [skill] if skill is not None else []
            else:
                from promptchain.profiler.item_bank import CAPABILITY_DIMENSIONS

                skills = [s for d, s in profile.skills.items() if d in CAPABILITY_DIMENSIONS]
                skills = skills or list(profile.skills.values())
            if not skills:
                return None, None
            theta = sum(s.theta for s in skills) / len(skills)
            note = "low_precision theta" if any(s.low_precision for s in skills) else None
            return (theta, note) if math.isfinite(theta) else (None, "invalid ability")
        except Exception as e:
            self._warn_once(f"profile-{model}", "model_router: profile for %s unreadable: %s",
                            model, e)
            return None, None

    # ---- plan

    async def _score(self, task: str) -> "Tuple[Optional[Difficulty], Optional[str]]":
        if self.difficulty is None:
            return None, "no difficulty source configured"
        try:
            result = await asyncio.wait_for(self.difficulty(task), timeout=self.timeout_s)
            if not isinstance(result, Difficulty):
                raise TypeError(f"difficulty source returned {type(result).__name__}, not Difficulty")
            return result, None
        except asyncio.TimeoutError:
            msg = f"difficulty call timed out after {self.timeout_s}s"
        except Exception as e:
            msg = f"difficulty call failed: {type(e).__name__}: {e}"
        logger.warning("model_router: %s; falling back to %s", msg, self.models[-1])
        return None, msg

    async def plan(self, task: str) -> RoutePlan:
        """Pick a model for ``task``. Never raises; a cancelled caller still gets CancelledError."""
        models = self.models
        last = models[-1]
        try:
            needs_store = any(m not in self.abilities for m in models)
            profiles = self._read_store() if needs_store else {}
            thetas: Dict[str, Optional[float]] = {}
            notes: List[str] = []
            for m in models:
                thetas[m], note = self._ability(m, profiles)
                if note:
                    notes.append(f"{m}: {note}")
            known = [t for t in thetas.values() if t is not None]
            missing = [m for m in models if thetas[m] is None]
            if known != sorted(known):
                self._warn_once("order", "model_router: abilities are not non-decreasing in "
                                "list order %s; models should run cheapest to most able", models)
                notes.append("abilities not non-decreasing in list order")
            if missing and len(missing) < len(models):
                notes.append(f"no ability for {missing}")
            tail = f" ({'; '.join(notes)})" if notes else ""
            top = thetas[last] is not None and thetas[last] >= max(known)
            fallback = f"{'most able model' if top else 'last model in the list'} {last}"

            diff: Optional[Difficulty] = None
            degraded: Optional[str] = None
            if self.link is None:
                degraded = "no link: fit a DifficultyLink on outcome data"
            else:
                diff, degraded = await self._score(task)
            p: Dict[str, Optional[float]] = {m: None for m in models}
            if degraded is None and diff is not None:
                for m in models:
                    if thetas[m] is not None:
                        p[m] = self.link.p_success(thetas[m], diff.value)

            if degraded is not None:
                return RoutePlan(last, [last], p, diff, degraded,
                                 f"degraded, using {fallback}: {degraded}{tail}")
            for m in models:
                if p[m] is not None and p[m] >= self.threshold:
                    return RoutePlan(m, models[models.index(m):], p, diff, None,
                                     f"{m} is the first model with P(pass)={p[m]:.3f} "
                                     f">= {self.threshold}{tail}")
            why = (f"no ability for {missing}" if len(missing) == len(models)
                   else f"no model reaches P(pass) >= {self.threshold}")
            return RoutePlan(last, [last], p, diff, None, f"{why}; using {fallback}{tail}")
        except Exception as e:  # last-resort guard: planning must not stop a chain
            logger.warning("model_router: planning failed: %s: %s", type(e).__name__, e)
            return RoutePlan(last, [last], {m: None for m in models}, None,
                             f"planning failed: {type(e).__name__}: {e}",
                             f"degraded, using last model in the list {last}")


def route_model(
    router: ModelRouter,
    run: Callable[[str, str], Awaitable[str]],
    *,
    on_plan: Optional[Callable[[RoutePlan], Any]] = None,
) -> Callable[[str], Awaitable[str]]:
    """A chain step: plan, then ``await run(chosen_model, task)`` and return its text.

    ``on_plan`` receives each ``RoutePlan`` (sync or async) for logging or outcome capture;
    an error inside it is logged and does not stop the chain. Errors from ``run`` propagate.
    """

    async def _route_model_step(task: str) -> str:
        plan = await router.plan(task)
        if on_plan is not None:
            try:
                res = on_plan(plan)
                if inspect.isawaitable(res):
                    await res
            except Exception as e:
                logger.warning("model_router: on_plan callback failed: %s: %s", type(e).__name__, e)
        return await run(plan.chosen, task)

    _route_model_step.__name__ = "route_model"
    return _route_model_step


# --------------------------------------------------------------------------- outcomes JSONL


def write_outcomes(outcomes: Iterable[RoutingOutcome], path: Any, *, append: bool = True) -> int:
    """Write one JSON object per line; return the count written.

    Serializes every record first, so a record with NaN or infinity raises ``ValueError``
    before anything is written. On append, a missing final newline in the file is added first.
    """
    lines = []
    for i, o in enumerate(outcomes):
        try:
            lines.append(json.dumps(asdict(o), ensure_ascii=False, allow_nan=False))
        except ValueError as e:
            raise ValueError(f"outcome {i}: {e} (difficulty and theta must be finite)") from e
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lead = ""
    if append and p.exists() and p.stat().st_size > 0:
        with open(p, "rb") as fh:
            fh.seek(-1, 2)
            if fh.read(1) != b"\n":
                lead = "\n"
    with open(p, "a" if append else "w", encoding="utf-8") as fh:
        fh.write(lead + "".join(line + "\n" for line in lines))
    return len(lines)


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def read_outcomes(path: Any) -> List[RoutingOutcome]:
    """Read a JSONL file written by ``write_outcomes``.

    Raises ``ValueError`` naming the line when a record is malformed: ``passed`` must be a
    bool, ``difficulty`` and ``theta`` finite numbers, the other fields strings.
    """
    out: List[RoutingOutcome] = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                rec = RoutingOutcome(**json.loads(line))
                if not isinstance(rec.passed, bool):
                    raise ValueError(f"passed must be a bool, got {rec.passed!r}")
                for name in ("difficulty", "theta"):
                    if not _is_number(getattr(rec, name)):
                        raise ValueError(f"{name} must be a finite number, got {getattr(rec, name)!r}")
                for name in ("task_id", "task", "difficulty_source", "model", "source", "reason"):
                    if not isinstance(getattr(rec, name), str):
                        raise ValueError(f"{name} must be a string")
            except (ValueError, TypeError) as e:
                raise ValueError(f"{path}:{lineno}: bad outcome record: {e}") from e
            rec.difficulty = float(rec.difficulty)
            rec.theta = float(rec.theta)
            out.append(rec)
    return out
