"""F3 A/B eval harness — deterministic core (SC-007 / D7).

A tiny (N=5) set of scoped, programmatically-checkable coding tasks run through a
two-arm A/B harness:

- ``f3``          → ``DynamicModelPromptGenerator`` (the per-model F3 prompt)
- ``static_base`` → ``DynamicTUIPromptGenerator``   (the static foundation alone)

Each task×arm is scored deterministically by a programmatic ``EvalTask.check`` over
the output of an INJECTED ``model_runner(prompt, task) -> str``. No live calls live
here — the runner is supplied by the caller (a scripted fake in CI, a real LAN
ollama model in the offline live smoke ``scripts/ab_smoke.py``).

The harness only proves the MEASUREMENT is deterministic and directional; the real
"weak model beats the static base" claim is the offline live smoke, not this module.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from promptchain.prompts.budget import measure
from promptchain.prompts.model_dynamic import DynamicModelPromptGenerator
from promptchain.prompts.tui_dynamic import DynamicTUIPromptGenerator

# A model runner: takes the assembled prompt + the task, returns the model's output.
ModelRunner = Callable[[str, "EvalTask"], str]


@dataclass
class EvalTask:
    """A scoped, programmatically-checkable coding task.

    ``answer`` is the known-correct output marker; ``check`` returns True when that
    marker appears in the candidate output (case-insensitive substring). The check
    is intentionally lenient + deterministic so a fake runner returning ``answer``
    passes and a constant wrong output fails.
    """

    id: str
    objective: str
    tools: List[Dict[str, Any]]
    answer: str

    def check(self, output: str) -> bool:
        if not output:
            return False
        return self.answer.strip().lower() in output.lower()


# Exactly 5 small, deterministically-checkable tasks (D7: N=5).
#
# DIRECT-ANSWER framing (origin 2026-06-27): the tasks are phrased "reply with ONLY
# the answer" rather than "write and run a function". The smoke is SINGLE-SHOT text
# (nothing executes), so a "write and run + print" framing is an INVALID proxy — it
# measured "did the model happen to emit the literal value in code" (noise), and the
# foundation's heavy "USE TOOLS / act don't explain" guidance misleads a weak model
# into (impossible) tool use. Direct-answer is the valid single-shot measurement and
# lets the TINY simpler base's "answer directly" steer show through.
EVAL_TASKS: List[EvalTask] = [
    EvalTask(
        id="fibonacci",
        objective="Compute the 10th Fibonacci number (F(0)=0, F(1)=1) and reply with "
        "ONLY the number.",
        tools=[],
        answer="55",
    ),
    EvalTask(
        id="reverse_string",
        objective="Reverse the string 'hello' and reply with ONLY the reversed string.",
        tools=[],
        answer="olleh",
    ),
    EvalTask(
        id="sum_list",
        objective="Compute the sum of the integers [1, 2, 3, 4, 5] and reply with ONLY "
        "the number.",
        tools=[],
        answer="15",
    ),
    EvalTask(
        id="palindrome",
        objective="Is the string 'racecar' a palindrome? Reply with ONLY True or False.",
        tools=[],
        answer="True",
    ),
    EvalTask(
        id="fizzbuzz",
        objective="In FizzBuzz, what is printed for n=15? Reply with ONLY that single word.",
        tools=[],
        answer="FizzBuzz",
    ),
]


@dataclass
class EvalResult:
    """One task×arm (×budget) outcome."""

    arm: str
    task_id: str
    passed: bool
    tokens: int = 0
    budget: Optional[int] = None


def _percentile_interval(samples: List[float], ci_level: float) -> Tuple[float, float]:
    """Nearest-rank percentile interval of ``samples`` at ``ci_level``."""
    ordered = sorted(samples)
    last = len(ordered) - 1
    alpha = (1.0 - ci_level) / 2.0
    return ordered[round(alpha * last)], ordered[round((1.0 - alpha) * last)]


def _bootstrap_cis(
    results: List[EvalResult], *, n_boot: int, ci_level: float, seed: int
) -> Tuple[Dict[str, Tuple[float, float]], Tuple[float, float]]:
    """Paired task-cluster percentile bootstrap for per-arm rates and the delta.

    The resampling unit is the TASK, not the individual result: ``run_ab`` scores each
    task once per budget, so its rows are not independent, and resampling rows would
    overstate confidence. Both arms share each resampled task set, which keeps the
    f3 - static_base delta paired. Seeded, so deterministic for identical inputs.
    """
    # task_id -> arm -> [passed, total]
    clusters: Dict[str, Dict[str, List[int]]] = {}
    for r in results:
        counts = clusters.setdefault(r.task_id, {}).setdefault(r.arm, [0, 0])
        counts[0] += 1 if r.passed else 0
        counts[1] += 1
    task_ids = sorted(clusters)
    if not task_ids:
        return {}, (0.0, 0.0)

    arms = sorted({r.arm for r in results})
    rng = random.Random(seed)
    arm_samples: Dict[str, List[float]] = {arm: [] for arm in arms}
    delta_samples: List[float] = []
    for _ in range(n_boot):
        sample = rng.choices(task_ids, k=len(task_ids))
        rates: Dict[str, float] = {}
        for arm in arms:
            passed = sum(clusters[t].get(arm, [0, 0])[0] for t in sample)
            total = sum(clusters[t].get(arm, [0, 0])[1] for t in sample)
            if total:
                rates[arm] = passed / total
                arm_samples[arm].append(rates[arm])
        delta_samples.append(rates.get("f3", 0.0) - rates.get("static_base", 0.0))

    per_arm_ci = {
        arm: _percentile_interval(samples, ci_level)
        for arm, samples in arm_samples.items()
        if samples
    }
    return per_arm_ci, _percentile_interval(delta_samples, ci_level)


@dataclass
class EvalReport:
    """Aggregate of an A/B run: per-arm completion rate + the F3 win (delta).

    ``per_arm_ci`` / ``delta_ci`` are bootstrap confidence intervals at ``ci_level``.
    With N=5 tasks they are wide by design — a delta whose interval spans 0 is not
    yet distinguishable from noise.
    """

    results: List[EvalResult] = field(default_factory=list)
    per_arm_completion_rate: Dict[str, float] = field(default_factory=dict)
    delta: float = 0.0
    per_arm_ci: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    delta_ci: Tuple[float, float] = (0.0, 0.0)
    ci_level: float = 0.95

    @classmethod
    def from_results(
        cls,
        results: List[EvalResult],
        *,
        n_boot: int = 2000,
        ci_level: float = 0.95,
        seed: int = 0,
    ) -> "EvalReport":
        """Compute per-arm completion rate (passed/total), delta = f3 - static_base,
        and their paired task-cluster bootstrap CIs (see ``_bootstrap_cis``)."""
        totals: Dict[str, int] = {}
        passed: Dict[str, int] = {}
        for r in results:
            totals[r.arm] = totals.get(r.arm, 0) + 1
            passed[r.arm] = passed.get(r.arm, 0) + (1 if r.passed else 0)

        rates = {
            arm: (passed.get(arm, 0) / totals[arm]) if totals[arm] else 0.0
            for arm in totals
        }
        delta = rates.get("f3", 0.0) - rates.get("static_base", 0.0)
        per_arm_ci, delta_ci = _bootstrap_cis(
            results, n_boot=n_boot, ci_level=ci_level, seed=seed
        )
        return cls(
            results=results,
            per_arm_completion_rate=rates,
            delta=delta,
            per_arm_ci=per_arm_ci,
            delta_ci=delta_ci,
            ci_level=ci_level,
        )


def run_ab(
    model_runner: ModelRunner,
    *,
    model: str,
    store: Any = None,
    budgets: tuple = (1000, 300),
) -> EvalReport:
    """Run every ``EVAL_TASKS`` task through both arms (per budget) and aggregate.

    Deterministic for a deterministic ``model_runner``. The f3 arm differs from the
    static base via per-model tiering driven by ``store`` (e.g. a TINY-profiled model
    gets the simpler base + focusing directive); a fake runner can tell the arms
    apart by that directive. Budgets are iterated so each task contributes more than
    one observation per arm; the per-arm RATE is unaffected because the runner's
    per-arm verdict does not depend on budget.
    """
    f3_gen = DynamicModelPromptGenerator(model=model, store=store)
    base_gen = DynamicTUIPromptGenerator()
    arms = {"f3": f3_gen, "static_base": base_gen}

    results: List[EvalResult] = []
    for budget in budgets:
        for task in EVAL_TASKS:
            for arm_name, gen in arms.items():
                prompt = gen.generate(task.objective, task.tools)
                output = model_runner(prompt, task)
                results.append(
                    EvalResult(
                        arm=arm_name,
                        task_id=task.id,
                        passed=task.check(output),
                        tokens=measure(prompt),
                        budget=budget,
                    )
                )

    return EvalReport.from_results(results)
