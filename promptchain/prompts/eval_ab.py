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

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

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


@dataclass
class EvalReport:
    """Aggregate of an A/B run: per-arm completion rate + the F3 win (delta)."""

    results: List[EvalResult] = field(default_factory=list)
    per_arm_completion_rate: Dict[str, float] = field(default_factory=dict)
    delta: float = 0.0

    @classmethod
    def from_results(cls, results: List[EvalResult]) -> "EvalReport":
        """Compute per-arm completion rate (passed/total) and delta = f3 - static_base."""
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
        return cls(results=results, per_arm_completion_rate=rates, delta=delta)


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
