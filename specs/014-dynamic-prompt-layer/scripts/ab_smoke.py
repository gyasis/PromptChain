#!/usr/bin/env python3
"""F3 offline live smoke — A/B against a real weak LAN ollama model (SC-007).

Mirrors F2's offline live smoke: no secrets, talks to a LAN ollama via litellm's
``ollama/<model>`` prefix + ``OLLAMA_API_BASE`` (default the Mac Studio at
192.168.0.159). Runs the deterministic A/B harness (``promptchain.prompts.eval_ab``)
with a REAL model runner over both arms (f3 per-model prompt vs the static base) and
prints per-arm completion rate + the delta — the live "weak model beats static base"
demonstration that can't be a CI unit test.

Run (needs the worktree on PYTHONPATH; see quickstart.md section 6):

    OLLAMA_API_BASE=http://192.168.0.159:11434 \
    PYTHONPATH=/home/gyasis/Documents/PromptChain.wt-epic-adaptive-prompting \
    python specs/014-dynamic-prompt-layer/scripts/ab_smoke.py --weak ollama/llama3.2:1b

An unreachable model exits non-zero with a clear message (never a raw traceback).
"""
from __future__ import annotations

import argparse
import os
import sys

from promptchain.prompts.eval_ab import EvalTask, run_ab
from promptchain.profiler.jacket import CapabilityProfile, Jacket


class _SeededStore:
    """A one-model in-memory profile store (the F2 ``get_profile`` seam).

    Seeds a TINY-tier profile so the f3 arm assembles the small, family-adapted
    prompt (tiny budget) — what a weak model should actually receive.
    """

    def __init__(self, model_id: str) -> None:
        jacket = Jacket(
            tier="tiny",
            budget_tokens=600,
            mode="single-shot+retry",
            spawn_temp=0.8,
            compress_at=0.6,
            max_turns=10,
            role="executor",
            escalate=False,
        )
        self._profile = CapabilityProfile(
            model_id=model_id,
            capability=0.2,
            recommended_tier="tiny",
            budget_tokens=600,
            jacket=jacket,
        )

    def get_profile(self, model_id: str):
        return self._profile if model_id == self._profile.model_id else None


def _make_runner(model: str, api_base: str):
    """Build a real ``model_runner(prompt, task) -> str`` backed by litellm."""
    from litellm import completion

    def runner(prompt: str, task: EvalTask) -> str:
        resp = completion(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": task.objective},
            ],
            api_base=api_base,
            temperature=0.0,
        )
        return resp["choices"][0]["message"]["content"] or ""

    return runner


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="F3 offline live A/B smoke (LAN ollama).")
    parser.add_argument(
        "--weak",
        default="ollama/llama3.2:1b",
        help="weak model id (litellm ollama/<model> prefix); default ollama/llama3.2:1b",
    )
    parser.add_argument(
        "--model-base",
        default=os.environ.get("OLLAMA_API_BASE", "http://192.168.0.159:11434"),
        help="ollama API base (default $OLLAMA_API_BASE or http://192.168.0.159:11434)",
    )
    args = parser.parse_args(argv)

    model = args.weak
    api_base = args.model_base
    print(f"F3 A/B offline live smoke — model={model} api_base={api_base}\n")

    store = _SeededStore(model)
    try:
        runner = _make_runner(model, api_base)
        report = run_ab(runner, model=model, store=store)
    except Exception as exc:  # unreachable model / litellm error — fail clean
        print(f"ERROR: A/B smoke could not reach the model: {exc}", file=sys.stderr)
        print(
            "Check that the LAN ollama is up and OLLAMA_API_BASE is correct.",
            file=sys.stderr,
        )
        return 1

    rates = report.per_arm_completion_rate
    print("Per-arm completion rate:")
    for arm in ("f3", "static_base"):
        print(f"  {arm:<12} {rates.get(arm, 0.0):.2%}")
    print(f"\nDelta (f3 - static_base): {report.delta:+.2%}")
    verdict = "WIN" if report.delta > 0 else ("TIE" if report.delta == 0 else "REGRESSION")
    print(f"Verdict: F3 {verdict} on {model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
