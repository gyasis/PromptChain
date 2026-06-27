"""AutoResearch — a PromptChain-native "research, then build until verified" tool.

The Karpathy/Ralph **tool-forge**, packaged as a util you can register as a tool:
a brief goes in, a **runner-verified** artifact comes out. The verified-build core
IS :class:`~promptchain.utils.test_loop_chain.MicroPromptChain` (the sandboxed
iterate-until-tests-pass engine) — the exact "sandboxed docker exec + success-bar
check" stage the external ``autoresearch`` system left as a TODO. AutoResearch wraps
it with two optional stages:

  1. **research** — gather knowledge for the brief (pluggable: DeepLake / web /
     GitHub / anything). Its notes are folded into the build objective.
  2. **build**    — MicroPromptChain writes code until the success test passes
     (the falsifiable bar; the loop's pass-gate is the anti-hallucination guard).
  3. **critique** — an optional critic verdict gate over the verified build.

The success bar is supplied by the caller (a spec/test), per the autoresearch
constitution ("a spec supplies the test, or the model proposes one"). Auto-proposing
the test, and the full 7-agent / HITL bounce state-machine (via ``AgentChain`` router
mode), are documented extension points — not needed for the tool-usable core.

See ``research/foundation/architecture/05-tool-creation-and-subchains.md`` and
``06-test-loop-chain.md``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

from .test_loop_chain import GenerateFn, LoopResult, MicroPromptChain, run_coro_blocking

# brief -> research notes (gather stage). Return a string the build objective can use.
ResearchFn = Callable[[str], Awaitable[str]]
# (brief, LoopResult) -> verdict string (critic stage). "APPROVE"/"PASS" => accepted.
CritiqueFn = Callable[[str, LoopResult], Awaitable[str]]

_NOTES_MAX_CHARS = 4000  # token economy: bound the research notes folded into the objective


@dataclass
class ResearchResult:
    """Outcome of :meth:`AutoResearch.run`. Truthy iff the artifact is verified."""
    brief: str
    notes: str
    build: LoopResult
    verdict: str
    verified: bool
    winning_code: Optional[str] = None
    attempts: list = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.verified

    def to_dict(self) -> dict:
        """JSON-able summary (so AutoResearch can be wrapped as a tool — tools must
        return ``str``/``json.dumps(...)``, never a raw dict)."""
        return {
            "verified": self.verified,
            "result": self.build.result,
            "verdict": self.verdict,
            "iterations": self.build.iterations,
            "stopped_by": self.build.stopped_by,
            "winning_code": self.winning_code,
            "notes_chars": len(self.notes),
        }


def _approves(verdict: str) -> bool:
    v = (verdict or "").upper()
    if "REJECT" in v or "FAIL" in v:
        return False
    return "APPROVE" in v or "PASS" in v or "ACCEPT" in v


class AutoResearch:
    """research -> build-until-verified -> critique, reusing :class:`MicroPromptChain`.

    :param model: model for the build generator (or pass ``generate=``).
    :param generate: async ``(prompt) -> text`` override for the build generator.
    :param research: optional async ``(brief) -> notes`` gather stage.
    :param critique: optional async ``(brief, LoopResult) -> verdict`` critic gate.
    :param language/image/install_command/max_iterations/max_seconds/use_docker/
        executor/verbose: forwarded to the underlying :class:`MicroPromptChain`.
    """

    def __init__(self, model=None, *, generate: Optional[GenerateFn] = None,
                 research: Optional[ResearchFn] = None, critique: Optional[CritiqueFn] = None,
                 language: str = "python", image: str = "python:3.12-slim",
                 install_command: Optional[str] = None, max_iterations: int = 10,
                 max_seconds: Optional[float] = None, use_docker: bool = True,
                 executor=None, verbose: bool = False) -> None:
        self._research = research
        self._critique = critique
        self.verbose = verbose
        self._loop = MicroPromptChain(
            model=model, generate=generate, language=language, image=image,
            install_command=install_command, max_iterations=max_iterations,
            max_seconds=max_seconds, use_docker=use_docker, executor=executor,
            verbose=verbose)

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[AutoResearch] {msg}")

    @staticmethod
    def _build_objective(brief: str, notes: str, constraints: Optional[list]) -> str:
        parts = [brief.strip()]
        if constraints:
            parts.append("Constraints:\n" + "\n".join(f"- {c}" for c in constraints))
        if notes:
            n = notes if len(notes) <= _NOTES_MAX_CHARS else notes[:_NOTES_MAX_CHARS] + "\n...[truncated]"
            parts.append("Research notes (use what's relevant):\n" + n)
        return "\n\n".join(parts)

    async def run(self, brief: str, *, target_file: str = "solution.py",
                  test_command: Optional[str] = None, deps: Optional[dict] = None,
                  install_command: Optional[str] = None, constraints: Optional[list] = None,
                  starting_code: str = "", runner=None) -> ResearchResult:
        """Research the brief, build code until ``test_command`` passes, then (optionally)
        critique. ``test_command``/``deps`` (or ``runner=``) define the success bar."""
        # 1. research (optional gather)
        notes = ""
        if self._research is not None:
            self._log("research stage")
            notes = await self._research(brief) or ""

        # 2. build until the success test passes (the verified core)
        objective = self._build_objective(brief, notes, constraints)
        self._log("build stage (MicroPromptChain)")
        build = await self._loop.run(
            objective=objective, target_file=target_file, test_command=test_command,
            deps=deps, install_command=install_command, starting_code=starting_code,
            runner=runner)

        # 3. critique (optional gate)
        verdict = "skipped"
        if self._critique is not None:
            self._log("critique stage")
            verdict = await self._critique(brief, build) or ""

        verified = (build.result == "PASS") and (self._critique is None or _approves(verdict))
        self._log(f"verified={verified} (build={build.result}, verdict={verdict!r})")
        return ResearchResult(
            brief=brief, notes=notes, build=build, verdict=verdict, verified=verified,
            winning_code=build.winning_code, attempts=build.attempts)

    def run_sync(self, *args, **kwargs) -> ResearchResult:
        """Synchronous wrapper around :meth:`run` — safe even if an event loop is
        already running (see :func:`~promptchain.utils.test_loop_chain.run_coro_blocking`)."""
        return run_coro_blocking(self.run(*args, **kwargs))


async def auto_research(brief: str, *, model=None, generate: Optional[GenerateFn] = None,
                        research: Optional[ResearchFn] = None,
                        critique: Optional[CritiqueFn] = None, **run_kwargs) -> ResearchResult:
    """One-shot convenience: construct an :class:`AutoResearch` and run a brief."""
    ar = AutoResearch(model=model, generate=generate, research=research, critique=critique,
                      **{k: run_kwargs.pop(k) for k in list(run_kwargs)
                         if k in {"language", "image", "install_command", "max_iterations",
                                  "max_seconds", "use_docker", "executor", "verbose"}})
    return await ar.run(brief, **run_kwargs)
