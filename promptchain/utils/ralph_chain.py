"""RalphChain — the multi-agent, fresh-context "MA loop" (engine B).

The loop the micro-agent **fork** added — its ``ralph-*`` state machine. Per outer
iteration: **librarian -> artisan -> critic -> testing**, then the context is
**reset** and a fresh iteration runs; only the structured test result (+ critic
notes) threads forward. Brute-force toward a falsifiable goal (tests pass) while
**discarding accumulated context each pass** — the Ralph principle that keeps even
weak / local models converging (every pass is a clean, bounded task, not an
ever-growing transcript).

Mirrors ``micro-agent/src/state-machine/ralph-machine.ts`` (the staged pipeline) +
``lifecycle/iteration-manager.ts`` (fresh reset + entropy/iteration bounds), made
**language-agnostic** and **PromptChain-native**:

  * **librarian** — gather/summarize the context needed (cheap model)
  * **artisan**   — write the COMPLETE target file from the brief (strong model)
  * **critic**    — review the code before testing (medium model; advisory — the
                    test is the hard gate; its notes thread forward on failure)
  * **testing**   — run the test command in the sandbox (exit 0 = pass)

**Heterogeneous models per role** (the "big-brother" design) — exactly the fork's
table (librarian=gemini-flash, artisan=claude-sonnet, critic=gpt-4o-mini). Pass one
``model`` for all, or ``librarian_model`` / ``artisan_model`` / ``critic_model``.

Engine **A** (:class:`~promptchain.utils.test_loop_chain.MicroPromptChain`) is the
substrate; this **explodes the single generator into the staged pipeline** while
reusing the identical sandbox/test contract. See
``research/foundation/architecture/06-test-loop-chain.md``.
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Optional

from .external_loop import ExternalLoop
from .test_loop_chain import (
    GenerateFn,
    extract_code,
    make_generator,
    resolve_executor,
    truncate_tail,
)

_SIG_CHARS = 240  # entropy: how much of the (normalized) test output forms a failure signature


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
@dataclass
class RalphIteration:
    """One fresh-context iteration through the staged pipeline (full observability)."""
    iteration: int
    librarian: str
    code: str
    critic: str
    passed: bool
    test_output: str


@dataclass
class RalphResult:
    """Outcome of :meth:`RalphChain.run`. ``result`` is one of
    ``PASS`` | ``FAIL`` | ``TIMEOUT`` | ``STAGNATED`` | ``NO_CODE``."""
    result: str
    winning_code: Optional[str]
    iterations: int
    stopped_by: str
    history: list = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.result == "PASS"


# --------------------------------------------------------------------------- #
# Stage prompts (lean, model-agnostic; each stage gets a FRESH prompt — no
# accumulated transcript, only threaded structured state)
# --------------------------------------------------------------------------- #
_LIBRARIAN = """You are the LIBRARIAN agent. Gather and summarize ONLY the context needed to solve the objective in {language}.
Produce a short brief: the key requirements, the current state of {target_file}, and — if a previous attempt failed — the precise reason. Do NOT write code.

<objective>
{objective}
</objective>

<current_file path="{target_file}">
{code}
</current_file>{failure}"""

_ARTISAN = """You are the ARTISAN agent. Using the librarian's brief, write the COMPLETE {language} contents of {target_file} so its tests pass.
Output ONLY one fenced ```{language} code block with the full file — no prose.

<objective>
{objective}
</objective>

<librarian_brief>
{brief}
</librarian_brief>

<current_file path="{target_file}">
{code}
</current_file>{failure}"""

_CRITIC = """You are the CRITIC agent. Review the artisan's code against the objective BEFORE it is tested.
If it looks correct and complete, reply with APPROVE on the first line. Otherwise list the concrete problems to fix (brief, bullet form).

<objective>
{objective}
</objective>

<code path="{target_file}">
{code}
</code>"""

_FAILURE = """

<last_attempt_failed>
{output}
</last_attempt_failed>"""


def _approves(text: str) -> bool:
    if not text or not text.strip():
        return False
    return "APPROVE" in text.strip().splitlines()[0].upper()


def _norm_sig(text: str) -> str:
    """Normalize test output into a stagnation signature (drop digits/paths/whitespace)."""
    t = re.sub(r"0x[0-9a-fA-F]+|\d+", "#", text or "")
    t = re.sub(r"\s+", " ", t).strip()
    return t[:_SIG_CHARS]


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #
class RalphChain:
    """Multi-agent, fresh-context iterate-until-tests-pass loop (engine B)."""

    def __init__(self, model=None, *, librarian_model=None, artisan_model=None,
                 critic_model=None, librarian: Optional[GenerateFn] = None,
                 artisan: Optional[GenerateFn] = None, critic: Optional[GenerateFn] = None,
                 language: str = "python", image: str = "python:3.12-slim",
                 install_command: Optional[str] = None, max_iterations: int = 10,
                 max_seconds: Optional[float] = None, network: Optional[str] = "none",
                 timeout: int = 120, use_docker: bool = True, executor=None,
                 run_critic: bool = True, entropy_threshold: int = 3,
                 verbose: bool = False) -> None:
        self.language = language
        self.image = image
        self.install_command = install_command
        self.max_iterations = max_iterations
        self.max_seconds = max_seconds
        self.network = network
        self.timeout = timeout
        self.use_docker = use_docker
        self._executor = executor
        self.run_critic = run_critic
        self.entropy_threshold = entropy_threshold
        self.verbose = verbose

        def pick(injected, role_model):
            if injected is not None:
                return injected
            m = role_model or model
            if m is None:
                raise ValueError(
                    "provide model= (shared) or a per-role model / callable for each agent")
            return make_generator(m)

        self._librarian = pick(librarian, librarian_model)
        self._artisan = pick(artisan, artisan_model)
        # critic only needed when run_critic
        self._critic = (pick(critic, critic_model) if run_critic
                        else (lambda _p: _noop()))

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[RalphChain] {msg}")

    async def run(self, objective: str, target_file: str = "solution.py", *,
                  test_command: Optional[str] = None, deps: Optional[dict] = None,
                  install_command: Optional[str] = None, starting_code: str = "",
                  runner=None) -> RalphResult:
        if runner is not None:
            self.image = runner.image
            if install_command is None:
                install_command = runner.install_command
            if test_command is None:
                test_command = runner.test_command(target_file)
        if not test_command:
            raise ValueError("test_command is required (or pass runner=)")
        install = install_command if install_command is not None else self.install_command
        full_test = f"({install}) && ({test_command})" if install else test_command

        executor, owns = resolve_executor(
            self._executor, self.use_docker, self.image, self.network, self.timeout)
        try:
            for path, content in (deps or {}).items():
                executor.write_file(path, content)

            state: dict = {"code": starting_code or "", "failure": "",
                           "winning_code": None, "result": "FAIL",
                           "history": [], "sigs": []}

            def entropy_breaker(it: int, st: dict):
                sigs = st["sigs"]
                k = self.entropy_threshold
                if k >= 1 and len(sigs) >= k and len(set(sigs[-k:])) == 1:
                    st["result"] = "STAGNATED"
                    return True, f"stagnated(entropy={k})"
                return False, ""

            async def step(it: int, st: dict) -> bool:
                fail_block = _FAILURE.format(output=st["failure"]) if st["failure"] else ""
                # --- librarian (fresh context: only objective + threaded state) ---
                brief = await self._librarian(_LIBRARIAN.format(
                    language=self.language, objective=objective, target_file=target_file,
                    code=st["code"] or "(empty)", failure=fail_block))
                # --- artisan ---
                raw = await self._artisan(_ARTISAN.format(
                    language=self.language, objective=objective, target_file=target_file,
                    brief=brief, code=st["code"] or "(empty)", failure=fail_block))
                code = extract_code(raw, self.language)
                if not code:
                    self._log(f"iter {it}: artisan produced no code")
                    st["result"] = "NO_CODE"
                    st["history"].append(RalphIteration(it, brief, "", "", False, "no code"))
                    return True
                # --- critic (advisory; the test is the hard gate) ---
                critic_out = ""
                if self.run_critic:
                    critic_out = await self._critic(_CRITIC.format(
                        language=self.language, objective=objective,
                        target_file=target_file, code=code)) or ""
                # --- testing ---
                executor.write_file(target_file, code)
                res = executor.run(full_test)
                passed = res.exit_code == 0 and not res.timed_out
                out = truncate_tail(res.output)
                st["history"].append(RalphIteration(it, brief, code, critic_out, passed, out))
                self._log(f"iter {it}: {'PASS' if passed else 'fail'} (exit={res.exit_code})")
                if passed:
                    st["winning_code"] = code
                    st["result"] = "PASS"
                    return False
                # thread forward: test output + critic notes (the ONLY carried state)
                notes = out
                if critic_out and not _approves(critic_out):
                    notes = out + "\n\nCritic notes (pre-test review):\n" + critic_out
                st["failure"] = truncate_tail(notes)
                st["code"] = code
                st["sigs"].append(_norm_sig(res.output))
                st["result"] = "TIMEOUT" if res.timed_out else "FAIL"
                return True  # fresh iteration next

            loop = ExternalLoop(max_iters=self.max_iterations, max_seconds=self.max_seconds,
                                breakers=(entropy_breaker,))
            state = await loop.run(step, state)

            result = "PASS" if state["winning_code"] else state.get("result", "FAIL")
            return RalphResult(
                result=result, winning_code=state["winning_code"],
                iterations=state.get("_iterations", 0),
                stopped_by=state.get("_stopped", ""), history=state["history"])
        finally:
            if owns and hasattr(executor, "stop"):
                executor.stop()

    def run_sync(self, *args, **kwargs) -> RalphResult:
        """Synchronous convenience wrapper around :meth:`run`."""
        return asyncio.run(self.run(*args, **kwargs))


async def _noop() -> str:
    return ""
