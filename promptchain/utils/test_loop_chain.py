"""MicroPromptChain — a PromptChain-native "iterate until the tests pass" loop.

The lean substrate (the *original* micro-agent loop, generalized): one model
generates code, a sandbox runs the project's test command, and on failure the
test output is threaded back into the next generation — looping until the tests
pass or a bound (iterations / wall-clock) trips.

Built entirely on existing PromptChain primitives:
  * generation -> a single-instruction :class:`~promptchain.utils.promptchaining.PromptChain`
    (any model, no tools required)
  * the loop   -> :class:`~promptchain.utils.external_loop.ExternalLoop` (bounded, deterministic)
  * the sandbox-> :class:`~promptchain.utils.docker_executor.DockerExecutor` (container)
    or :class:`LocalExecutor` (host, trusted-only)

**Language-agnostic by construction:** the universal pass signal is the test
command's **exit code** (0 = pass), so swapping ``image`` + ``test_command``
(python -> rust / go / node) is all it takes. The model writes ``target_file``;
the sandbox runs ``test_command``; exit 0 stops the loop.

This is engine **(A)**. The multi-agent, fresh-context "RalphChain" (engine B —
the loop the micro-agent *fork* added: librarian -> artisan -> critic -> testing,
fresh context each iteration) wraps this same sandbox/test contract. See
``research/foundation/architecture/06-test-loop-chain.md``.
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
import threading
import warnings
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

from .docker_executor import DockerExecutor, ExecResult, extract_code_blocks
from .external_loop import ExternalLoop

# An injectable generator: (assembled_prompt) -> model output text.
GenerateFn = Callable[[str], Awaitable[str]]

_FAILURE_TAIL_CHARS = 3000  # token economy: keep the END of test output (failures live there)


# --------------------------------------------------------------------------- #
# Shared helpers (reused by RalphChain — engine B — so the sandbox/test contract
# is identical across both loops)
# --------------------------------------------------------------------------- #
def extract_code(raw: str, language: str) -> str:
    """Pull code out of model output: prefer a fenced block matching ``language``,
    else the longest block; if none, treat the whole output as code."""
    blocks = extract_code_blocks(raw or "")
    if not blocks:
        return (raw or "").strip()
    lang = (language or "").lower()
    matching = [c for ln, c in blocks if ln == lang]
    pool = matching or [c for _, c in blocks]
    return max(pool, key=len)


def truncate_tail(text: str, limit: int = _FAILURE_TAIL_CHARS) -> str:
    """Keep the last ``limit`` chars (test failures live at the end)."""
    text = text or ""
    if len(text) <= limit:
        return text
    return "...[truncated]...\n" + text[-limit:]


def run_coro_blocking(coro):
    """Run ``coro`` to completion and return its result — SAFELY whether or not an
    event loop is already running on this thread.

    The footgun this guards: ``asyncio.run()`` (and ``loop.run_until_complete``)
    raise ``RuntimeError: asyncio.run() cannot be called from a running event loop``
    when called from inside an already-running loop (e.g. a ``run_sync`` invoked from
    within the TUI's async tool path). Here we DETECT a running loop and, if present,
    execute the coroutine on a fresh loop in a short-lived worker thread (the outer
    loop is never touched). With no running loop, it's just ``asyncio.run``.

    Prefer ``await``-ing the async API directly; this exists so the synchronous
    ``run_sync`` wrappers can never blow up no matter where they are called from.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)  # no loop on this thread — the normal, cheap path

    box: dict = {}

    def _worker():
        loop = asyncio.new_event_loop()
        try:
            box["value"] = loop.run_until_complete(coro)
        except BaseException as exc:  # noqa: BLE001 — re-raised on the caller's thread
            box["error"] = exc
        finally:
            asyncio.set_event_loop(None)
            loop.close()

    t = threading.Thread(target=_worker, name="promptchain-run-sync", daemon=True)
    t.start()
    t.join()
    if "error" in box:
        raise box["error"]
    return box["value"]


def make_generator(model) -> "GenerateFn":
    """Build an async ``(prompt) -> text`` generator from a model, as a single-
    instruction PromptChain (any model, no tools)."""
    from .promptchaining import PromptChain
    chain = PromptChain(models=[model], instructions=["{input}"])

    async def _gen(prompt: str) -> str:
        return await chain.process_prompt_async(prompt)

    return _gen


def resolve_executor(executor, use_docker: bool, image: str,
                     network: Optional[str], timeout: int):
    """Return ``(executor, owns)``; ``owns`` is True when the caller must close it.
    Prefers :class:`DockerExecutor`; falls back to :class:`LocalExecutor` (warned)."""
    if executor is not None:
        return executor, False
    if use_docker and DockerExecutor.available():
        return DockerExecutor(image=image, network=network, timeout=timeout), True
    if use_docker:
        warnings.warn(
            "Docker unavailable — falling back to LocalExecutor (host, UNSANDBOXED). "
            "Run untrusted LLM code only under Docker.", RuntimeWarning, stacklevel=2)
    return LocalExecutor(timeout=timeout), True


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
@dataclass
class Attempt:
    """One iteration's record (for observability — every pass is logged)."""
    iteration: int
    code: str
    passed: bool
    output: str


@dataclass
class LoopResult:
    """The outcome of a :meth:`MicroPromptChain.run`.

    ``result`` is one of ``PASS`` | ``FAIL`` | ``TIMEOUT`` | ``NO_CODE``. Truthiness
    mirrors ``result == "PASS"``.
    """
    result: str
    winning_code: Optional[str]
    iterations: int
    stopped_by: str
    attempts: list = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.result == "PASS"


# --------------------------------------------------------------------------- #
# Executors
# --------------------------------------------------------------------------- #
class LocalExecutor:
    """Host-process executor with :class:`DockerExecutor`-parity (``work_dir`` /
    ``write_file`` / ``run`` -> :class:`ExecResult`).

    For **trusted** code only (CI, your own tests, quick local iteration) — it runs
    on the host with NO sandbox. Prefer :class:`DockerExecutor` for untrusted LLM
    output. Exists so the loop is runnable and unit-testable without a Docker daemon.
    """

    def __init__(self, work_dir: Optional[str] = None, timeout: int = 120) -> None:
        self._tmp = None
        if work_dir is None:
            self._tmp = tempfile.TemporaryDirectory(prefix="pc_local_")
            work_dir = self._tmp.name
        self.work_dir = os.path.abspath(work_dir)
        os.makedirs(self.work_dir, exist_ok=True)
        self.timeout = timeout

    def __enter__(self) -> "LocalExecutor":
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def stop(self) -> None:
        if self._tmp is not None:
            self._tmp.cleanup()
            self._tmp = None

    def write_file(self, relpath: str, content: str) -> str:
        dest = os.path.abspath(os.path.join(self.work_dir, relpath))
        if not dest.startswith(self.work_dir + os.sep) and dest != self.work_dir:
            raise ValueError(f"relpath escapes work_dir: {relpath!r}")
        os.makedirs(os.path.dirname(dest) or self.work_dir, exist_ok=True)
        with open(dest, "w") as f:
            f.write(content)
        return dest

    def run(self, command: str) -> ExecResult:
        # Disable .pyc writing: across iterations a same-size source rewritten within
        # one second would otherwise re-use a stale cached bytecode (mtime+size match).
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        try:
            p = subprocess.run(command, shell=True, cwd=self.work_dir, env=env,
                               capture_output=True, text=True, timeout=self.timeout)
            return ExecResult(p.returncode, (p.stdout or "") + (p.stderr or ""))
        except subprocess.TimeoutExpired as e:
            partial = ((e.stdout or "") + (e.stderr or "")) if (e.stdout or e.stderr) else ""
            return ExecResult(124, partial + f"\n[timeout after {self.timeout}s]", timed_out=True)


# --------------------------------------------------------------------------- #
# Optional test-runner sugar (image + install + default test command per language)
# --------------------------------------------------------------------------- #
@dataclass
class PytestRunner:
    """Convenience defaults for a Python/pytest project. Pass ``runner=PytestRunner()``
    to :meth:`MicroPromptChain.run` instead of spelling out image/install/test_command."""
    image: str = "python:3.12-slim"
    install_command: Optional[str] = "pip install -q pytest"

    def test_command(self, target_file: str) -> str:
        return "python -m pytest -q"


# --------------------------------------------------------------------------- #
# Prompt templates (model-agnostic, lean — Constraint A)
# --------------------------------------------------------------------------- #
_GENERATION_PROMPT = """You are an expert {language} engineer. Write {language} code that satisfies the objective and passes its tests.
Output ONLY one fenced ```{language} code block containing the COMPLETE contents of {target_file}. No prose, no explanation.

<objective>
{objective}
</objective>

<current_file path="{target_file}">
{code}
</current_file>{failure}"""

_FAILURE_BLOCK = """

<test_failure>
Your previous attempt did NOT pass its tests. Read the output below, find the cause, and rewrite the COMPLETE {target_file} to fix it.
{output}
</test_failure>"""


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #
class MicroPromptChain:
    """Generate -> test -> repair, until the tests pass or a bound trips.

    :param model: model name/config for the default generator (a single-instruction
        ``PromptChain``). Optional if ``generate=`` is supplied.
    :param generate: an async ``(prompt) -> text`` callable to override the default
        generator (used by tests, or to plug a custom chain).
    :param language: source language (selects the fenced-block tag + prompt wording).
    :param image: Docker image for the sandbox (ignored when a ``runner`` is passed).
    :param install_command: shell command to install test deps (run inside the
        sandbox each iteration, since containers are ephemeral; ``None`` to skip).
    :param max_iterations: hard iteration cap (the always-on guard).
    :param max_seconds: optional wall-clock budget.
    :param network: Docker ``--network`` (default ``"none"``; set ``"bridge"`` if
        ``install_command`` needs to reach the network).
    :param use_docker: prefer :class:`DockerExecutor`; falls back to
        :class:`LocalExecutor` (with a warning) when no daemon is reachable.
    :param executor: inject a ready executor (``work_dir``/``write_file``/``run``);
        when given it is used as-is and not closed by this loop.
    """

    def __init__(self, model=None, *, generate: Optional[GenerateFn] = None,
                 language: str = "python", image: str = "python:3.12-slim",
                 install_command: Optional[str] = None, max_iterations: int = 10,
                 max_seconds: Optional[float] = None, network: Optional[str] = "none",
                 timeout: int = 120, use_docker: bool = True, executor=None,
                 verbose: bool = False) -> None:
        if model is None and generate is None:
            raise ValueError("provide model= (for the default generator) or generate= (a callable)")
        self.model = model
        self._generate = generate
        self.language = language
        self.image = image
        self.install_command = install_command
        self.max_iterations = max_iterations
        self.max_seconds = max_seconds
        self.network = network
        self.timeout = timeout
        self.use_docker = use_docker
        self._executor = executor
        self.verbose = verbose
        self._chain = None  # lazily built default generator

    # -- generation -------------------------------------------------------- #
    async def _generate_impl(self, prompt: str) -> str:
        if self._generate is not None:
            return await self._generate(prompt)
        if self._chain is None:
            self._chain = make_generator(self.model)
        return await self._chain(prompt)

    def _build_prompt(self, objective: str, target_file: str, code: str, failure: str) -> str:
        failure_section = ""
        if failure:
            failure_section = _FAILURE_BLOCK.format(target_file=target_file, output=failure)
        return _GENERATION_PROMPT.format(
            language=self.language, objective=objective, target_file=target_file,
            code=code or "(empty — write it from scratch)", failure=failure_section,
        )

    def _extract(self, raw: str) -> str:
        return extract_code(raw, self.language)

    @staticmethod
    def _truncate(text: str) -> str:
        return truncate_tail(text)

    # -- executor ---------------------------------------------------------- #
    def _make_executor(self):
        """Return ``(executor, owns)``; ``owns`` is True when this loop must close it."""
        return resolve_executor(self._executor, self.use_docker, self.image,
                                self.network, self.timeout)

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[MicroPromptChain] {msg}")

    # -- the run ----------------------------------------------------------- #
    async def run(self, objective: str, target_file: str, *,
                  test_command: Optional[str] = None, deps: Optional[dict] = None,
                  install_command: Optional[str] = None, starting_code: str = "",
                  runner=None) -> LoopResult:
        """Run the loop. The model writes ``target_file``; ``test_command`` is run in
        the sandbox each iteration; exit 0 = pass (loop stops with the winning code).

        :param deps: ``{relpath: content}`` extra files (tests, fixtures) staged once.
        :param runner: optional :class:`PytestRunner`-like object supplying
            ``image`` / ``install_command`` / ``test_command(target_file)`` defaults.
        :returns: a :class:`LoopResult`.
        """
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

        executor, owns = self._make_executor()
        try:
            for path, content in (deps or {}).items():
                executor.write_file(path, content)

            state: dict = {"code": starting_code or "", "failure": "",
                           "winning_code": None, "result": "FAIL", "attempts": []}

            async def step(it: int, st: dict) -> bool:
                prompt = self._build_prompt(objective, target_file, st["code"], st["failure"])
                raw = await self._generate_impl(prompt)
                code = self._extract(raw)
                if not code:
                    self._log(f"iter {it}: model returned no code")
                    st["result"] = "NO_CODE"
                    st["attempts"].append(Attempt(it, "", False, "model returned no code block"))
                    return True  # keep trying
                st["code"] = code
                executor.write_file(target_file, code)
                res = executor.run(full_test)
                passed = res.exit_code == 0 and not res.timed_out
                output = self._truncate(res.output)
                st["attempts"].append(Attempt(it, code, passed, output))
                self._log(f"iter {it}: {'PASS' if passed else 'fail'} (exit={res.exit_code})")
                if passed:
                    st["winning_code"] = code
                    st["result"] = "PASS"
                    return False  # stop
                st["result"] = "TIMEOUT" if res.timed_out else "FAIL"
                st["failure"] = output
                return True

            loop = ExternalLoop(max_iters=self.max_iterations, max_seconds=self.max_seconds)
            state = await loop.run(step, state)

            result = "PASS" if state["winning_code"] else state.get("result", "FAIL")
            return LoopResult(
                result=result,
                winning_code=state["winning_code"],
                iterations=state.get("_iterations", 0),
                stopped_by=state.get("_stopped", ""),
                attempts=state["attempts"],
            )
        finally:
            if owns and hasattr(executor, "stop"):
                executor.stop()

    def run_sync(self, *args, **kwargs) -> LoopResult:
        """Synchronous wrapper around :meth:`run` — safe even if an event loop is
        already running (see :func:`run_coro_blocking`)."""
        return run_coro_blocking(self.run(*args, **kwargs))
