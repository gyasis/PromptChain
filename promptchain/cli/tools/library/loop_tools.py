"""TUI loop tools — expose the iterate-until-tests-pass engines as agent tools.

Registers two tools with the CLI tool registry so a TUI agent can FORGE verified
code (not just describe how): write code in a sandbox and iterate until the given
test command passes.

  * ``build_until_tests_pass``  -> MicroPromptChain (engine A, single agent)
  * ``multi_agent_build``       -> RalphChain (engine B, librarian/artisan/critic,
                                   fresh context each iteration, per-role models)

Both are ``async def`` so PromptChain's tool path AWAITS them natively (see
``agentic_step_processor`` / ``promptchaining`` coroutine-tool handling) — they
never call ``asyncio.run`` from inside the running loop, so the
running-event-loop footgun cannot occur here. (The loops' synchronous
``run_sync`` is independently guarded by ``run_coro_blocking``.)

Model resolution: an explicit ``model`` arg wins; else ``PROMPTCHAIN_LOOP_MODEL``;
else the tool returns a clear error (no silent wrong default). The success bar is
the caller-supplied ``test_command`` + ``deps`` (the tests/fixtures to stage) — a
self-contained forge that returns proven code the agent can then write to disk.
"""
import json
import os
from typing import Optional

from promptchain.cli.tools import ToolCategory, registry
from promptchain.utils.ralph_chain import RalphChain
from promptchain.utils.test_loop_chain import MicroPromptChain

_CODE_CAP = 20000  # cap returned code so a huge file can't blow the tool result


def _resolve_model(explicit: Optional[str]):
    model = explicit or os.environ.get("PROMPTCHAIN_LOOP_MODEL")
    if not model:
        return None, json.dumps({
            "error": "no model configured for the loop. Pass model=\"<provider/model>\" "
                     "or set the PROMPTCHAIN_LOOP_MODEL environment variable."})
    return model, None


def _cap(code: Optional[str]):
    if code and len(code) > _CODE_CAP:
        return code[:_CODE_CAP] + "\n# ...[truncated]..."
    return code


_BUILD_DESC = (
    "BUILD-UNTIL-TESTS-PASS: write code in a sandbox and iterate until a test command "
    "passes (single-agent loop). USE WHEN you need a PROVEN, runner-verified piece of "
    "code — supply the objective and a test (via deps + test_command); get back code "
    "whose tests actually pass.\n\n"
    "The loop generates code, runs `test_command` in a sandbox, and on failure feeds the "
    "test output back and retries (bounded by max_iterations). Exit code 0 = pass. "
    "Language-agnostic: set `image` + `test_command` for python/rust/go/node. "
    "Returns JSON {result, verified, iterations, stopped_by, winning_code}. "
    "`verified=true` means the tests genuinely passed — write winning_code to disk yourself."
)


@registry.register(
    category=ToolCategory.UTILITY,
    description=_BUILD_DESC,
    parameters={
        "objective": {"type": "string", "required": True,
                      "description": "What the code must do (the build goal)."},
        "test_command": {"type": "string", "required": True,
                         "description": "Shell command run in the sandbox; exit 0 = pass "
                                        "(e.g. 'python -m pytest -q test_solution.py')."},
        "target_file": {"type": "string", "required": False,
                        "description": "File the loop writes (default 'solution.py')."},
        "deps": {"type": "object", "required": False,
                 "description": "Map {filename: content} of tests/fixtures to stage in the "
                                "sandbox (e.g. the test file the test_command runs)."},
        "language": {"type": "string", "required": False,
                     "description": "Source language (default 'python')."},
        "max_iterations": {"type": "integer", "required": False,
                           "description": "Max generate→test→repair cycles (default 8)."},
        "model": {"type": "string", "required": False,
                  "description": "Model for generation (default: $PROMPTCHAIN_LOOP_MODEL)."},
        "use_docker": {"type": "boolean", "required": False,
                       "description": "Sandbox in Docker (default true; falls back to host if "
                                      "no daemon)."},
    },
    tags=["loop", "codegen", "test", "verify", "build", "forge"],
)
async def build_until_tests_pass(objective: str, test_command: str,
                                 target_file: str = "solution.py", deps: Optional[dict] = None,
                                 language: str = "python", max_iterations: int = 8,
                                 model: Optional[str] = None, use_docker: bool = True) -> str:
    m, err = _resolve_model(model)
    if err:
        return err
    loop = MicroPromptChain(model=m, language=language, max_iterations=max_iterations,
                            use_docker=use_docker)
    res = await loop.run(objective=objective, target_file=target_file,
                         test_command=test_command, deps=deps or {})
    return json.dumps({
        "result": res.result, "verified": res.result == "PASS",
        "iterations": res.iterations, "stopped_by": res.stopped_by,
        "winning_code": _cap(res.winning_code),
    })


_RALPH_DESC = (
    "MULTI-AGENT BUILD (MA loop): like build_until_tests_pass but uses a staged pipeline — "
    "librarian (gather context) → artisan (write code) → critic (review) → testing — with "
    "FRESH context each iteration (only the test result threads forward). USE WHEN the task "
    "is harder and benefits from the structured pipeline, or you want a different model per "
    "role. Optional per-role models (librarian cheap, artisan strong, critic medium). "
    "Returns JSON {result, verified, iterations, stopped_by, winning_code}."
)


@registry.register(
    category=ToolCategory.UTILITY,
    description=_RALPH_DESC,
    parameters={
        "objective": {"type": "string", "required": True, "description": "The build goal."},
        "test_command": {"type": "string", "required": True,
                         "description": "Shell command; exit 0 = pass."},
        "target_file": {"type": "string", "required": False,
                        "description": "File the loop writes (default 'solution.py')."},
        "deps": {"type": "object", "required": False,
                 "description": "Map {filename: content} of tests/fixtures to stage."},
        "language": {"type": "string", "required": False, "description": "Default 'python'."},
        "max_iterations": {"type": "integer", "required": False, "description": "Default 8."},
        "model": {"type": "string", "required": False,
                  "description": "Shared model for all roles (default $PROMPTCHAIN_LOOP_MODEL)."},
        "librarian_model": {"type": "string", "required": False,
                            "description": "Override model for the librarian (context) stage."},
        "artisan_model": {"type": "string", "required": False,
                          "description": "Override model for the artisan (codegen) stage."},
        "critic_model": {"type": "string", "required": False,
                         "description": "Override model for the critic (review) stage."},
        "use_docker": {"type": "boolean", "required": False, "description": "Default true."},
    },
    tags=["loop", "codegen", "test", "verify", "multi-agent", "ralph", "forge"],
)
async def multi_agent_build(objective: str, test_command: str,
                            target_file: str = "solution.py", deps: Optional[dict] = None,
                            language: str = "python", max_iterations: int = 8,
                            model: Optional[str] = None, librarian_model: Optional[str] = None,
                            artisan_model: Optional[str] = None, critic_model: Optional[str] = None,
                            use_docker: bool = True) -> str:
    # need at least a shared model OR all per-role models
    if not (model or os.environ.get("PROMPTCHAIN_LOOP_MODEL")) and not (
            librarian_model and artisan_model and critic_model):
        return json.dumps({
            "error": "no model configured. Pass model= (shared) or librarian_model/"
                     "artisan_model/critic_model, or set PROMPTCHAIN_LOOP_MODEL."})
    shared = model or os.environ.get("PROMPTCHAIN_LOOP_MODEL")
    loop = RalphChain(model=shared, librarian_model=librarian_model,
                      artisan_model=artisan_model, critic_model=critic_model,
                      language=language, max_iterations=max_iterations, use_docker=use_docker)
    res = await loop.run(objective=objective, target_file=target_file,
                         test_command=test_command, deps=deps or {})
    return json.dumps({
        "result": res.result, "verified": res.result == "PASS",
        "iterations": res.iterations, "stopped_by": res.stopped_by,
        "winning_code": _cap(res.winning_code),
    })
