---
id: recipe-test-loop
title: Generate code until the tests pass (MicroPromptChain, RalphChain)
when: A model must write a file and a test command (exit code 0 = pass) is the judge; you want bounded retries with the failure fed back.
api_version: 0.10.0+
---

# Test loops: MicroPromptChain and RalphChain

Both loops write `target_file` from model output and run your test command in an executor
(a Docker container when available, otherwise the host). They stop when it exits 0. The exit code is the
only judge. Neither needs a tool-calling model.

## Which loop?

| Need | Use |
|---|---|
| One generator, test output threaded back each pass | `MicroPromptChain` |
| Staged librarian, artisan, critic per pass; fresh context each pass; different model per role; stagnation stop | `RalphChain` |
| Research notes first, critic verdict after | `AutoResearch` (`recipe-autoresearch`) |
| Repeat any step with no tests | `ExternalLoop` (`recipe-external-loop`) |

Start with `MicroPromptChain`. Use `RalphChain` when a weak or local model needs a clean,
small prompt on every pass.

## Safety: the sandbox

`use_docker=True` is the default. If no Docker daemon is reachable, the loop **falls back to
`LocalExecutor`, which runs on the host with no sandbox**, and emits a `RuntimeWarning`
("Docker unavailable ... UNSANDBOXED"). Model-written code then runs as your user.

- The warning fires on every fallback. To avoid the fallback, check `DockerExecutor.available()`
  yourself, or pass `executor=DockerExecutor(...)`, which never falls back.
- For untrusted model output, inject the executor and treat the warning as an error
  (`python -W error::RuntimeWarning`).
- `use_docker=False` is the explicit opt-in to host execution. Use it only for trusted code, CI and tests.
- `network="none"` is the default for Docker. It blocks `install_command` (for example `pip install`).
  Set `network="bridge"` if the install needs the network, or bake dependencies into `image`.
- `PytestRunner` under Docker needs both: it selects `python:3.12-slim` (no pytest) and runs
  `pip install -q pytest`, which fails under `network="none"`. Pass `network="bridge"`, or skip the
  runner and use `test_command=` with your own `image=` that has pytest.
- `resolve_executor` builds `DockerExecutor(image, network, timeout)` only. The container runs as
  root with no memory, CPU or process limits. Inject `executor=DockerExecutor(..., memory="512m",
  pids_limit=256, user="1000:1000", read_only=True)` when that matters.
- On a Docker `timeout` the executor kills the named container (`pc-exec-...`) and returns
  `TIMEOUT` with the partial output. `LocalExecutor` runs the command in its own process group
  and kills the whole group on `timeout`, so the shell's children do not outlive it.
- The work dir is a read-write mount. Generated code can overwrite `deps` (your tests). It can also
  force a pass (`os._exit(0)` exits 0 before any assert runs). To catch that, check the pass count in the
  test output (for example `3 passed`). The other option is tests the code cannot reach.
- `executor=` injects a ready executor. The loop uses it as is and does not close it.
- `docker pull` the image first. A first-time pull inside the loop counts against `timeout`.

## Cost and bounds

- Model calls: at most `max_iterations` for `MicroPromptChain`; at most 3 times that for
  `RalphChain` (librarian, artisan, critic). A `NO_CODE` pass still counts toward the limit.
- The loop checks `max_seconds` between passes. It does not interrupt a running model call or test run.
- `run_sync` blocks the calling thread, and inside a running event loop it blocks that loop
  until the run ends. Prefer `await chain.run(...)` in async code.
- `stopped_by` is not a reliable verdict: a pass on the last allowed iteration still reports
  `max_iters=N`. Read `res.result` or `bool(res)`.

## MicroPromptChain, offline

```python
from promptchain.utils.test_loop_chain import MicroPromptChain

TESTS = (
    "from solution import multiply\n\n"
    "def test_mul():\n"
    "    assert multiply(2, 3) == 6\n"
)

FENCE = "`" * 3                               # a fenced block, as a model would reply

def code(expr: str) -> str:
    return f"{FENCE}python\ndef multiply(a, b):\n    return {expr}\n{FENCE}"

calls = {"n": 0}

async def fake_generate(prompt: str) -> str:
    # Replace with a model in production. Here: wrong first, right second.
    calls["n"] += 1
    if calls["n"] == 1:
        return code("a + b")
    assert "<test_failure>" in prompt          # the failure output is threaded back
    return code("a * b")

loop = MicroPromptChain(generate=fake_generate, use_docker=False, max_iterations=5)
res = loop.run_sync(
    "implement multiply(a, b)",                # objective
    "solution.py",                             # target_file: required, 2nd positional
    test_command="python -m pytest -q test_solution.py",
    deps={"test_solution.py": TESTS},          # extra files written once before the loop
)
print(res.result, res.iterations, res.stopped_by, bool(res))   # PASS 2 exhausted True
print(res.winning_code)
print([a.passed for a in res.attempts])                        # [False, True]
```

Real use replaces `generate=` with `model="ollama_chat/glm-5.3-flash:cloud"` (any model string PromptChain
accepts); the loop then builds a single-instruction `PromptChain` as the generator.

### MicroPromptChain contract

| Item | Value |
|---|---|
| ctor | `MicroPromptChain(model=None, *, generate=None, language="python", image="python:3.12-slim", install_command=None, max_iterations=10, max_seconds=None, network="none", timeout=120, use_docker=True, executor=None, verbose=False)`. Needs `model` or `generate`, else `ValueError`. |
| `generate` | `async (prompt: str) -> str`. The model output. The loop takes the fenced block of `language` from it. |
| `run(objective, target_file, *, test_command=None, deps=None, install_command=None, starting_code="", runner=None)` | `target_file` has no default here. You must give `test_command` or `runner`, else `ValueError`. |
| `run_sync(...)` | Same args. Safe inside a running event loop (runs on a worker thread). |
| `runner=PytestRunner()` | Supplies `image`, `install_command="pip install -q pytest"` and `test_command` (`python -m pytest -q`). It overwrites `self.image` and fills `install_command` / `test_command` only if you left them `None`. |
| Returns `LoopResult` | `.result` is `PASS`, `FAIL`, `TIMEOUT` or `NO_CODE`; `.winning_code`, `.iterations`, `.stopped_by`, `.attempts` (list of `Attempt(iteration, code, passed, output)`). `bool(res)` is `res.result == "PASS"`. |

`max_iterations` is the always-on guard. Reaching it with no pass returns `FAIL` (or the last
failure kind: `TIMEOUT`, `NO_CODE`) with `stopped_by == "max_iters=N"`.

## RalphChain, offline

```python
from promptchain.utils.ralph_chain import RalphChain

TESTS = (
    "from solution import multiply\n\n"
    "def test_mul():\n"
    "    assert multiply(2, 3) == 6\n"
)

FENCE = "`" * 3                               # a fenced block, as a model would reply

def code(expr: str) -> str:
    return f"{FENCE}python\ndef multiply(a, b):\n    return {expr}\n{FENCE}"

calls = {"n": 0}

async def librarian(prompt: str) -> str:
    return "Brief: implement multiply(a, b) returning the product."

async def artisan(prompt: str) -> str:
    calls["n"] += 1
    op = "+" if calls["n"] == 1 else "*"
    return code(f"a {op} b")

async def critic(prompt: str) -> str:
    return "APPROVE"                           # first line decides; it is advisory only

chain = RalphChain(                            # no model= needed: all three roles injected
    librarian=librarian, artisan=artisan, critic=critic,
    use_docker=False, max_iterations=5,
)
res = chain.run_sync(                          # target_file defaults to "solution.py"
    "implement multiply(a, b)",
    test_command="python -m pytest -q test_solution.py",   # test_command is a run() arg
    deps={"test_solution.py": TESTS},
)
print(res.result, res.iterations, [h.passed for h in res.history])   # PASS 2 [False, True]

# Stagnation: the same failure `entropy_threshold` times in a row stops the loop early.
async def always_wrong(prompt: str) -> str:
    return code("a + b")

stuck = RalphChain(
    librarian=librarian, artisan=always_wrong, run_critic=False,
    use_docker=False, max_iterations=10, entropy_threshold=3,
)
res = stuck.run_sync("implement multiply(a, b)",
                     test_command="python -m pytest -q test_solution.py",
                     deps={"test_solution.py": TESTS})
print(res.result, res.iterations, res.stopped_by)   # STAGNATED 3 stagnated(entropy=3)
```

### RalphChain contract

| Item | Value |
|---|---|
| ctor | `RalphChain(model=None, *, librarian_model=None, artisan_model=None, critic_model=None, librarian=None, artisan=None, critic=None, language=..., image=..., install_command=None, max_iterations=10, max_seconds=None, network="none", timeout=120, use_docker=True, executor=None, run_critic=True, entropy_threshold=3, verbose=False)` |
| Role resolution | Per role: the injected callable, else `<role>_model`, else the shared `model`. A role with none of them raises `ValueError`. With `run_critic=False` the critic is not needed. |
| `run(objective, target_file="solution.py", *, test_command=None, deps=None, install_command=None, starting_code="", runner=None)` | `test_command` belongs to `run`, not the constructor. Passing it to `RalphChain(...)` raises `TypeError`. |
| Returns `RalphResult` | `.result` is `PASS`, `FAIL`, `TIMEOUT`, `STAGNATED` or `NO_CODE`; `.winning_code`, `.iterations`, `.stopped_by`, `.history` (list of `RalphIteration(iteration, librarian, code, critic, passed, test_output)`). |
| Carried between passes | Only the test output, plus the critic notes when the critic did not approve, and the last code. Prompts are rebuilt each pass. |
| `entropy_threshold` | `k >= 1`: stop with `STAGNATED` when the last `k` failure signatures are identical. The signature comes from the pytest `E ` lines (else the output tail) after masking digits and hex. A shared banner does not make two different failures look alike. Output with no `E ` lines and the same tail still counts as a repeat. Set `entropy_threshold=0` to disable. `1` stops after the first failure. |

The critic is a pre-test review. It never blocks a test run; a failed run only gets its notes
appended to the failure text. The loop adds the notes unless the first line is a bare approval: it opens with `APPROVE` or
`APPROVED` (optionally after `VERDICT:`) and has no `NOT`, `NO`, `CANNOT`, `NEVER`, `n't` or
`REJECT`. `DO NOT APPROVE`, `Not approved` and `APPROVE, no issues` get their notes appended.

## Common mistakes

- **`RalphChain(..., test_command=...)`.** The constructor has no such kwarg. Pass it to `.run`.
- **`MicroPromptChain.run("goal")` without `target_file`.** There it has no default. `RalphChain.run` and `AutoResearch.run` default it to `"solution.py"`.
- **Leaving `use_docker=True` on a host with no Docker and ignoring the warning.** The code runs unsandboxed on the host.
- **`install_command="pip install ..."` with the default `network="none"`.** The install fails inside Docker. Use `network="bridge"` or a prebuilt image.
- **`test_command` that cannot see the target.** The loop writes files into the sandbox work dir; tests must import the module by the same name as `target_file` (`solution.py` -> `from solution import ...`).
- **A test directory with no tests.** `pytest` exits 5, so every iteration is `FAIL`.
- **Test command that always exits 0.** The loop reports `PASS` on the first pass. The exit code is the only signal.
- **Trusting `winning_code` when `result != "PASS"`.** It is `None` unless a pass occurred.
- **`generate` that returns prose with no fenced block.** The loop uses the whole output as code. A fully empty output gives `NO_CODE` and the loop keeps trying.
- **A sync `generate`.** The loop awaits it. Use `async def`.
