---
id: recipe-autoresearch
title: Research, build until verified, critique (AutoResearch)
when: A brief should become runner-verified code, optionally with a research stage first and a critic verdict after.
api_version: 0.10.0+
---

# AutoResearch

`AutoResearch` wraps `MicroPromptChain` (the build loop) with two optional stages:
`research(brief) -> notes` before, `critique(brief, build) -> verdict` after. The loop accepts a build
only if your test command exits 0. The model cannot just say it succeeded. It can still make the
command exit 0 (see "Common mistakes"), so keep the tests out of its reach.

## Which loop?

| Need | Use |
|---|---|
| Build until a test passes, with research notes and/or a critic gate | `AutoResearch` |
| Build until a test passes, nothing else | `MicroPromptChain` (`recipe-test-loop`) |
| Staged librarian, artisan, critic every pass | `RalphChain` (`recipe-test-loop`) |
| Register the whole thing as a tool | `AutoResearch.run` + `ResearchResult.to_dict()` (below) |

## Offline example

```python
import json
from promptchain.utils.autoresearch import AutoResearch, auto_research

TESTS = (
    "from solution import slugify\n\n"
    "def test_slug():\n"
    "    assert slugify('Hello, World!') == 'hello-world'\n"
)
FENCE = "`" * 3

async def fake_generate(prompt: str) -> str:
    # Replace with model="..." in production. The notes reach the prompt via the objective.
    assert "Research notes" in prompt
    body = (
        "import re\n"
        "def slugify(s):\n"
        "    return re.sub(r'[^a-z0-9]+', '-', s.lower()).strip('-')\n"
    )
    return f"{FENCE}python\n{body}{FENCE}"

async def research(brief: str) -> str:
    return "Slugs are lowercase, ASCII, with runs of other characters collapsed to one dash."

async def critique(brief: str, build) -> str:
    # build is a LoopResult. Return a verdict string; see "Verdict rule" below.
    return "APPROVE" if build.result == "PASS" else "REJECT"

ar = AutoResearch(
    generate=fake_generate, research=research, critique=critique,
    use_docker=False, max_iterations=3,
)
res = ar.run_sync(
    "Write slugify(s) that lowercases and replaces non-alphanumeric runs with '-'.",
    test_command="python -m pytest -q test_solution.py",
    deps={"test_solution.py": TESTS},
    constraints=["standard library only"],
)
print(res.verified, res.verdict, res.build.result, res.build.iterations)   # True APPROVE PASS 1
print(json.dumps(res.to_dict())[:80])

# One-shot function: constructor kwargs and run kwargs share one call.
import asyncio

async def main():
    r = await auto_research(
        "Write slugify(s).",
        generate=fake_generate, research=research,
        use_docker=False, max_iterations=3,            # constructor kwargs
        test_command="python -m pytest -q test_solution.py",   # run kwargs
        deps={"test_solution.py": TESTS},
    )
    print(r.verified, r.verdict)                       # True skipped

asyncio.run(main())
```

## Contract

| Item | Value |
|---|---|
| ctor | `AutoResearch(model=None, *, generate=None, research=None, critique=None, language="python", image="python:3.12-slim", install_command=None, max_iterations=10, max_seconds=None, use_docker=True, executor=None, verbose=False)` |
| Not in the ctor | `network` and `timeout`. They exist on `MicroPromptChain` but `AutoResearch` does not forward them, so the Docker network is the `MicroPromptChain` default (`"none"`). Inject `executor=DockerExecutor(...)` to change it. |
| `run(brief, *, target_file="solution.py", test_command=None, deps=None, install_command=None, constraints=None, starting_code="", runner=None)` | Everything after `brief` is keyword-only. You must give `test_command` or `runner`, else `ValueError`. |
| `run_sync(...)` | Same args; safe inside a running event loop. |
| `research` | `async (brief) -> str`. Notes are cut to 4000 chars and added to the objective. |
| `critique` | `async (brief, LoopResult) -> str`. Runs even when the build failed. |
| `auto_research(brief, *, model, generate, research, critique, **kwargs)` | Constructor kwargs (`language`, `image`, `install_command`, `max_iterations`, `max_seconds`, `use_docker`, `executor`, `verbose`) are split out; the rest go to `run`. |
| Returns `ResearchResult` | `.verified`, `.verdict`, `.build` (`LoopResult`), `.notes`, `.brief`, `.winning_code`, `.attempts`. `bool(res)` is `res.verified`. |
| `res.to_dict()` | `{verified, result, verdict, iterations, stopped_by, winning_code, notes_chars}`. JSON-serialisable. |

`verified` is true only when the build result is `PASS` **and** (no critique given, or the
verdict approves). With no `critique`, `verdict` is `"skipped"`.

## Verdict rule

AutoResearch checks the critic verdict by substring on the upper-cased text:

1. Contains `REJECT` or `FAIL` anywhere: not approved.
2. Otherwise contains `APPROVE`, `PASS` or `ACCEPT`: approved.
3. Anything else: not approved.

False rejects: `"APPROVE, no failures found"` (it contains `FAIL`), or any reply that quotes
`failed`. False approves: `"DO NOT APPROVE"`, `"Not approved"` and `"Unacceptable: ..."` all
contain `APPROVE` or `ACCEPT` and none of `REJECT` / `FAIL`. Make the critic reply with exactly
one word, `APPROVE` or `REJECT`, and normalise it yourself in the `critique` function before
returning it:

```python
async def critique(brief, build):
    raw = await ask_model(...)                  # your model call
    return "APPROVE" if raw.strip().upper() == "APPROVE" else "REJECT"
```

A build that passed its tests but drew a critic rejection gives `verified=False`
while `winning_code` is still set.

## As a tool

Tools must return strings, so return `json.dumps(...)`. `AutoResearch` takes no `network` or
`timeout`, so inject the executor to control them:

```python
import json
from promptchain.utils.autoresearch import AutoResearch
from promptchain.utils.docker_executor import DockerExecutor

async def build_tool(brief: str, tests: str) -> str:
    # The image must already contain pytest and be pulled locally (docker pull ...).
    # network="none" is fine then: nothing is installed at run time.
    executor = DockerExecutor(image="my-python-pytest:latest", network="none",
                              timeout=120, memory="512m", pids_limit=256)
    ar = AutoResearch(model="openai/gpt-4o-mini", executor=executor, max_iterations=5)
    res = await ar.run(brief, test_command="python -m pytest -q test_solution.py",
                       deps={"test_solution.py": tests})
    return json.dumps(res.to_dict())
```

With Docker unavailable and `use_docker=True` (and no injected executor), the build falls back to
unsandboxed host execution with a `RuntimeWarning`, shown once per process. An injected
`DockerExecutor` never falls back. See `recipe-test-loop` for the safety notes.

Cost: one build call per iteration, plus one `research` call and one `critique` call. A run makes
up to `max_iterations` plus 2 model calls.

## Common mistakes

- **No test command.** `run` raises `ValueError` without `test_command` or `runner`. The test is the success bar; the loop cannot propose one.
- **`AutoResearch(network="bridge")`.** `TypeError`. The kwarg is not in the constructor. Through `auto_research(...)` it lands in `run` and also raises `TypeError`.
- **Positional args after `brief`.** `ar.run("brief", "solution.py")` raises `TypeError`; `target_file` is keyword-only here.
- **A chatty critic.** See "Verdict rule": one `FAIL` substring rejects the build.
- **Returning the `ResearchResult` from a tool.** Return `json.dumps(res.to_dict())`.
- **Trusting the exit code alone.** Generated code can call `os._exit(0)` before any assert runs, or rewrite the test file in the shared work dir. Print and check a pass count in the test command, or keep the tests unreachable.
- **`run_sync` in async code.** It blocks the calling thread and the running event loop until the run ends. Use `await ar.run(...)`.
- **Skipping `docker pull`.** A first pull inside the loop eats the executor `timeout`.
- **Calling `run` without `await`.** It is a coroutine. Use `await` or `run_sync`.
- **Large research notes.** The loop cuts notes past 4000 chars and adds a marker. Summarise in the `research` function.
