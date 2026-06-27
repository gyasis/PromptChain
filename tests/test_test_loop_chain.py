"""Tests for MicroPromptChain (engine A — the lean iterate-until-tests-pass loop).

These run fully OFFLINE: the LLM is a fake ``generate`` callable, and the sandbox
is :class:`LocalExecutor` running real ``python -m pytest`` on the host. No Docker
daemon and no model API are required.
"""
import asyncio

from promptchain.utils.test_loop_chain import (
    Attempt,
    LocalExecutor,
    LoopResult,
    MicroPromptChain,
    PytestRunner,
)

# A tiny test file the generated solution must satisfy.
_TEST_FILE = (
    "from solution import multiply\n\n"
    "def test_mul():\n"
    "    assert multiply(2, 3) == 6\n"
    "    assert multiply(-1, 5) == -5\n"
)
_TEST_CMD = "python -m pytest -q test_solution.py"


def test_passes_after_repair():
    """First attempt is wrong (a+b), second is right (a*b) — loop threads the
    failure back and stops the moment the tests go green."""
    calls = {"n": 0}

    async def fake_gen(prompt: str) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            assert "<test_failure>" not in prompt  # first pass: no failure threaded
            return "```python\ndef multiply(a, b):\n    return a + b\n```"
        assert "<test_failure>" in prompt          # repair pass sees the failure
        return "```python\ndef multiply(a, b):\n    return a * b\n```"

    loop = MicroPromptChain(generate=fake_gen, use_docker=False, max_iterations=5)
    res = loop.run_sync(
        objective="implement multiply(a, b)",
        target_file="solution.py",
        test_command=_TEST_CMD,
        deps={"test_solution.py": _TEST_FILE},
    )

    assert isinstance(res, LoopResult)
    assert res.result == "PASS"
    assert bool(res) is True
    assert "return a * b" in res.winning_code
    assert res.iterations == 2
    assert calls["n"] == 2
    assert [a.passed for a in res.attempts] == [False, True]


def test_bounds_on_persistent_failure():
    """A generator that never gets it right stops at the iteration guard, FAIL."""

    async def always_wrong(prompt: str) -> str:
        return "```python\ndef multiply(a, b):\n    return a + b\n```"

    loop = MicroPromptChain(generate=always_wrong, use_docker=False, max_iterations=3)
    res = loop.run_sync(
        objective="implement multiply",
        target_file="solution.py",
        test_command=_TEST_CMD,
        deps={"test_solution.py": _TEST_FILE},
    )

    assert res.result == "FAIL"
    assert res.iterations == 3
    assert res.stopped_by == "max_iters=3"
    assert res.winning_code is None
    assert len(res.attempts) == 3


def test_passes_first_try():
    """Correct on the first shot — exactly one iteration, no failure threading."""

    async def correct(prompt: str) -> str:
        return "```python\ndef multiply(a, b):\n    return a * b\n```"

    loop = MicroPromptChain(generate=correct, use_docker=False)
    res = loop.run_sync(
        objective="implement multiply",
        target_file="solution.py",
        test_command=_TEST_CMD,
        deps={"test_solution.py": _TEST_FILE},
    )

    assert res.result == "PASS"
    assert res.iterations == 1


def test_no_code_block_is_retried_not_crashed():
    """Prose-only output yields NO_CODE and is retried (self-correcting), then passes."""
    calls = {"n": 0}

    async def flaky(prompt: str) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            return "I think the answer is to multiply them."  # no fenced block -> still treated as code, fails
        return "```python\ndef multiply(a, b):\n    return a * b\n```"

    loop = MicroPromptChain(generate=flaky, use_docker=False, max_iterations=5)
    res = loop.run_sync(
        objective="implement multiply",
        target_file="solution.py",
        test_command=_TEST_CMD,
        deps={"test_solution.py": _TEST_FILE},
    )

    assert res.result == "PASS"
    assert res.iterations == 2


def test_runner_supplies_defaults():
    """PytestRunner fills test_command when none is passed explicitly."""
    captured = {}

    async def correct(prompt: str) -> str:
        return "```python\ndef multiply(a, b):\n    return a * b\n```"

    runner = PytestRunner(install_command=None)  # no install (pytest already on host)
    assert runner.test_command("solution.py") == "python -m pytest -q"

    # place the test under the default pytest discovery name so "pytest -q" finds it
    loop = MicroPromptChain(generate=correct, use_docker=False)
    res = loop.run_sync(
        objective="implement multiply",
        target_file="solution.py",
        deps={"test_solution.py": _TEST_FILE},
        runner=runner,
    )
    assert res.result == "PASS"


def test_requires_model_or_generate():
    try:
        MicroPromptChain()
    except ValueError as e:
        assert "generate=" in str(e)
    else:
        raise AssertionError("expected ValueError when neither model nor generate is given")


def test_local_executor_confines_writes(tmp_path):
    ex = LocalExecutor(work_dir=str(tmp_path))
    ex.write_file("a/b.py", "x = 1\n")
    assert (tmp_path / "a" / "b.py").read_text() == "x = 1\n"
    try:
        ex.write_file("../escape.py", "nope")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on path escape")
