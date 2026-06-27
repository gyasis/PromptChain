"""Tests for AutoResearch (the research -> build-until-verified -> critique tool).

Offline: fake research + fake generate (LLM) + LocalExecutor running real pytest.
"""
from promptchain.utils.autoresearch import AutoResearch, ResearchResult, auto_research

_TEST_FILE = (
    "from solution import multiply\n\n"
    "def test_mul():\n"
    "    assert multiply(2, 3) == 6\n"
)
_TEST_CMD = "python -m pytest -q test_solution.py"


def test_research_then_verified_build():
    """Research notes are folded into the objective; build repairs to PASS; verified."""
    seen = {}

    async def research(brief: str) -> str:
        seen["brief"] = brief
        return "NOTE: multiply means a*b, not a+b."

    calls = {"n": 0}

    async def gen(prompt: str) -> str:
        calls["n"] += 1
        # the research notes must be visible to the builder
        assert "NOTE: multiply means a*b" in prompt
        if calls["n"] == 1:
            return "```python\ndef multiply(a, b):\n    return a + b\n```"
        return "```python\ndef multiply(a, b):\n    return a * b\n```"

    ar = AutoResearch(generate=gen, research=research, use_docker=False, max_iterations=5)
    res = ar.run_sync("build a multiply(a, b)", target_file="solution.py",
                      test_command=_TEST_CMD, deps={"test_solution.py": _TEST_FILE})

    assert isinstance(res, ResearchResult)
    assert res.verified is True
    assert bool(res) is True
    assert res.build.result == "PASS"
    assert "return a * b" in res.winning_code
    assert res.verdict == "skipped"          # no critic supplied
    assert seen["brief"] == "build a multiply(a, b)"


def test_critic_can_reject_a_passing_build():
    """Even a PASSing build is not 'verified' if the critic rejects it."""

    async def gen(prompt: str) -> str:
        return "```python\ndef multiply(a, b):\n    return a * b\n```"

    async def critic(brief: str, build) -> str:
        assert build.result == "PASS"
        return "REJECT: style violation"

    ar = AutoResearch(generate=gen, critique=critic, use_docker=False)
    res = ar.run_sync("build multiply", target_file="solution.py",
                      test_command=_TEST_CMD, deps={"test_solution.py": _TEST_FILE})

    assert res.build.result == "PASS"
    assert res.verified is False
    assert "REJECT" in res.verdict


def test_critic_approves():
    async def gen(prompt: str) -> str:
        return "```python\ndef multiply(a, b):\n    return a * b\n```"

    async def critic(brief: str, build) -> str:
        return "APPROVE"

    ar = AutoResearch(generate=gen, critique=critic, use_docker=False)
    res = ar.run_sync("build multiply", target_file="solution.py",
                      test_command=_TEST_CMD, deps={"test_solution.py": _TEST_FILE})
    assert res.verified is True
    assert res.to_dict()["verified"] is True
    assert res.to_dict()["result"] == "PASS"


def test_no_research_stage_is_fine():
    async def gen(prompt: str) -> str:
        assert "Research notes" not in prompt  # no research => no notes block
        return "```python\ndef multiply(a, b):\n    return a * b\n```"

    ar = AutoResearch(generate=gen, use_docker=False)
    res = ar.run_sync("build multiply", target_file="solution.py",
                      test_command=_TEST_CMD, deps={"test_solution.py": _TEST_FILE})
    assert res.verified is True
    assert res.notes == ""


def test_one_shot_auto_research_helper():
    import asyncio

    async def gen(prompt: str) -> str:
        return "```python\ndef multiply(a, b):\n    return a * b\n```"

    res = asyncio.run(auto_research(
        "build multiply", generate=gen, use_docker=False,
        target_file="solution.py", test_command=_TEST_CMD,
        deps={"test_solution.py": _TEST_FILE}))
    assert res.verified is True
