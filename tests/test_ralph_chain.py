"""Tests for RalphChain (engine B — the multi-agent fresh-context MA loop).

Offline: fake librarian/artisan/critic agents + LocalExecutor running real pytest.
Verifies the staged pipeline runs in order, the test result threads forward into a
FRESH iteration, per-role models/agents are honored, and the entropy breaker fires.
"""
from promptchain.utils.ralph_chain import RalphChain, RalphIteration, RalphResult

_TEST_FILE = (
    "from solution import multiply\n\n"
    "def test_mul():\n"
    "    assert multiply(2, 3) == 6\n"
)
_TEST_CMD = "python -m pytest -q test_solution.py"


def test_staged_pipeline_repairs_to_pass():
    """librarian -> artisan -> critic -> testing; first attempt wrong, second right.
    The artisan sees the librarian's brief; the 2nd iteration sees the threaded failure."""
    order = []

    async def librarian(prompt: str) -> str:
        order.append("librarian")
        return "BRIEF: implement multiply as a*b."

    calls = {"n": 0}

    async def artisan(prompt: str) -> str:
        order.append("artisan")
        assert "BRIEF: implement multiply as a*b." in prompt  # got librarian's brief
        calls["n"] += 1
        if calls["n"] == 1:
            assert "<last_attempt_failed>" not in prompt
            return "```python\ndef multiply(a, b):\n    return a + b\n```"
        assert "<last_attempt_failed>" in prompt              # fresh iter sees the failure
        return "```python\ndef multiply(a, b):\n    return a * b\n```"

    async def critic(prompt: str) -> str:
        order.append("critic")
        return "APPROVE"

    loop = RalphChain(librarian=librarian, artisan=artisan, critic=critic,
                      use_docker=False, max_iterations=5)
    res = loop.run_sync("build multiply(a, b)", target_file="solution.py",
                        test_command=_TEST_CMD, deps={"test_solution.py": _TEST_FILE})

    assert isinstance(res, RalphResult)
    assert res.result == "PASS"
    assert "return a * b" in res.winning_code
    assert res.iterations == 2
    # the staged order is preserved per iteration
    assert order[:4] == ["librarian", "artisan", "critic", "librarian"]
    assert all(isinstance(h, RalphIteration) for h in res.history)
    assert [h.passed for h in res.history] == [False, True]


def test_per_role_models_resolve_independently():
    """Different injected agents per role all run (the heterogeneous-model design)."""
    hits = {"lib": 0, "art": 0, "crit": 0}

    async def lib(p):
        hits["lib"] += 1
        return "brief"

    async def art(p):
        hits["art"] += 1
        return "```python\ndef multiply(a, b):\n    return a * b\n```"

    async def crit(p):
        hits["crit"] += 1
        return "APPROVE"

    loop = RalphChain(librarian=lib, artisan=art, critic=crit, use_docker=False)
    res = loop.run_sync("multiply", target_file="solution.py",
                        test_command=_TEST_CMD, deps={"test_solution.py": _TEST_FILE})
    assert res.result == "PASS"
    assert hits == {"lib": 1, "art": 1, "crit": 1}


def test_critic_can_be_disabled():
    async def lib(p):
        return "brief"

    async def art(p):
        return "```python\ndef multiply(a, b):\n    return a * b\n```"

    # run_critic=False => no critic model/agent required
    loop = RalphChain(librarian=lib, artisan=art, run_critic=False, use_docker=False)
    res = loop.run_sync("multiply", target_file="solution.py",
                        test_command=_TEST_CMD, deps={"test_solution.py": _TEST_FILE})
    assert res.result == "PASS"
    assert all(h.critic == "" for h in res.history)


def test_entropy_breaker_stops_on_repeated_failure():
    """A persistently-identical failure trips the entropy breaker before max_iterations."""

    async def lib(p):
        return "brief"

    async def art(p):
        return "```python\ndef multiply(a, b):\n    return a + b\n```"  # always wrong, same failure

    async def crit(p):
        return "looks off"

    loop = RalphChain(librarian=lib, artisan=art, critic=crit, use_docker=False,
                      max_iterations=10, entropy_threshold=3)
    res = loop.run_sync("multiply", target_file="solution.py",
                        test_command=_TEST_CMD, deps={"test_solution.py": _TEST_FILE})

    assert res.result == "STAGNATED"
    assert "stagnated(entropy=3)" in res.stopped_by
    assert res.iterations == 3          # stopped at the entropy threshold, not 10
    assert res.winning_code is None


def test_requires_model_or_agents():
    try:
        RalphChain()  # no shared model, no per-role agents
    except ValueError as e:
        assert "model" in str(e)
    else:
        raise AssertionError("expected ValueError when no model/agents provided")
