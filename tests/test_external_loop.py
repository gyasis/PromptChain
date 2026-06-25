"""Tests for promptchain.utils.external_loop.ExternalLoop (issue #8).

No LLM / API key needed — the step is a plain async callable, so loop semantics
(guard, breakers, exhaustion) are verified deterministically. Sync wrappers via
asyncio.run() so no pytest-asyncio plugin is required.
"""

import asyncio

import pytest

from promptchain.utils.external_loop import ExternalLoop, over_worklist


def _run(coro):
    return asyncio.run(coro)


def test_exhaustion_natural_stop():
    calls = []

    async def step(it, st):
        calls.append(it)
        return it < 3  # keep going for it=1,2 ; returns False at it=3

    st = _run(ExternalLoop(max_iters=100).run(step))
    assert calls == [1, 2, 3]
    assert st["_stopped"] == "exhausted"
    assert st["_iterations"] == 3


def test_max_iters_guard_always_on():
    async def step(it, st):
        return True  # never naturally stops

    st = _run(ExternalLoop(max_iters=5).run(step))
    assert st["_iterations"] == 5
    assert st["_stopped"] == "max_iters=5"


def test_max_seconds_guard():
    async def step(it, st):
        await asyncio.sleep(0.02)
        return True

    st = _run(ExternalLoop(max_iters=10_000, max_seconds=0.05).run(step))
    assert st["_stopped"].startswith("max_seconds=")
    assert st["_iterations"] >= 1


def test_custom_breaker_fires_after_guard():
    async def step(it, st):
        st["seen"] = st.get("seen", 0) + 1
        return True

    breaker = lambda it, st: (st.get("seen", 0) >= 3, "seen-3")
    st = _run(ExternalLoop(max_iters=100, breakers=(breaker,)).run(step))
    assert st["_stopped"] == "seen-3"
    assert st["seen"] == 3


def test_required_guard_cannot_be_disabled():
    with pytest.raises(ValueError):
        ExternalLoop(max_iters=0)
    with pytest.raises(ValueError):
        ExternalLoop(max_seconds=0)


def test_over_worklist_collects_results():
    async def handler(item, st):
        st["results"][item] = item * 10

    st = _run(over_worklist([1, 2, 3], handler, max_iters=100))
    assert st["results"] == {1: 10, 2: 20, 3: 30}
    assert st["_stopped"] == "exhausted"


def test_over_worklist_respects_iter_guard():
    async def handler(item, st):
        st["results"][item] = True

    # 10 items but max_iters=4 → only 4 processed before the guard fires
    st = _run(over_worklist(list(range(10)), handler, max_iters=4))
    assert st["_stopped"] == "max_iters=4"
    assert len(st["results"]) == 4


def test_run_sync_wrapper():
    async def step(it, st):
        return it < 2

    st = ExternalLoop(max_iters=10).run_sync(step)
    assert st["_iterations"] == 2
    assert st["_stopped"] == "exhausted"
