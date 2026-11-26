"""Unit tests for LightRAGSpeculativeExecutor.

The executor takes a ``lightrag_core`` and a ``SpeculativeConfig``. It records
queries, predicts likely repeats, runs the predictions against the backend and
holds the results in a TTL cache that callers read through ``check_cache``.
The backend is an ``AsyncMock``; no retrieval service is contacted.

Already covered by tests/test_speculative_executor.py (not repeated here):
initialisation, history prediction, cache hit/miss/expiry lookups, stats.
"""

from unittest.mock import AsyncMock, patch

import pytest

from promptchain.integrations.lightrag.speculative import (
    LightRAGSpeculativeExecutor,
    SpeculativeConfig,
    SpeculativeExecutionResult,
)
from promptchain.patterns.base import BasePattern


def make_executor(core, **config_kwargs):
    config_kwargs.setdefault("min_confidence", 0.5)
    config = SpeculativeConfig(**config_kwargs)
    with patch(
        "promptchain.integrations.lightrag.speculative.LIGHTRAG_AVAILABLE", True
    ):
        return LightRAGSpeculativeExecutor(core, config)


@pytest.fixture
def core():
    mock = AsyncMock()
    mock.local_query.return_value = {"result": "local"}
    mock.global_query.return_value = {"result": "global"}
    mock.hybrid_query.return_value = {"result": "hybrid"}
    return mock


@pytest.mark.asyncio
class TestPrefetchFlow:
    async def test_prefetched_query_is_served_from_cache(self, core):
        executor = make_executor(core)
        executor.record_call("What is Python?", "local")

        await executor.execute(context="python basics")
        core.local_query.assert_awaited_once_with("What is Python?")

        cached = executor.check_cache("What is Python?", "local")
        assert cached is not None
        assert cached.result == {"result": "local"}
        # Reading the cache does not reach the backend again.
        core.local_query.assert_awaited_once()

    async def test_unrelated_query_is_a_cache_miss(self, core):
        executor = make_executor(core)
        executor.record_call("What is Python?", "local")
        await executor.execute(context="python basics")

        assert executor.check_cache("something else", "local") is None
        assert executor.check_cache("What is Python?", "global") is None

    async def test_execute_prefetches_each_prediction_by_mode(self, core):
        executor = make_executor(core, max_concurrent=3)
        executor.record_call("alpha", "local")
        executor.record_call("beta", "global")
        executor.record_call("gamma", "hybrid")

        result = await executor.execute(context="ctx")

        assert result.success is True
        core.local_query.assert_awaited_once_with("alpha")
        core.global_query.assert_awaited_once_with("beta")
        core.hybrid_query.assert_awaited_once_with("gamma")
        assert {p.query for p in result.executed} == {"alpha", "beta", "gamma"}

    async def test_failed_prefetch_is_dropped_not_cached(self, core):
        core.local_query.side_effect = RuntimeError("backend down")
        executor = make_executor(core)
        executor.record_call("alpha", "local")

        result = await executor.execute(context="ctx")

        assert result.success is True
        assert result.cached_results == []
        assert executor.cache == {}

    async def test_stats_track_prefetch_hits_and_misses(self, core):
        executor = make_executor(core)
        executor.record_call("alpha", "local")
        await executor.execute(context="ctx")

        executor.check_cache("alpha", "local")  # hit
        executor.check_cache("alpha", "local")  # hit
        executor.check_cache("never prefetched", "local")  # miss

        stats = executor.get_stats()
        assert stats["cache_hits"] == 2
        assert stats["cache_misses"] == 1
        assert stats["cache_hit_rate"] == pytest.approx(2 / 3)
        assert stats["total_executions"] == 1


@pytest.mark.asyncio
class TestPatternInterface:
    async def test_implements_base_pattern_contract(self, core):
        executor = make_executor(core)
        executor.record_call("alpha", "local")

        assert isinstance(executor, BasePattern)
        result = await executor.execute(context="ctx")
        assert isinstance(result, SpeculativeExecutionResult)
        assert result.pattern_id == executor.config.pattern_id
        assert result.execution_time_ms >= 0

    async def test_execute_emits_prediction_and_completion_events(
        self, core, event_collector
    ):
        executor = make_executor(core)
        executor.add_event_handler(event_collector.collect)
        executor.record_call("alpha", "local")

        await executor.execute(context="ctx")

        types = [e["type"] for e in event_collector.get_events()]
        assert types[0] == "pattern.speculative.started"
        assert "pattern.speculative.predicted" in types
        assert types[-1] == "pattern.speculative.completed"
        predicted = event_collector.get_events("pattern.speculative.predicted")[0]
        assert predicted["data"]["num_predictions"] == 1


class TestCacheEvents:
    def test_cache_hit_event(self, core, event_collector):
        executor = make_executor(core)
        executor.add_event_handler(event_collector.collect)
        executor.cache["q_local"] = _entry("q")

        executor.check_cache("q", "local")

        events = event_collector.get_events("pattern.speculative.cache_hit")
        assert len(events) == 1
        assert events[0]["data"]["query"] == "q"
        assert events[0]["data"]["mode"] == "local"
        assert event_collector.get_events("pattern.speculative.cache_miss") == []

    def test_cache_miss_event(self, core, event_collector):
        executor = make_executor(core)
        executor.add_event_handler(event_collector.collect)

        executor.check_cache("absent", "global")

        events = event_collector.get_events("pattern.speculative.cache_miss")
        assert len(events) == 1
        assert events[0]["data"]["query"] == "absent"
        assert event_collector.get_events("pattern.speculative.cache_hit") == []

    def test_expired_entry_reports_a_miss(self, core, event_collector):
        executor = make_executor(core)
        executor.add_event_handler(event_collector.collect)
        executor.cache["q_local"] = _entry("q", ttl_seconds=-1.0)

        assert executor.check_cache("q", "local") is None
        assert len(event_collector.get_events("pattern.speculative.cache_miss")) == 1


def _entry(query, ttl_seconds=60.0, result="r"):
    from datetime import datetime

    from promptchain.integrations.lightrag.speculative import SpeculativeResult

    return SpeculativeResult(
        prediction_id=f"p_{query}",
        query=query,
        result=result,
        cached_at=datetime.utcnow(),
        ttl_seconds=ttl_seconds,
    )


class TestInvalidateCache:
    def test_removes_every_mode_of_a_query(self, core):
        executor = make_executor(core)
        executor.cache["q_local"] = _entry("q")
        executor.cache["q_global"] = _entry("q")
        executor.cache["other_local"] = _entry("other")

        assert executor.invalidate_cache("q") == 2
        assert list(executor.cache) == ["other_local"]

    def test_mode_restricts_removal(self, core):
        executor = make_executor(core)
        executor.cache["q_local"] = _entry("q")
        executor.cache["q_global"] = _entry("q")

        assert executor.invalidate_cache("q", mode="local") == 1
        assert list(executor.cache) == ["q_global"]

    def test_unknown_query_removes_nothing(self, core, event_collector):
        executor = make_executor(core)
        executor.add_event_handler(event_collector.collect)
        executor.cache["q_local"] = _entry("q")

        assert executor.invalidate_cache("nope") == 0
        assert len(executor.cache) == 1
        assert event_collector.get_events("pattern.speculative.invalidated") == []

    @pytest.mark.asyncio
    async def test_invalidated_query_is_a_miss_and_can_be_prefetched_again(self, core):
        executor = make_executor(core)
        executor.record_call("alpha", "local")
        await executor.execute(context="ctx")
        executor.invalidate_cache("alpha")
        assert executor.check_cache("alpha", "local") is None

        await executor.execute(context="ctx")
        assert core.local_query.await_count == 2
        assert executor.check_cache("alpha", "local") is not None


@pytest.mark.asyncio
class TestCacheSizeLimit:
    async def test_oldest_entries_are_evicted_past_the_limit(self, core):
        executor = make_executor(core, max_cache_size=2)
        for i in range(4):
            await executor._execute_prediction(_prediction(f"q{i}"))

        assert list(executor.cache) == ["q2_local", "q3_local"]

    async def test_refreshing_a_key_makes_it_newest(self, core):
        executor = make_executor(core, max_cache_size=2)
        await executor._execute_prediction(_prediction("a"))
        await executor._execute_prediction(_prediction("b"))
        await executor._execute_prediction(_prediction("a"))
        await executor._execute_prediction(_prediction("c"))

        assert list(executor.cache) == ["a_local", "c_local"]

    async def test_cache_is_unbounded_by_default(self, core):
        executor = make_executor(core)
        for i in range(30):
            await executor._execute_prediction(_prediction(f"q{i}"))

        assert len(executor.cache) == 30


def _prediction(query, mode="local"):
    from promptchain.integrations.lightrag.speculative import ToolPrediction

    return ToolPrediction(
        prediction_id=f"p_{query}",
        query=query,
        mode=mode,
        confidence=0.9,
        pattern_matched="test",
        context_hash="h",
    )


class TestConfigValidation:
    @pytest.mark.parametrize("window", [0, -1])
    def test_history_window_must_be_positive(self, window):
        with pytest.raises(ValueError, match="history_window must be at least 1"):
            SpeculativeConfig(history_window=window)

    @pytest.mark.parametrize("size", [0, -5])
    def test_max_cache_size_must_be_positive(self, size):
        with pytest.raises(ValueError, match="max_cache_size must be at least 1"):
            SpeculativeConfig(max_cache_size=size)

    def test_defaults_stay_valid(self):
        config = SpeculativeConfig()
        assert config.max_cache_size is None
        assert config.history_window == 20
