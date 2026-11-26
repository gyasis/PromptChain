"""Unit tests for LightRAGQueryExpander.

The expander builds query variations (synonym, semantic, acronym,
reformulation) from knowledge-graph context, searches the original query plus
every variation through ``SearchInterface.multi_query_search`` and returns the
de-duplicated results in first-seen order. Search interface and integration
are mocks.

Already covered by tests/test_query_expansion_pattern.py (not repeated here):
the per-strategy expansion rules, multiple strategies, the sequential
fallback, similarity filtering, context-extraction failure, strategy override.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from promptchain.integrations.lightrag.query_expansion import (
    ExpansionStrategy,
    LightRAGQueryExpander,
    QueryExpansionConfig,
)
from promptchain.patterns.base import BasePattern

CONTEXT = {
    "entities": [{"name": "machine learning", "type": "concept"}],
    "relationships": [{"type": "uses", "target": "algorithms"}],
}


def make_expander(results=None, context=None, **config_kwargs):
    search = MagicMock()
    search.multi_query_search = AsyncMock(
        return_value=results if results is not None else ["r1", "r2"]
    )
    search.hybrid_search = AsyncMock(return_value="sequential result")
    integration = MagicMock()
    integration.search = search
    integration.extract_context = AsyncMock(
        return_value=CONTEXT if context is None else context
    )
    config_kwargs.setdefault("strategies", [ExpansionStrategy.SEMANTIC])
    expander = LightRAGQueryExpander(
        lightrag_integration=integration, config=QueryExpansionConfig(**config_kwargs)
    )
    return expander, search


@pytest.mark.asyncio
class TestSearchDelegation:
    async def test_original_and_expansions_go_to_multi_query_search(self):
        expander, search = make_expander()

        result = await expander.execute(query="machine learning")

        search.multi_query_search.assert_awaited_once_with(
            queries=["machine learning", "machine learning uses algorithms"],
            mode="hybrid",
        )
        assert [eq.expanded_query for eq in result.expanded_queries] == [
            "machine learning uses algorithms"
        ]

    async def test_no_expansions_still_searches_the_original_query(self):
        expander, search = make_expander(context={})

        result = await expander.execute(query="test")

        assert result.success is True
        assert result.expanded_queries == []
        search.multi_query_search.assert_awaited_once_with(
            queries=["test"], mode="hybrid"
        )
        assert result.metadata["total_queries"] == 1

    async def test_expansion_cap_limits_the_query_list(self):
        context = {"relationships": [{"type": "r", "target": f"t{i}"} for i in range(6)]}
        expander, search = make_expander(context=context, max_expansions_per_strategy=2)

        await expander.execute(query="q")

        queries = search.multi_query_search.await_args.kwargs["queries"]
        assert len(queries) == 3  # original + 2 expansions

    async def test_strategy_override_replaces_configured_strategies(self):
        expander, search = make_expander()

        result = await expander.execute(
            query="what", strategies=[ExpansionStrategy.REFORMULATION]
        )

        assert {eq.strategy for eq in result.expanded_queries} == {
            ExpansionStrategy.REFORMULATION
        }
        assert result.metadata["strategies_used"] == ["reformulation"]


@pytest.mark.asyncio
class TestResultOrdering:
    async def test_duplicates_are_dropped_keeping_first_occurrence_order(self):
        expander, _ = make_expander(results=["best", "good", "best", "ok", "good"])

        result = await expander.execute(query="q")

        assert result.search_results == ["best", "good", "ok"]
        assert result.unique_results_found == 3

    async def test_ranked_input_keeps_its_ranking(self):
        ranked = [{"id": i, "score": 1 - i / 10} for i in range(4)]
        expander, _ = make_expander(results=ranked)

        result = await expander.execute(query="q")

        assert [r["id"] for r in result.search_results] == [0, 1, 2, 3]

    async def test_deduplicate_off_returns_every_result(self):
        expander, _ = make_expander(results=["a", "a", "b"], deduplicate=False)

        result = await expander.execute(query="q")

        assert result.search_results == ["a", "a", "b"]


class TestConfigValidation:
    @pytest.mark.parametrize("count", [0, -2])
    def test_expansion_cap_must_be_positive(self, count):
        with pytest.raises(
            ValueError, match="max_expansions_per_strategy must be at least 1"
        ):
            QueryExpansionConfig(max_expansions_per_strategy=count)

    def test_strategies_must_not_be_empty(self):
        with pytest.raises(ValueError, match="strategies must not be empty"):
            QueryExpansionConfig(strategies=[])

    def test_defaults_stay_valid(self):
        config = QueryExpansionConfig()
        assert config.strategies == [ExpansionStrategy.SEMANTIC]
        assert config.max_expansions_per_strategy == 3

    def test_search_source_is_required(self):
        with pytest.raises(ValueError, match="search_interface or lightrag_integration"):
            LightRAGQueryExpander()


@pytest.mark.asyncio
class TestEventsAndInterface:
    async def test_event_sequence(self, event_collector):
        expander, _ = make_expander()
        expander.add_event_handler(event_collector.collect)

        await expander.execute(query="machine learning")

        assert [e["type"] for e in event_collector.get_events()] == [
            "pattern.query_expansion.started",
            "pattern.query_expansion.expanded",
            "pattern.query_expansion.searching",
            "pattern.query_expansion.completed",
        ]

    async def test_completed_event_reports_counts(self, event_collector):
        expander, _ = make_expander(results=["a", "a", "b"])
        expander.add_event_handler(event_collector.collect)

        await expander.execute(query="machine learning")

        data = event_collector.get_events("pattern.query_expansion.completed")[0]["data"]
        assert data["unique_results"] == 2
        assert data["total_expansions"] == 1

    async def test_metadata_records_run_shape(self):
        expander, _ = make_expander(parallel_search=True)

        result = await expander.execute(query="machine learning")

        assert result.metadata == {
            "strategies_used": ["semantic"],
            "total_queries": 2,
            "parallel_execution": True,
        }

    async def test_search_failure_returns_a_failed_result(self):
        expander, search = make_expander()
        search.multi_query_search.side_effect = RuntimeError("index offline")

        result = await expander.execute(query="q")

        assert result.success is False
        assert result.errors == ["index offline"]

    async def test_implements_base_pattern_contract(self):
        expander, _ = make_expander()

        assert isinstance(expander, BasePattern)
        result = await expander.execute(query="q")
        assert result.pattern_id == expander.config.pattern_id
        assert result.original_query == "q"
        assert result.to_dict()["unique_results_found"] == 2
