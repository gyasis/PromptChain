"""Unit tests for LightRAGHybridSearcher.

The searcher runs several LightRAG techniques in parallel and fuses their
ranked lists with RRF, weighted-linear or Borda fusion. The LightRAG
integration is a mock whose query methods return fixed ranked lists.

Already covered by tests/test_hybrid_search_pattern.py (not repeated here):
config defaults, basic execute, top_k override, technique contribution
totals, single/all technique runs, normalisation, the rrf_k parameter.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from promptchain.integrations.lightrag.hybrid_search import (
    FusionAlgorithm,
    HybridSearchConfig,
    LightRAGHybridSearcher,
    SearchTechnique,
)
from promptchain.patterns.base import BasePattern

LOCAL, GLOBAL, HYBRID = (
    SearchTechnique.LOCAL,
    SearchTechnique.GLOBAL,
    SearchTechnique.HYBRID,
)


def make_integration(local=None, global_=None, hybrid=None):
    integration = MagicMock()
    integration.core = MagicMock()
    integration.local_query = AsyncMock(return_value=local or {"results": []})
    integration.global_query = AsyncMock(return_value=global_ or {"results": []})
    integration.hybrid_query = AsyncMock(return_value=hybrid or {"results": []})
    return integration


def make_searcher(integration, **config_kwargs):
    config_kwargs.setdefault("techniques", [LOCAL, GLOBAL])
    return LightRAGHybridSearcher(integration, config=HybridSearchConfig(**config_kwargs))


@pytest.mark.asyncio
class TestFusionAlgorithms:
    async def test_rrf_ranks_documents_found_by_both_techniques_first(self):
        integration = make_integration(
            local={"results": ["a", "shared", "c"]},
            global_={"results": ["shared", "d"]},
        )
        searcher = make_searcher(integration, rrf_k=60)

        result = await searcher.execute(query="q")

        assert result.fused_results[0] == "shared"
        # shared: rank 2 in local (1/62) + rank 1 in global (1/61)
        assert result.fused_scores[0] == pytest.approx(1 / 62 + 1 / 61)
        assert result.fused_scores == sorted(result.fused_scores, reverse=True)

    async def test_borda_awards_points_by_rank_position(self):
        integration = make_integration(
            local={"results": ["d1", "d2", "d3"]},
            global_={"results": ["d3", "d1", "d2"]},
        )
        searcher = make_searcher(integration, fusion_algorithm=FusionAlgorithm.BORDA)

        result = await searcher.execute(query="q")

        scores = dict(zip(result.fused_results, result.fused_scores))
        # points per list = list length - rank: d1 3+2, d2 2+1, d3 1+3
        assert scores == {"d1": 5, "d2": 3, "d3": 4}
        assert result.fused_results == ["d1", "d3", "d2"]

    async def test_duplicates_across_techniques_appear_once(self):
        doc = {"id": "dup", "text": "same document"}
        integration = make_integration(
            local={"results": [doc]}, global_={"results": [doc]}
        )
        searcher = make_searcher(integration)

        result = await searcher.execute(query="q")

        assert result.fused_results == [doc]
        assert result.technique_contributions == {"local": 1, "global": 1}

    @pytest.mark.parametrize("algorithm", list(FusionAlgorithm))
    async def test_empty_results_fuse_to_nothing(self, algorithm):
        searcher = make_searcher(make_integration(), fusion_algorithm=algorithm)

        result = await searcher.execute(query="q")

        assert result.success is True
        assert result.fused_results == []
        assert result.fused_scores == []

    async def test_top_k_truncates_the_fused_list(self):
        integration = make_integration(
            local={"results": [f"l{i}" for i in range(5)]},
            global_={"results": [f"g{i}" for i in range(5)]},
        )
        searcher = make_searcher(integration, top_k=3)

        result = await searcher.execute(query="q")

        assert len(result.fused_results) == 3


@pytest.mark.asyncio
class TestWeightedLinearFusion:
    async def test_weights_scale_each_techniques_scores(self):
        integration = make_integration(
            local={"results": ["doc1"], "scores": [1.0]},
            global_={"results": ["doc1"], "scores": [0.5]},
        )
        searcher = make_searcher(
            integration,
            fusion_algorithm=FusionAlgorithm.LINEAR,
            normalize_scores=False,
            weights={LOCAL: 0.6, GLOBAL: 0.4},
        )

        result = await searcher.execute(query="q")

        assert result.fused_scores[0] == pytest.approx(0.6 * 1.0 + 0.4 * 0.5)

    async def test_heavier_weight_can_change_the_winner(self):
        integration = make_integration(
            local={"results": ["from_local"], "scores": [1.0]},
            global_={"results": ["from_global"], "scores": [1.0]},
        )
        searcher = make_searcher(
            integration,
            fusion_algorithm=FusionAlgorithm.LINEAR,
            weights={LOCAL: 0.2, GLOBAL: 0.8},
        )

        result = await searcher.execute(query="q")

        assert result.fused_results == ["from_global", "from_local"]

    async def test_without_weights_techniques_count_equally(self):
        integration = make_integration(
            local={"results": ["x"], "scores": [1.0]},
            global_={"results": ["x"], "scores": [1.0]},
        )
        searcher = make_searcher(integration, fusion_algorithm=FusionAlgorithm.LINEAR)

        result = await searcher.execute(query="q")

        assert result.fused_scores[0] == pytest.approx(1.0)

    async def test_weights_renormalise_when_a_technique_fails(self):
        integration = make_integration(local={"results": ["x"], "scores": [1.0]})
        integration.global_query.side_effect = RuntimeError("offline")
        searcher = make_searcher(
            integration,
            fusion_algorithm=FusionAlgorithm.LINEAR,
            normalize_scores=False,
            weights={LOCAL: 0.6, GLOBAL: 0.4},
        )

        result = await searcher.execute(query="q")

        # The surviving technique carries the whole weight.
        assert result.fused_scores == [pytest.approx(1.0)]

    def test_weights_must_sum_to_one(self):
        with pytest.raises(ValueError, match="weights must sum to 1.0"):
            HybridSearchConfig(
                techniques=[LOCAL, GLOBAL], weights={LOCAL: 0.7, GLOBAL: 0.5}
            )

    def test_weights_must_cover_every_technique(self):
        with pytest.raises(ValueError, match="weights missing for techniques"):
            HybridSearchConfig(techniques=[LOCAL, GLOBAL], weights={LOCAL: 1.0})

    def test_extra_weight_keys_are_ignored(self):
        config = HybridSearchConfig(
            techniques=[LOCAL], weights={LOCAL: 1.0, GLOBAL: 5.0}
        )
        assert config.weights[LOCAL] == 1.0

    def test_weights_default_to_none(self):
        assert HybridSearchConfig().weights is None


class TestConfigValidation:
    def test_string_technique_is_rejected(self):
        with pytest.raises(ValueError, match="Invalid technique"):
            HybridSearchConfig(techniques=["local", "invalid_technique"])

    def test_empty_techniques_are_rejected(self):
        with pytest.raises(ValueError, match="techniques must not be empty"):
            HybridSearchConfig(techniques=[])


@pytest.mark.asyncio
class TestTechniqueExecution:
    async def test_every_configured_technique_is_queried_once(self):
        integration = make_integration()
        searcher = make_searcher(integration, techniques=[LOCAL, GLOBAL, HYBRID])

        await searcher.execute(query="the query")

        integration.local_query.assert_awaited_once_with("the query", top_k=10)
        integration.global_query.assert_awaited_once_with("the query", top_k=10)
        integration.hybrid_query.assert_awaited_once_with("the query", top_k=10)

    async def test_partial_failure_keeps_the_surviving_results(self, event_collector):
        integration = make_integration(local={"results": ["doc1"]})
        integration.global_query.side_effect = RuntimeError("Query failed")
        searcher = make_searcher(integration)
        searcher.add_event_handler(event_collector.collect)

        result = await searcher.execute(query="q")

        assert result.success is True
        assert result.fused_results == ["doc1"]
        assert [tr.technique for tr in result.technique_results] == [LOCAL]
        failed = event_collector.get_events("pattern.hybrid_search.technique_failed")
        assert failed[0]["data"]["technique"] == "global"
        assert "Query failed" in failed[0]["data"]["error"]

    @pytest.mark.parametrize("algorithm", list(FusionAlgorithm))
    async def test_all_techniques_failing_is_a_failure(self, algorithm):
        integration = make_integration()
        integration.local_query.side_effect = RuntimeError("down")
        integration.global_query.side_effect = RuntimeError("down")
        searcher = make_searcher(integration, fusion_algorithm=algorithm)

        result = await searcher.execute(query="q")

        assert result.success is False
        assert result.errors == ["Every configured search technique failed"]
        assert result.fused_results == []


@pytest.mark.asyncio
class TestEventsAndInterface:
    async def test_each_technique_reports_completion(self, event_collector):
        searcher = make_searcher(make_integration(local={"results": ["a", "b"]}))
        searcher.add_event_handler(event_collector.collect)

        await searcher.execute(query="q")

        done = event_collector.get_events("pattern.hybrid_search.technique_completed")
        assert {e["data"]["technique"] for e in done} == {"local", "global"}
        counts = {e["data"]["technique"]: e["data"]["num_results"] for e in done}
        assert counts == {"local": 2, "global": 0}

    async def test_fusing_and_completed_events_carry_counts(self, event_collector):
        searcher = make_searcher(
            make_integration(local={"results": ["a"]}, global_={"results": ["b"]})
        )
        searcher.add_event_handler(event_collector.collect)

        await searcher.execute(query="q")

        fusing = event_collector.get_events("pattern.hybrid_search.fusing")[0]
        assert fusing["data"]["num_techniques"] == 2
        assert fusing["data"]["total_results"] == 2
        completed = event_collector.get_events("pattern.hybrid_search.completed")[0]
        assert completed["data"]["num_fused_results"] == 2

    async def test_implements_base_pattern_contract(self):
        searcher = make_searcher(make_integration(local={"results": ["a"]}))

        assert isinstance(searcher, BasePattern)
        result = await searcher.execute(query="q")
        assert result.pattern_id == searcher.config.pattern_id
        assert result.result == result.fused_results == ["a"]
