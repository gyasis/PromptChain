"""E2E tests for research workflow patterns.

Tests multi-pattern compositions:
1. Query Expansion -> Multi-Hop workflow
2. Branching -> Hybrid Fusion workflow
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from promptchain.integrations.lightrag.query_expansion import (
    ExpansionStrategy,
    QueryExpansionConfig,
    LightRAGQueryExpander,
)
from promptchain.integrations.lightrag.multi_hop import (
    MultiHopConfig,
    LightRAGMultiHop,
)
from promptchain.integrations.lightrag.branching import (
    BranchingConfig,
    LightRAGBranchingThoughts,
)
from promptchain.integrations.lightrag.hybrid_search import (
    HybridSearchConfig,
    LightRAGHybridSearcher,
)


@pytest.mark.e2e
@pytest.mark.asyncio
class TestQueryExpansionToMultiHop:
    """Test Query Expansion -> Multi-Hop workflow."""

    async def test_expansion_feeds_multihop(
        self, e2e_test_context, mock_message_bus
    ):
        """Test query expansion results feeding into multi-hop reasoning.

        Workflow:
        1. Expand complex query into multiple variations
        2. Use expanded queries in multi-hop reasoning
        3. Synthesize comprehensive answer
        """
        # Setup
        lightrag = e2e_test_context["lightrag"]
        search = e2e_test_context["search"]

        # Phase 1: Query Expansion
        expander = LightRAGQueryExpander(
            lightrag_integration=lightrag,
            config=QueryExpansionConfig(
                strategies=[ExpansionStrategy.SEMANTIC, ExpansionStrategy.REFORMULATION],
                max_expansions_per_strategy=2,
                min_similarity=0.5,
            )
        )
        expander.connect_messagebus(mock_message_bus)

        expansion_result = await expander.execute(
            query="What are the differences between transformers and RNNs?"
        )

        # Verify expansion
        assert expansion_result.success
        assert len(expansion_result.expanded_queries) > 0
        assert expansion_result.unique_results_found > 0

        # Verify events emitted
        expansion_events = mock_message_bus.get_events_by_type("pattern.query_expansion.*")
        assert len(expansion_events) >= 3  # started, expanded, completed
        assert any("started" in e["type"] for e in expansion_events)
        assert any("expanded" in e["type"] for e in expansion_events)
        assert any("completed" in e["type"] for e in expansion_events)

        # Phase 2: Multi-Hop Reasoning with expanded context
        multi_hop = LightRAGMultiHop(
            search_interface=search,
            config=MultiHopConfig(max_hops=3, decompose_first=True)
        )
        multi_hop.connect_messagebus(mock_message_bus)

        # Use insights from expansion to guide multi-hop
        expanded_queries_text = " ".join([
            eq.expanded_query for eq in expansion_result.expanded_queries
        ])

        multi_hop_result = await multi_hop.execute(
            question=f"{expansion_result.original_query}. Consider: {expanded_queries_text[:200]}"
        )

        # Verify multi-hop execution
        assert multi_hop_result.success
        assert multi_hop_result.hops_executed >= 1
        assert len(multi_hop_result.unified_answer) > 0

        # Verify multi-hop events
        multihop_events = mock_message_bus.get_events_by_type("pattern.multi_hop.*")
        assert len(multihop_events) >= 3  # started, hop_completed, completed

        # Verify workflow event sequence
        event_sequence = mock_message_bus.get_event_sequence()
        expansion_started = event_sequence.index("pattern.query_expansion.started")
        expansion_completed = event_sequence.index("pattern.query_expansion.completed")
        multihop_started = event_sequence.index("pattern.multi_hop.started")

        # Expansion should complete before multi-hop starts
        assert expansion_completed < multihop_started

        # Verify data flow
        assert multi_hop_result.execution_time_ms > 0
        assert expansion_result.execution_time_ms > 0

        # Total workflow timing
        total_time = expansion_result.execution_time_ms + multi_hop_result.execution_time_ms
        assert total_time > 0

    async def test_expansion_failure_recovery(
        self, e2e_test_context, mock_message_bus
    ):
        """Test multi-hop can still execute if expansion partially fails."""
        lightrag = e2e_test_context["lightrag"]
        search = e2e_test_context["search"]

        # Create expander with very high similarity threshold (will filter most)
        expander = LightRAGQueryExpander(
            lightrag_integration=lightrag,
            config=QueryExpansionConfig(
                strategies=[ExpansionStrategy.SEMANTIC],
                min_similarity=0.99,  # Very high threshold
            )
        )
        expander.connect_messagebus(mock_message_bus)

        expansion_result = await expander.execute(
            query="What is machine learning?"
        )

        # Expansion may succeed but with few/no expansions
        # Multi-hop should still work with original query
        multi_hop = LightRAGMultiHop(
            search_interface=search,
            config=MultiHopConfig(max_hops=2, decompose_first=False)
        )
        multi_hop.connect_messagebus(mock_message_bus)

        multi_hop_result = await multi_hop.execute(
            question="What is machine learning?"
        )

        # Multi-hop should succeed even if expansion was limited
        assert multi_hop_result.success
        assert len(multi_hop_result.unified_answer) > 0


@pytest.mark.e2e
@pytest.mark.asyncio
class TestBranchingToHybridFusion:
    """Test Branching -> Hybrid Fusion workflow."""

    async def test_branching_hypotheses_feed_fusion(
        self, e2e_test_context, mock_message_bus
    ):
        """Test branching hypothesis generation feeding into hybrid fusion.

        Workflow:
        1. Generate multiple hypotheses via branching
        2. Use top hypotheses to guide hybrid search fusion
        3. Combine results for comprehensive answer
        """
        lightrag = e2e_test_context["lightrag"]

        # Phase 1: Branching Thoughts
        branching = LightRAGBranchingThoughts(
            lightrag_integration=lightrag,
            config=BranchingConfig(
                num_branches=3,
                selection_strategy="top_n",
                top_n=2,
            )
        )
        branching.connect_messagebus(mock_message_bus)

        branching_result = await branching.execute(
            query="How to optimize neural network training?"
        )

        # Verify branching
        assert branching_result.success
        assert len(branching_result.hypotheses) == 3
        assert len(branching_result.selected_hypotheses) == 2

        # Verify branching events
        branching_events = mock_message_bus.get_events_by_type("pattern.branching.*")
        assert any("hypothesis_generated" in e["type"] for e in branching_events)
        assert any("selected" in e["type"] for e in branching_events)

        # Phase 2: Hybrid Fusion with selected hypotheses
        fusion = LightRAGHybridSearcher(
            lightrag_integration=lightrag,
            config=HybridSearchConfig()
        )
        fusion.connect_messagebus(mock_message_bus)

        # Use selected hypotheses to guide search
        hypothesis_queries = [
            h.text for h in branching_result.selected_hypotheses
        ]

        # Execute fusion for each hypothesis
        fusion_results = []
        for hypothesis in hypothesis_queries:
            result = await fusion.execute(query=hypothesis)
            fusion_results.append(result)

        # Verify all fusions succeeded
        assert all(r.success for r in fusion_results)
        assert all(len(r.fused_results) > 0 for r in fusion_results)

        # Verify fusion events
        fusion_events = mock_message_bus.get_events_by_type("pattern.hybrid_search.*")
        assert len(fusion_events) >= 6  # 2 hypotheses * 3 events each

        # Verify workflow coordination
        event_sequence = mock_message_bus.get_event_sequence()

        # Find last branching event
        branching_completed_idx = max([
            i for i, evt in enumerate(event_sequence)
            if "branching.completed" in evt
        ])

        # Find first fusion event
        fusion_started_idx = min([
            i for i, evt in enumerate(event_sequence)
            if "hybrid_search.started" in evt
        ])

        # Branching should complete before fusion starts
        assert branching_completed_idx < fusion_started_idx

        # Verify result aggregation
        total_fused_results = sum(len(r.fused_results) for r in fusion_results)
        assert total_fused_results > 0

    async def test_branching_with_metric_filtering(
        self, e2e_test_context, mock_message_bus
    ):
        """Test branching with quality metrics filters fusion inputs."""
        lightrag = e2e_test_context["lightrag"]

        # Branching with quality threshold
        branching = LightRAGBranchingThoughts(
            lightrag_integration=lightrag,
            config=BranchingConfig(
                num_branches=5,
                selection_strategy="threshold",
                quality_threshold=0.6,
            )
        )
        branching.connect_messagebus(mock_message_bus)

        branching_result = await branching.execute(
            query="What are best practices for deep learning?"
        )

        # Only high-quality hypotheses should be selected
        assert branching_result.success
        assert len(branching_result.selected_hypotheses) <= len(branching_result.hypotheses)

        # All selected should meet threshold
        for h in branching_result.selected_hypotheses:
            assert h.quality_score >= 0.6

        # Use filtered hypotheses in fusion
        fusion = LightRAGHybridSearcher(
            lightrag_integration=lightrag,
            config=HybridSearchConfig()
        )

        if branching_result.selected_hypotheses:
            fusion_result = await fusion.execute(
                query=branching_result.selected_hypotheses[0].text
            )
            assert fusion_result.success


@pytest.mark.e2e
@pytest.mark.asyncio
class TestMultiPatternComposition:
    """Test complex multi-pattern compositions."""

    async def test_three_pattern_pipeline(
        self, e2e_test_context, mock_message_bus, research_queries
    ):
        """Test Query Expansion -> Branching -> Multi-Hop pipeline.

        Complex workflow:
        1. Expand query into variations
        2. Generate hypotheses from expansions
        3. Multi-hop reasoning for each hypothesis
        """
        lightrag = e2e_test_context["lightrag"]
        search = e2e_test_context["search"]

        query = research_queries[2]  # "Compare transformers vs RNNs"

        # Phase 1: Expansion
        expander = LightRAGQueryExpander(
            lightrag_integration=lightrag,
            config=QueryExpansionConfig(
                strategies=[ExpansionStrategy.SEMANTIC],
                max_expansions_per_strategy=2,
            )
        )
        expander.connect_messagebus(mock_message_bus)

        expansion_result = await expander.execute(query=query)
        assert expansion_result.success

        # Phase 2: Branching from expanded queries
        branching = LightRAGBranchingThoughts(
            lightrag_integration=lightrag,
            config=BranchingConfig(num_branches=3, selection_strategy="top_n", top_n=2)
        )
        branching.connect_messagebus(mock_message_bus)

        # Use first expanded query for branching
        branching_query = (
            expansion_result.expanded_queries[0].expanded_query
            if expansion_result.expanded_queries
            else query
        )

        branching_result = await branching.execute(query=branching_query)
        assert branching_result.success

        # Phase 3: Multi-hop for top hypothesis
        multi_hop = LightRAGMultiHop(
            search_interface=search,
            config=MultiHopConfig(max_hops=2)
        )
        multi_hop.connect_messagebus(mock_message_bus)

        top_hypothesis = branching_result.selected_hypotheses[0].text
        multi_hop_result = await multi_hop.execute(question=top_hypothesis)
        assert multi_hop_result.success

        # Verify complete event sequence
        event_sequence = mock_message_bus.get_event_sequence()

        # Find phase boundaries
        expansion_end = max([i for i, e in enumerate(event_sequence) if "expansion.completed" in e])
        branching_end = max([i for i, e in enumerate(event_sequence) if "branching.completed" in e])
        multihop_end = max([i for i, e in enumerate(event_sequence) if "multi_hop.completed" in e])

        # Verify sequential execution
        assert expansion_end < branching_end < multihop_end

        # Verify data flow integrity
        assert len(expansion_result.expanded_queries) > 0
        assert len(branching_result.selected_hypotheses) > 0
        assert len(multi_hop_result.unified_answer) > 0

    async def test_parallel_pattern_execution(
        self, e2e_test_context, mock_message_bus
    ):
        """Test executing multiple patterns in parallel for same query."""
        lightrag = e2e_test_context["lightrag"]
        search = e2e_test_context["search"]

        query = "What is deep learning?"

        # Execute multiple patterns in parallel
        expander = LightRAGQueryExpander(
            lightrag_integration=lightrag,
            config=QueryExpansionConfig(strategies=[ExpansionStrategy.SEMANTIC])
        )
        expander.connect_messagebus(mock_message_bus)

        branching = LightRAGBranchingThoughts(
            lightrag_integration=lightrag,
            config=BranchingConfig(num_branches=2)
        )
        branching.connect_messagebus(mock_message_bus)

        multi_hop = LightRAGMultiHop(
            search_interface=search,
            config=MultiHopConfig(max_hops=2)
        )
        multi_hop.connect_messagebus(mock_message_bus)

        # Run all patterns concurrently
        results = await asyncio.gather(
            expander.execute(query=query),
            branching.execute(query=query),
            multi_hop.execute(question=query),
        )

        expansion_result, branching_result, multi_hop_result = results

        # All should succeed
        assert expansion_result.success
        assert branching_result.success
        assert multi_hop_result.success

        # Verify parallel execution via timing
        # (events may be interleaved)
        all_events = mock_message_bus.events

        # Should have events from all three patterns
        assert any("expansion" in e["type"] for e in all_events)
        assert any("branching" in e["type"] for e in all_events)
        assert any("multi_hop" in e["type"] for e in all_events)

        # Each pattern should have completed
        assert any("expansion.completed" in e["type"] for e in all_events)
        assert any("branching.completed" in e["type"] for e in all_events)
        assert any("multi_hop.completed" in e["type"] for e in all_events)
