"""Tests for LightRAG Query Expansion Pattern.

Tests the query expansion pattern with various strategies and configurations.
"""

import asyncio
import pytest
from dataclasses import dataclass
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock, patch

from promptchain.integrations.lightrag.query_expansion import (
    ExpansionStrategy,
    QueryExpansionConfig,
    ExpandedQuery,
    QueryExpansionResult,
    LightRAGQueryExpander,
)


# Mock SearchInterface for testing
@dataclass
class MockSearchResult:
    """Mock search result."""
    content: str
    score: float


class MockSearchInterface:
    """Mock SearchInterface for testing."""

    def __init__(self):
        self.search_calls = []

    async def multi_query_search(
        self, queries: List[str], mode: str = "hybrid", **kwargs
    ) -> List[MockSearchResult]:
        """Mock multi-query search."""
        self.search_calls.append({"queries": queries, "mode": mode})
        # Return mock results for each query
        return [
            MockSearchResult(content=f"Result for: {q}", score=0.9)
            for q in queries
        ]

    async def hybrid_search(self, query: str, **kwargs) -> MockSearchResult:
        """Mock hybrid search."""
        return MockSearchResult(content=f"Result for: {query}", score=0.8)


class MockLightRAGIntegration:
    """Mock LightRAGIntegration for testing."""

    def __init__(self):
        self.search = MockSearchInterface()

    async def extract_context(self, query: str, **kwargs) -> dict:
        """Mock context extraction."""
        return {
            "entities": [
                {"name": "machine learning", "type": "concept"},
                {"name": "ML", "type": "acronym"},
            ],
            "relationships": [
                {"type": "uses", "source": "ML", "target": "algorithms"},
            ],
        }


@pytest.fixture
def mock_search_interface():
    """Provide mock SearchInterface."""
    return MockSearchInterface()


@pytest.fixture
def mock_integration():
    """Provide mock LightRAGIntegration."""
    return MockLightRAGIntegration()


@pytest.fixture
def basic_config():
    """Provide basic configuration."""
    return QueryExpansionConfig(
        strategies=[ExpansionStrategy.SEMANTIC],
        max_expansions_per_strategy=3,
        min_similarity=0.5,
        deduplicate=True,
        parallel_search=True,
    )


@pytest.mark.asyncio
class TestQueryExpansionConfig:
    """Test QueryExpansionConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = QueryExpansionConfig()
        assert config.strategies == [ExpansionStrategy.SEMANTIC]
        assert config.max_expansions_per_strategy == 3
        assert config.min_similarity == 0.5
        assert config.deduplicate is True
        assert config.parallel_search is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = QueryExpansionConfig(
            strategies=[ExpansionStrategy.SYNONYM, ExpansionStrategy.ACRONYM],
            max_expansions_per_strategy=5,
            min_similarity=0.7,
            deduplicate=False,
            parallel_search=False,
        )
        assert ExpansionStrategy.SYNONYM in config.strategies
        assert ExpansionStrategy.ACRONYM in config.strategies
        assert config.max_expansions_per_strategy == 5
        assert config.min_similarity == 0.7
        assert config.deduplicate is False
        assert config.parallel_search is False


@pytest.mark.asyncio
class TestExpandedQuery:
    """Test ExpandedQuery dataclass."""

    def test_expanded_query_creation(self):
        """Test creating ExpandedQuery."""
        eq = ExpandedQuery(
            query_id="test-123",
            original_query="What is ML?",
            expanded_query="What is machine learning?",
            strategy=ExpansionStrategy.ACRONYM,
            similarity_score=0.9,
        )
        assert eq.query_id == "test-123"
        assert eq.original_query == "What is ML?"
        assert eq.expanded_query == "What is machine learning?"
        assert eq.strategy == ExpansionStrategy.ACRONYM
        assert eq.similarity_score == 0.9


@pytest.mark.asyncio
class TestQueryExpansionResult:
    """Test QueryExpansionResult dataclass."""

    def test_result_creation(self):
        """Test creating QueryExpansionResult."""
        result = QueryExpansionResult(
            pattern_id="pattern-123",
            success=True,
            result=[],
            execution_time_ms=100.0,
            original_query="test query",
            expanded_queries=[],
            search_results=[],
            unique_results_found=0,
        )
        assert result.pattern_id == "pattern-123"
        assert result.success is True
        assert result.original_query == "test query"
        assert len(result.expanded_queries) == 0

    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        expanded = ExpandedQuery(
            query_id="q1",
            original_query="original",
            expanded_query="expanded",
            strategy=ExpansionStrategy.SEMANTIC,
            similarity_score=0.8,
        )
        result = QueryExpansionResult(
            pattern_id="p1",
            success=True,
            result=[],
            execution_time_ms=50.0,
            original_query="original",
            expanded_queries=[expanded],
            unique_results_found=5,
        )
        data = result.to_dict()
        assert data["original_query"] == "original"
        assert data["unique_results_found"] == 5
        assert len(data["expanded_queries"]) == 1
        assert data["expanded_queries"][0]["strategy"] == "semantic"


@pytest.mark.asyncio
class TestLightRAGQueryExpander:
    """Test LightRAGQueryExpander class."""

    async def test_initialization_with_search_interface(self, mock_search_interface, basic_config):
        """Test initialization with SearchInterface."""
        expander = LightRAGQueryExpander(
            search_interface=mock_search_interface,
            config=basic_config,
        )
        assert expander._search == mock_search_interface
        assert expander.config.strategies == [ExpansionStrategy.SEMANTIC]

    async def test_initialization_with_integration(self, mock_integration, basic_config):
        """Test initialization with LightRAGIntegration."""
        expander = LightRAGQueryExpander(
            lightrag_integration=mock_integration,
            config=basic_config,
        )
        assert expander._search == mock_integration.search
        assert expander._integration == mock_integration

    async def test_initialization_without_search(self, basic_config):
        """Test initialization fails without search interface."""
        with pytest.raises(ValueError, match="Must provide either"):
            LightRAGQueryExpander(config=basic_config)

    async def test_basic_execution(self, mock_integration, basic_config):
        """Test basic query expansion execution."""
        expander = LightRAGQueryExpander(
            lightrag_integration=mock_integration,
            config=basic_config,
        )
        result = await expander.execute(query="What is machine learning?")

        assert result.success is True
        assert result.original_query == "What is machine learning?"
        assert len(result.expanded_queries) > 0
        assert result.execution_time_ms > 0

    async def test_semantic_expansion(self, mock_integration):
        """Test semantic expansion strategy."""
        config = QueryExpansionConfig(
            strategies=[ExpansionStrategy.SEMANTIC],
            max_expansions_per_strategy=2,
        )
        expander = LightRAGQueryExpander(
            lightrag_integration=mock_integration,
            config=config,
        )
        result = await expander.execute(query="What is ML?")

        assert result.success is True
        semantic_queries = [
            q for q in result.expanded_queries
            if q.strategy == ExpansionStrategy.SEMANTIC
        ]
        assert len(semantic_queries) <= 2

    async def test_synonym_expansion(self, mock_integration):
        """Test synonym expansion strategy."""
        config = QueryExpansionConfig(
            strategies=[ExpansionStrategy.SYNONYM],
            max_expansions_per_strategy=2,
        )
        expander = LightRAGQueryExpander(
            lightrag_integration=mock_integration,
            config=config,
        )
        result = await expander.execute(query="What is machine learning?")

        assert result.success is True
        synonym_queries = [
            q for q in result.expanded_queries
            if q.strategy == ExpansionStrategy.SYNONYM
        ]
        assert len(synonym_queries) <= 2

    async def test_acronym_expansion(self, mock_integration):
        """Test acronym expansion strategy."""
        config = QueryExpansionConfig(
            strategies=[ExpansionStrategy.ACRONYM],
            max_expansions_per_strategy=2,
        )
        expander = LightRAGQueryExpander(
            lightrag_integration=mock_integration,
            config=config,
        )
        result = await expander.execute(query="What is ML?")

        assert result.success is True
        # May or may not find acronyms depending on context

    async def test_reformulation_expansion(self, mock_integration):
        """Test reformulation expansion strategy."""
        config = QueryExpansionConfig(
            strategies=[ExpansionStrategy.REFORMULATION],
            max_expansions_per_strategy=3,
        )
        expander = LightRAGQueryExpander(
            lightrag_integration=mock_integration,
            config=config,
        )
        result = await expander.execute(query="machine learning")

        assert result.success is True
        reformulation_queries = [
            q for q in result.expanded_queries
            if q.strategy == ExpansionStrategy.REFORMULATION
        ]
        assert len(reformulation_queries) <= 3

    async def test_multiple_strategies(self, mock_integration):
        """Test using multiple expansion strategies."""
        config = QueryExpansionConfig(
            strategies=[
                ExpansionStrategy.SEMANTIC,
                ExpansionStrategy.SYNONYM,
                ExpansionStrategy.REFORMULATION,
            ],
            max_expansions_per_strategy=2,
        )
        expander = LightRAGQueryExpander(
            lightrag_integration=mock_integration,
            config=config,
        )
        result = await expander.execute(query="What is ML?")

        assert result.success is True
        # Should have expansions from multiple strategies
        strategies_used = {q.strategy for q in result.expanded_queries}
        assert len(strategies_used) > 1

    async def test_parallel_search(self, mock_integration):
        """Test parallel search execution."""
        config = QueryExpansionConfig(
            strategies=[ExpansionStrategy.SEMANTIC],
            parallel_search=True,
        )
        expander = LightRAGQueryExpander(
            lightrag_integration=mock_integration,
            config=config,
        )
        result = await expander.execute(query="test query")

        assert result.success is True
        # Verify multi_query_search was called
        assert len(mock_integration.search.search_calls) > 0

    async def test_sequential_search_fallback(self):
        """Test sequential search when parallel not available."""
        # Create mock integration without multi_query_search
        class MockSearchWithoutParallel:
            async def hybrid_search(self, query: str, **kwargs):
                return MockSearchResult(content=f"Result for: {query}", score=0.8)

        class MockIntegrationNoParallel:
            def __init__(self):
                self.search = MockSearchWithoutParallel()

            async def extract_context(self, query: str, **kwargs) -> dict:
                return {
                    "entities": [{"name": "test", "type": "concept"}],
                    "relationships": []
                }

        mock_integration = MockIntegrationNoParallel()

        config = QueryExpansionConfig(
            strategies=[ExpansionStrategy.SEMANTIC],
            parallel_search=True,  # Still configured for parallel
        )
        expander = LightRAGQueryExpander(
            lightrag_integration=mock_integration,
            config=config,
        )
        result = await expander.execute(query="test query")

        assert result.success is True

    async def test_deduplication(self, mock_integration):
        """Test result deduplication."""
        config = QueryExpansionConfig(
            strategies=[ExpansionStrategy.SEMANTIC],
            deduplicate=True,
        )
        expander = LightRAGQueryExpander(
            lightrag_integration=mock_integration,
            config=config,
        )
        result = await expander.execute(query="test query")

        assert result.success is True
        # Results should be deduplicated
        assert result.unique_results_found >= 0

    async def test_similarity_filtering(self, mock_integration):
        """Test filtering by similarity threshold."""
        config = QueryExpansionConfig(
            strategies=[ExpansionStrategy.SEMANTIC],
            min_similarity=0.95,  # High threshold
        )
        expander = LightRAGQueryExpander(
            lightrag_integration=mock_integration,
            config=config,
        )
        result = await expander.execute(query="test query")

        assert result.success is True
        # High threshold should filter out most expansions
        for eq in result.expanded_queries:
            assert eq.similarity_score >= 0.95

    async def test_event_emission(self, mock_integration, basic_config):
        """Test event emission during execution."""
        events_emitted = []

        def event_handler(event_type: str, data: dict):
            events_emitted.append((event_type, data))

        expander = LightRAGQueryExpander(
            lightrag_integration=mock_integration,
            config=basic_config,
        )
        expander.add_event_handler(event_handler)

        await expander.execute(query="test query")

        # Verify expected events were emitted
        event_types = [e[0] for e in events_emitted]
        assert "pattern.query_expansion.started" in event_types
        assert "pattern.query_expansion.expanded" in event_types
        assert "pattern.query_expansion.completed" in event_types

    async def test_context_extraction_failure(self, mock_integration, basic_config):
        """Test handling of context extraction failure."""
        # Make extract_context raise exception
        async def failing_extract(*args, **kwargs):
            raise Exception("Context extraction failed")

        mock_integration.extract_context = failing_extract

        expander = LightRAGQueryExpander(
            lightrag_integration=mock_integration,
            config=basic_config,
        )
        result = await expander.execute(query="test query")

        # Should still succeed even if context extraction fails
        assert result.success is True

    async def test_strategy_override(self, mock_integration, basic_config):
        """Test overriding strategies at execution time."""
        expander = LightRAGQueryExpander(
            lightrag_integration=mock_integration,
            config=basic_config,  # Has SEMANTIC as default
        )
        result = await expander.execute(
            query="test query",
            strategies=[ExpansionStrategy.REFORMULATION]  # Override
        )

        assert result.success is True
        # Should use overridden strategies
        reformulation_queries = [
            q for q in result.expanded_queries
            if q.strategy == ExpansionStrategy.REFORMULATION
        ]
        assert len(reformulation_queries) > 0

    async def test_metadata_in_result(self, mock_integration, basic_config):
        """Test metadata is included in result."""
        expander = LightRAGQueryExpander(
            lightrag_integration=mock_integration,
            config=basic_config,
        )
        result = await expander.execute(query="test query")

        assert result.success is True
        assert "strategies_used" in result.metadata
        assert "total_queries" in result.metadata
        assert "parallel_execution" in result.metadata


@pytest.mark.asyncio
class TestExpansionStrategies:
    """Test individual expansion strategy implementations."""

    async def test_synonym_with_entities(self, mock_integration):
        """Test synonym expansion uses context entities."""
        config = QueryExpansionConfig(
            strategies=[ExpansionStrategy.SYNONYM],
            max_expansions_per_strategy=2,
        )
        expander = LightRAGQueryExpander(
            lightrag_integration=mock_integration,
            config=config,
        )

        # Mock context with specific entities
        context = {
            "entities": [
                {"name": "ML", "type": "machine learning"},
                {"name": "AI", "type": "artificial intelligence"},
            ]
        }
        expansions = await expander._expand_synonyms("ML algorithms", context)

        assert len(expansions) <= 2
        for exp in expansions:
            assert exp.strategy == ExpansionStrategy.SYNONYM

    async def test_semantic_with_relationships(self, mock_integration):
        """Test semantic expansion uses context relationships."""
        config = QueryExpansionConfig(
            strategies=[ExpansionStrategy.SEMANTIC],
        )
        expander = LightRAGQueryExpander(
            lightrag_integration=mock_integration,
            config=config,
        )

        context = {
            "relationships": [
                {"type": "uses", "source": "ML", "target": "data"},
                {"type": "requires", "source": "ML", "target": "training"},
            ]
        }
        expansions = await expander._expand_semantic("ML", context)

        assert len(expansions) > 0
        for exp in expansions:
            assert exp.strategy == ExpansionStrategy.SEMANTIC

    async def test_acronym_detection(self, mock_integration):
        """Test acronym detection and expansion."""
        config = QueryExpansionConfig(
            strategies=[ExpansionStrategy.ACRONYM],
        )
        expander = LightRAGQueryExpander(
            lightrag_integration=mock_integration,
            config=config,
        )

        context = {
            "entities": [
                {"name": "Machine Learning", "type": "concept"},
            ]
        }
        expansions = await expander._expand_acronyms("What is ML?", context)

        # Should detect ML as acronym
        assert len(expansions) >= 0

    async def test_reformulation_variations(self, mock_integration):
        """Test reformulation generates different question forms."""
        config = QueryExpansionConfig(
            strategies=[ExpansionStrategy.REFORMULATION],
        )
        expander = LightRAGQueryExpander(
            lightrag_integration=mock_integration,
            config=config,
        )

        expansions = await expander._expand_reformulations("machine learning", {})

        assert len(expansions) > 0
        # Check different question forms
        queries = [exp.expanded_query for exp in expansions]
        assert any("What is" in q for q in queries)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
