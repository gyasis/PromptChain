#!/usr/bin/env python3
"""
Comprehensive Test Suite for Intelligent Search Variant Generation

Tests the PromptChain-based search variant generation system including:
- TopicDecompositionChain
- QueryGenerationChain  
- QueryOptimizationChain
- TopicComplexityAnalyzer
- SearchVariantGenerator
- Integration with LiteratureSearchAgent
"""

import asyncio
import json
import pytest
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from research_agent.core.search_variant_generator import (
    SearchVariantGenerator,
    TopicDecompositionChain,
    QueryGenerationChain,
    QueryOptimizationChain,
    TopicComplexityAnalyzer,
    TopicComponents,
    ComplexityMetrics,
    GeneratedQuery
)


class TestTopicComponents:
    """Test the TopicComponents dataclass"""
    
    def test_topic_components_creation(self):
        components = TopicComponents(
            primary_concept="early detection",
            methodology="machine learning",
            application_area="medical diagnosis",
            domain="biomedical engineering",
            temporal_focus="current",
            related_fields=["computer science", "neurology"],
            technical_aspects=["gait analysis", "sensors"],
            clinical_context="hospital setting",
            geographic_scope="global"
        )
        
        assert components.primary_concept == "early detection"
        assert components.methodology == "machine learning"
        assert len(components.related_fields) == 2
        assert components.clinical_context == "hospital setting"
    
    def test_topic_components_optional_fields(self):
        components = TopicComponents(
            primary_concept="analysis",
            methodology="statistical",
            application_area="research",
            domain="data science",
            temporal_focus="current",
            related_fields=["statistics"],
            technical_aspects=["modeling"]
        )
        
        assert components.clinical_context is None
        assert components.geographic_scope is None


class TestComplexityMetrics:
    """Test the ComplexityMetrics dataclass"""
    
    def test_complexity_metrics_creation(self):
        metrics = ComplexityMetrics(
            concept_count=3,
            methodology_count=2,
            application_count=1,
            scope_breadth=7,
            cross_disciplinary_elements=2,
            temporal_span=5,
            technical_depth=8
        )
        
        assert metrics.concept_count == 3
        assert metrics.scope_breadth == 7
        assert metrics.technical_depth == 8


class TestGeneratedQuery:
    """Test the GeneratedQuery dataclass"""
    
    def test_generated_query_creation(self):
        query = GeneratedQuery(
            query="machine learning gait analysis",
            category="methodological",
            database_preference="arxiv",
            confidence_score=0.85,
            reasoning="Focus on ML methods for gait analysis"
        )
        
        assert query.query == "machine learning gait analysis"
        assert query.category == "methodological"
        assert query.database_preference == "arxiv"
        assert query.confidence_score == 0.85


class TestTopicComplexityAnalyzer:
    """Test the TopicComplexityAnalyzer PromptChain"""
    
    @pytest.fixture
    def analyzer(self):
        return TopicComplexityAnalyzer(model="openai/gpt-4")
    
    @pytest.fixture
    def mock_components(self):
        return TopicComponents(
            primary_concept="early detection",
            methodology="machine learning",
            application_area="medical diagnosis",
            domain="neurology",
            temporal_focus="current",
            related_fields=["computer science", "biomedical engineering"],
            technical_aspects=["gait analysis", "sensor data", "pattern recognition"]
        )
    
    @pytest.mark.asyncio
    async def test_analyze_complexity_success(self, analyzer, mock_components):
        """Test successful complexity analysis"""
        mock_response = {
            "concept_count": 3,
            "methodology_count": 2,
            "application_count": 1,
            "scope_breadth": 7,
            "cross_disciplinary_elements": 2,
            "temporal_span": 5,
            "technical_depth": 8
        }
        
        with patch.object(analyzer, 'process_prompt', return_value=mock_response) as mock_process:
            result = await analyzer.analyze_complexity("test topic", mock_components)
            
            assert isinstance(result, ComplexityMetrics)
            assert result.concept_count == 3
            assert result.scope_breadth == 7
            assert result.technical_depth == 8
            mock_process.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_analyze_complexity_json_string_response(self, analyzer, mock_components):
        """Test complexity analysis with JSON string response"""
        mock_response = json.dumps({
            "concept_count": 2,
            "methodology_count": 1,
            "application_count": 1,
            "scope_breadth": 5,
            "cross_disciplinary_elements": 1,
            "temporal_span": 3,
            "technical_depth": 4
        })
        
        with patch.object(analyzer, 'process_prompt', return_value=mock_response):
            result = await analyzer.analyze_complexity("test topic", mock_components)
            
            assert isinstance(result, ComplexityMetrics)
            assert result.concept_count == 2
            assert result.scope_breadth == 5
    
    @pytest.mark.asyncio
    async def test_analyze_complexity_error_fallback(self, analyzer, mock_components):
        """Test complexity analysis error handling with fallback"""
        with patch.object(analyzer, 'process_prompt', side_effect=Exception("API Error")):
            result = await analyzer.analyze_complexity("test topic", mock_components)
            
            # Should return default metrics on error
            assert isinstance(result, ComplexityMetrics)
            assert result.concept_count == 2  # Default fallback value
            assert result.scope_breadth == 3   # Default fallback value


class TestTopicDecompositionChain:
    """Test the TopicDecompositionChain PromptChain"""
    
    @pytest.fixture
    def decomposition_chain(self):
        return TopicDecompositionChain(model="openai/gpt-4")
    
    @pytest.mark.asyncio
    async def test_decompose_topic_success(self, decomposition_chain):
        """Test successful topic decomposition"""
        mock_response = {
            "primary_concept": "early detection",
            "methodology": "machine learning",
            "application_area": "medical diagnosis", 
            "domain": "neurology",
            "temporal_focus": "current",
            "related_fields": ["computer science", "biomedical engineering"],
            "technical_aspects": ["gait analysis", "sensors"],
            "clinical_context": "hospital setting",
            "geographic_scope": null
        }
        
        with patch.object(decomposition_chain, 'process_prompt', return_value=mock_response):
            result = await decomposition_chain.decompose_topic("early detection of neurological diseases")
            
            assert isinstance(result, TopicComponents)
            assert result.primary_concept == "early detection"
            assert result.methodology == "machine learning"
            assert result.clinical_context == "hospital setting"
            assert result.geographic_scope is None  # null should convert to None
    
    @pytest.mark.asyncio
    async def test_decompose_topic_json_string_response(self, decomposition_chain):
        """Test topic decomposition with JSON string response"""
        mock_response = json.dumps({
            "primary_concept": "analysis",
            "methodology": "statistical methods",
            "application_area": "research",
            "domain": "data science",
            "temporal_focus": "current",
            "related_fields": ["statistics", "mathematics"],
            "technical_aspects": ["modeling", "algorithms"],
            "clinical_context": "null",
            "geographic_scope": "null"
        })
        
        with patch.object(decomposition_chain, 'process_prompt', return_value=mock_response):
            result = await decomposition_chain.decompose_topic("statistical analysis")
            
            assert isinstance(result, TopicComponents)
            assert result.primary_concept == "analysis"
            assert result.clinical_context is None
            assert result.geographic_scope is None
    
    @pytest.mark.asyncio
    async def test_decompose_topic_error_fallback(self, decomposition_chain):
        """Test topic decomposition error handling with fallback"""
        with patch.object(decomposition_chain, 'process_prompt', side_effect=Exception("API Error")):
            result = await decomposition_chain.decompose_topic("machine learning")
            
            # Should return fallback components
            assert isinstance(result, TopicComponents)
            assert result.primary_concept == "machine"  # First word of topic
            assert result.methodology == "analysis"
            assert result.domain == "interdisciplinary"


class TestQueryGenerationChain:
    """Test the QueryGenerationChain PromptChain"""
    
    @pytest.fixture
    def generation_chain(self):
        return QueryGenerationChain(model="openai/gpt-4")
    
    @pytest.fixture
    def mock_components(self):
        return TopicComponents(
            primary_concept="early detection",
            methodology="machine learning",
            application_area="medical diagnosis",
            domain="neurology",
            temporal_focus="current",
            related_fields=["computer science"],
            technical_aspects=["gait analysis"]
        )
    
    @pytest.mark.asyncio
    async def test_generate_queries_success(self, generation_chain, mock_components):
        """Test successful query generation"""
        mock_response = [
            {
                "query": "machine learning early detection neurological diseases",
                "category": "methodological",
                "database_preference": "arxiv",
                "confidence_score": 0.9,
                "reasoning": "ML methods for early detection"
            },
            {
                "query": "gait analysis clinical diagnosis neurology",
                "category": "application",
                "database_preference": "pubmed",
                "confidence_score": 0.85,
                "reasoning": "Clinical applications of gait analysis"
            }
        ]
        
        with patch.object(generation_chain, 'process_prompt', return_value=mock_response):
            result = await generation_chain.generate_queries("test topic", mock_components, 2)
            
            assert len(result) == 2
            assert all(isinstance(q, GeneratedQuery) for q in result)
            assert result[0].query == "machine learning early detection neurological diseases"
            assert result[0].category == "methodological"
            assert result[1].database_preference == "pubmed"
    
    @pytest.mark.asyncio
    async def test_generate_queries_json_string_response(self, generation_chain, mock_components):
        """Test query generation with JSON string response"""
        mock_response = json.dumps([
            {
                "query": "statistical analysis methods",
                "category": "methodological",
                "database_preference": "general",
                "confidence_score": 0.8,
                "reasoning": "Statistical methodology focus"
            }
        ])
        
        with patch.object(generation_chain, 'process_prompt', return_value=mock_response):
            result = await generation_chain.generate_queries("test topic", mock_components, 1)
            
            assert len(result) == 1
            assert isinstance(result[0], GeneratedQuery)
            assert result[0].query == "statistical analysis methods"
    
    @pytest.mark.asyncio
    async def test_generate_queries_error_fallback(self, generation_chain, mock_components):
        """Test query generation error handling with fallback"""
        with patch.object(generation_chain, 'process_prompt', side_effect=Exception("API Error")):
            result = await generation_chain.generate_queries("machine learning", mock_components, 3)
            
            # Should return fallback queries (limited to target count)
            assert len(result) == 2  # Only 2 fallback queries available
            assert all(isinstance(q, GeneratedQuery) for q in result)
            assert result[0].query == "machine learning"
            assert result[1].query == '"machine learning"'


class TestQueryOptimizationChain:
    """Test the QueryOptimizationChain PromptChain"""
    
    @pytest.fixture
    def optimization_chain(self):
        return QueryOptimizationChain(model="openai/gpt-4")
    
    @pytest.fixture
    def mock_queries(self):
        return [
            GeneratedQuery(
                query="machine learning gait analysis",
                category="methodological",
                database_preference="arxiv",
                confidence_score=0.9,
                reasoning="ML methods"
            ),
            GeneratedQuery(
                query="clinical gait assessment",
                category="application",
                database_preference="pubmed", 
                confidence_score=0.85,
                reasoning="Clinical applications"
            ),
            GeneratedQuery(
                query="neurological disease detection",
                category="application",
                database_preference="general",
                confidence_score=0.8,
                reasoning="General disease detection"
            )
        ]
    
    @pytest.mark.asyncio
    async def test_optimize_queries_success(self, optimization_chain, mock_queries):
        """Test successful query optimization"""
        mock_response = {
            "arxiv": [
                "machine learning AND gait analysis cat:cs.LG",
                "neurological disease detection cat:cs.AI"
            ],
            "pubmed": [
                "clinical gait assessment[Title/Abstract] AND neurology[MeSH Terms]",
                "neurological disease detection[Title/Abstract]"
            ],
            "sci_hub": [
                "machine learning gait analysis neurology",
                "clinical gait assessment neurological diseases"
            ]
        }
        
        with patch.object(optimization_chain, 'process_prompt', return_value=mock_response):
            result = await optimization_chain.optimize_queries(mock_queries)
            
            assert isinstance(result, dict)
            assert "arxiv" in result
            assert "pubmed" in result
            assert "sci_hub" in result
            assert len(result["arxiv"]) == 2
            assert "cat:cs.LG" in result["arxiv"][0]
            assert "[MeSH Terms]" in result["pubmed"][0]
    
    @pytest.mark.asyncio
    async def test_optimize_queries_json_string_response(self, optimization_chain, mock_queries):
        """Test query optimization with JSON string response"""
        mock_response = json.dumps({
            "arxiv": ["optimized arxiv query"],
            "pubmed": ["optimized pubmed query"],
            "sci_hub": ["optimized scihub query"]
        })
        
        with patch.object(optimization_chain, 'process_prompt', return_value=mock_response):
            result = await optimization_chain.optimize_queries(mock_queries)
            
            assert isinstance(result, dict)
            assert result["arxiv"] == ["optimized arxiv query"]
            assert result["pubmed"] == ["optimized pubmed query"]
    
    @pytest.mark.asyncio
    async def test_optimize_queries_error_fallback(self, optimization_chain, mock_queries):
        """Test query optimization error handling with fallback"""
        with patch.object(optimization_chain, 'process_prompt', side_effect=Exception("API Error")):
            result = await optimization_chain.optimize_queries(mock_queries)
            
            # Should return unoptimized queries organized by database preference
            assert isinstance(result, dict)
            assert "arxiv" in result
            assert "pubmed" in result
            assert "sci_hub" in result
            # Should contain original queries distributed by preference
            assert len(result["arxiv"]) >= 1  # At least the arxiv-preferred query
            assert len(result["pubmed"]) >= 1  # At least the pubmed-preferred query


class TestSearchVariantGenerator:
    """Test the main SearchVariantGenerator class"""
    
    @pytest.fixture
    def generator(self):
        return SearchVariantGenerator(
            model="openai/gpt-4",
            min_queries=4,
            max_queries=20,
            enable_adaptive_spanning=True,
            enable_query_optimization=True
        )
    
    @pytest.fixture
    def mock_topic_components(self):
        return TopicComponents(
            primary_concept="early detection",
            methodology="machine learning",
            application_area="medical diagnosis",
            domain="neurology",
            temporal_focus="current",
            related_fields=["computer science"],
            technical_aspects=["gait analysis"]
        )
    
    @pytest.fixture
    def mock_complexity_metrics(self):
        return ComplexityMetrics(
            concept_count=3,
            methodology_count=2,
            application_count=1,
            scope_breadth=7,
            cross_disciplinary_elements=2,
            temporal_span=5,
            technical_depth=8
        )
    
    def test_calculate_adaptive_count(self, generator, mock_complexity_metrics):
        """Test adaptive query count calculation"""
        count = generator._calculate_adaptive_count(mock_complexity_metrics)
        
        # Should be between min and max
        assert generator.min_queries <= count <= generator.max_queries
        # Should be influenced by complexity
        assert count > generator.min_queries  # Given high complexity scores
    
    def test_calculate_adaptive_count_disabled(self, mock_complexity_metrics):
        """Test adaptive count calculation when disabled"""
        generator = SearchVariantGenerator(enable_adaptive_spanning=False)
        count = generator._calculate_adaptive_count(mock_complexity_metrics)
        
        assert count == 8  # Fixed count when adaptive spanning disabled
    
    @pytest.mark.asyncio
    async def test_generate_search_variants_success(self, generator, mock_topic_components, mock_complexity_metrics):
        """Test successful search variant generation"""
        # Mock the PromptChain components
        with patch.object(generator.decomposition_chain, 'decompose_topic', return_value=mock_topic_components), \
             patch.object(generator.complexity_analyzer, 'analyze_complexity', return_value=mock_complexity_metrics), \
             patch.object(generator.generation_chain, 'generate_queries') as mock_gen, \
             patch.object(generator.optimization_chain, 'optimize_queries') as mock_opt:
            
            # Setup mock responses
            mock_generated_queries = [
                GeneratedQuery("query 1", "methodological", "arxiv", 0.9, "reason 1"),
                GeneratedQuery("query 2", "application", "pubmed", 0.85, "reason 2")
            ]
            mock_gen.return_value = mock_generated_queries
            
            mock_optimized = {
                "arxiv": ["optimized query 1"],
                "pubmed": ["optimized query 2"],
                "sci_hub": ["optimized query 3"]
            }
            mock_opt.return_value = mock_optimized
            
            # Execute test
            result = await generator.generate_search_variants("test topic")
            
            # Verify results
            assert isinstance(result, dict)
            assert "queries" in result
            assert "metadata" in result
            assert "generated_queries_details" in result
            assert "performance_metrics" in result
            
            # Check queries structure
            assert result["queries"] == mock_optimized
            
            # Check metadata
            metadata = result["metadata"]
            assert metadata["original_topic"] == "test topic"
            assert metadata["actual_query_count"] == 2
            assert metadata["adaptive_spanning_enabled"] is True
            assert metadata["query_optimization_enabled"] is True
            
            # Verify all chains were called
            generator.decomposition_chain.decompose_topic.assert_called_once_with("test topic")
            generator.complexity_analyzer.analyze_complexity.assert_called_once()
            mock_gen.assert_called_once()
            mock_opt.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_search_variants_optimization_disabled(self, mock_topic_components, mock_complexity_metrics):
        """Test search variant generation with optimization disabled"""
        generator = SearchVariantGenerator(enable_query_optimization=False)
        
        with patch.object(generator.decomposition_chain, 'decompose_topic', return_value=mock_topic_components), \
             patch.object(generator.complexity_analyzer, 'analyze_complexity', return_value=mock_complexity_metrics), \
             patch.object(generator.generation_chain, 'generate_queries') as mock_gen:
            
            mock_generated_queries = [
                GeneratedQuery("query 1", "methodological", "arxiv", 0.9, "reason 1"),
                GeneratedQuery("query 2", "application", "pubmed", 0.85, "reason 2")
            ]
            mock_gen.return_value = mock_generated_queries
            
            result = await generator.generate_search_variants("test topic")
            
            # Should have general queries instead of optimized
            assert "general" in result["queries"]
            assert result["queries"]["general"] == ["query 1", "query 2"]
            assert result["metadata"]["query_optimization_enabled"] is False
    
    @pytest.mark.asyncio
    async def test_generate_search_variants_error_fallback(self, generator):
        """Test search variant generation error handling with fallback"""
        # Mock all chains to raise errors
        with patch.object(generator.decomposition_chain, 'decompose_topic', side_effect=Exception("API Error")):
            result = await generator.generate_search_variants("test topic")
            
            # Should return fallback results
            assert isinstance(result, dict)
            assert "queries" in result
            assert "metadata" in result
            assert result["metadata"]["fallback_used"] is True
            assert "general" in result["queries"]
            
            # Should update metrics
            assert generator.generation_metrics["fallback_usage"] == 1
    
    def test_get_generation_metrics(self, generator):
        """Test getting generation metrics"""
        metrics = generator.get_generation_metrics()
        
        assert isinstance(metrics, dict)
        assert "total_generations" in metrics
        assert "successful_generations" in metrics
        assert "average_query_count" in metrics
        assert "fallback_usage" in metrics
    
    def test_reset_metrics(self, generator):
        """Test resetting generation metrics"""
        # Set some metrics
        generator.generation_metrics["total_generations"] = 5
        generator.generation_metrics["successful_generations"] = 4
        
        # Reset
        generator.reset_metrics()
        
        # Should be back to defaults
        assert generator.generation_metrics["total_generations"] == 0
        assert generator.generation_metrics["successful_generations"] == 0


class TestIntegrationWithLiteratureSearchAgent:
    """Test integration with LiteratureSearchAgent"""
    
    @pytest.mark.asyncio
    async def test_literature_searcher_integration(self):
        """Test that LiteratureSearchAgent can use SearchVariantGenerator"""
        # This is a high-level integration test
        from research_agent.agents.literature_searcher import LiteratureSearchAgent
        from research_agent.core.config import ResearchConfig
        
        # Mock config with search_variants enabled
        mock_config = {
            'literature_search': {
                'search_variants': {
                    'enabled': True,
                    'model': 'openai/gpt-4',
                    'min_queries': 4,
                    'max_queries': 10,
                    'adaptive_spanning': True,
                    'query_optimization': True
                }
            }
        }
        
        # Create agent with mocked dependencies
        agent = LiteratureSearchAgent(mock_config)
        
        # Mock the SearchVariantGenerator to avoid actual API calls
        mock_result = {
            'queries': {
                'arxiv': ['arxiv query 1', 'arxiv query 2'],
                'pubmed': ['pubmed query 1'],
                'general': ['general query 1']
            },
            'metadata': {
                'original_topic': 'test topic',
                'actual_query_count': 4,
                'generation_time': 1.5,
                'model_used': 'openai/gpt-4',
                'adaptive_spanning_enabled': True,
                'query_optimization_enabled': True
            }
        }
        
        with patch('research_agent.core.search_variant_generator.SearchVariantGenerator') as MockGenerator:
            mock_instance = Mock()
            mock_instance.generate_search_variants = AsyncMock(return_value=mock_result)
            MockGenerator.return_value = mock_instance
            
            # Test the integration
            queries = await agent.generate_search_queries("machine learning gait analysis")
            
            # Verify results
            assert isinstance(queries, list)
            assert len(queries) > 0
            # Should contain queries from all databases
            expected_queries = ['arxiv query 1', 'arxiv query 2', 'pubmed query 1', 'general query 1']
            assert all(q in queries for q in expected_queries)


def run_comprehensive_test():
    """Run all tests and generate a report"""
    print("🧪 Running Comprehensive Test Suite for Intelligent Search Variants")
    print("=" * 80)
    
    # Test configuration
    test_results = {
        'total_tests': 0,
        'passed_tests': 0,
        'failed_tests': 0,
        'errors': []
    }
    
    try:
        # Run pytest programmatically
        import pytest
        
        # Run tests and capture results
        result = pytest.main([
            __file__,
            '-v',
            '--tb=short',
            '--disable-warnings'
        ])
        
        if result == 0:
            print("✅ All tests passed!")
            test_results['passed_tests'] = 'all'
        else:
            print(f"❌ Some tests failed (exit code: {result})")
            test_results['failed_tests'] = result
        
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        test_results['errors'].append(str(e))
    
    print("\n📊 Test Summary:")
    print(f"   Result: {'PASSED' if test_results.get('passed_tests') == 'all' else 'FAILED'}")
    if test_results['errors']:
        print(f"   Errors: {len(test_results['errors'])}")
        for error in test_results['errors']:
            print(f"     - {error}")
    
    return test_results


if __name__ == "__main__":
    # Run comprehensive test suite
    asyncio.run(run_comprehensive_test())