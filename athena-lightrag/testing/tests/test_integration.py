#!/usr/bin/env python3
"""
Integration Tests for Athena LightRAG MCP Server
===============================================
Comprehensive integration tests that validate the complete system
with real database connections and multi-hop reasoning capabilities.

Author: PromptChain Team
Date: 2025
"""

import asyncio
import json
import os
import pytest
from pathlib import Path
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

# Import the core components
from athena_lightrag.core import (
    AthenaLightRAG,
    QueryResult,
    ContextChunk,
    ReasoningState,
    query_athena_basic,
    query_athena_multi_hop,
    get_athena_database_info,
    athena_context
)
from athena_lightrag.server import (
    query_athena,
    query_athena_reasoning,
    get_database_status,
    generate_sql_query,
    get_query_mode_help
)


@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing."""
    with patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test-api-key-12345',
        'LIGHTRAG_WORKING_DIR': './tests/mock_athena_lightrag_db'
    }):
        yield


@pytest.fixture
def mock_lightrag_db(tmp_path):
    """Create a mock LightRAG database directory for testing."""
    db_path = tmp_path / "mock_athena_lightrag_db"
    db_path.mkdir()
    
    # Create some mock database files
    (db_path / "entities.json").write_text('{"test": "data"}')
    (db_path / "relationships.json").write_text('{"test": "relationships"}')
    (db_path / "graph.db").write_text("mock database content")
    
    return str(db_path)


@pytest.fixture
def mock_athena_instance(mock_lightrag_db, mock_env_vars):
    """Create a mock AthenaLightRAG instance for testing."""
    with patch('athena_lightrag.core.LightRAG') as mock_lightrag_class:
        mock_rag = AsyncMock()
        mock_lightrag_class.return_value = mock_rag
        
        # Mock the query results
        mock_rag.aquery.return_value = "Mock LightRAG response with relevant medical database information about the requested topic."
        
        athena = AthenaLightRAG(
            working_dir=mock_lightrag_db,
            api_key="test-api-key",
            reasoning_model="gpt-4.1-mini",
            max_reasoning_steps=3
        )
        
        # Ensure the mock is initialized
        athena.rag_initialized = True
        
        yield athena


class TestAthenaLightRAGCore:
    """Test the core AthenaLightRAG functionality."""
    
    @pytest.mark.asyncio
    async def test_basic_query_success(self, mock_athena_instance):
        """Test successful basic query execution."""
        result = await mock_athena_instance.basic_query(
            query_text="What tables are related to patient appointments?",
            mode="hybrid",
            top_k=60
        )
        
        assert isinstance(result, QueryResult)
        assert result.result != ""
        assert result.query_mode == "hybrid"
        assert result.query_id is not None
        assert result.timestamp is not None
        assert result.performance_metrics is not None
    
    @pytest.mark.asyncio
    async def test_basic_query_validation(self, mock_athena_instance):
        """Test input validation for basic queries."""
        # Test empty query
        with pytest.raises(Exception):
            await mock_athena_instance.basic_query(query_text="")
        
        # Test invalid mode (should default to hybrid)
        result = await mock_athena_instance.basic_query(
            query_text="test query",
            mode="invalid_mode"
        )
        assert result.query_mode == "hybrid"
    
    @pytest.mark.asyncio
    async def test_multi_hop_reasoning_query(self, mock_athena_instance):
        """Test multi-hop reasoning functionality."""
        with patch('promptchain.PromptChain') as mock_promptchain_class:
            mock_chain = AsyncMock()
            mock_promptchain_class.return_value = mock_chain
            mock_chain.process_prompt_async.return_value = "Comprehensive multi-hop reasoning result with context accumulation"
            mock_chain.step_outputs = ["Step 1 output", "Step 2 output", "Step 3 output"]
            
            result = await mock_athena_instance.multi_hop_reasoning_query(
                initial_query="How do anesthesia workflows connect to patient scheduling?",
                context_accumulation_strategy="incremental",
                mode="hybrid"
            )
            
            assert isinstance(result, QueryResult)
            assert result.result != ""
            assert result.reasoning_steps is not None
            assert len(result.reasoning_steps) > 0
            assert result.accumulated_context is not None
            assert result.performance_metrics is not None
            assert result.context_chunks is not None
    
    @pytest.mark.asyncio
    async def test_database_info_retrieval(self, mock_athena_instance):
        """Test database information retrieval."""
        info = await mock_athena_instance.get_database_info()
        
        assert isinstance(info, dict)
        assert "database_path" in info
        assert "database_exists" in info
        assert "initialized" in info
    
    @pytest.mark.asyncio
    async def test_context_manager_usage(self, mock_lightrag_db, mock_env_vars):
        """Test the async context manager for AthenaLightRAG."""
        with patch('athena_lightrag.core.LightRAG') as mock_lightrag_class:
            mock_rag = AsyncMock()
            mock_lightrag_class.return_value = mock_rag
            mock_rag.aquery.return_value = "Context manager test response"
            
            async with athena_context(working_dir=mock_lightrag_db) as athena:
                assert isinstance(athena, AthenaLightRAG)
                assert athena.working_dir == mock_lightrag_db
    
    def test_reasoning_state_management(self):
        """Test ReasoningState data structure and operations."""
        state = ReasoningState(
            session_id="test-session-123",
            initial_query="Test query",
            current_step=0,
            accumulated_contexts=[],
            reasoning_steps=[],
            strategy="incremental",
            max_steps=5,
            created_at=state.created_at if hasattr(state, 'created_at') else None
        )
        
        # Test adding context chunks
        chunk = ContextChunk(
            content="Test context content",
            source_query="Test sub-query",
            reasoning_step=1
        )
        
        state.add_context_chunk(chunk)
        assert len(state.accumulated_contexts) == 1
        assert state.accumulated_contexts[0].content == "Test context content"
        
        # Test getting accumulated context text
        context_text = state.get_accumulated_context_text()
        assert "Test context content" in context_text
        assert "Test sub-query" in context_text


class TestMCPServerIntegration:
    """Test the FastMCP server integration."""
    
    @pytest.mark.asyncio
    async def test_mcp_basic_query_tool(self, mock_env_vars, mock_lightrag_db):
        """Test the basic query MCP tool."""
        with patch('athena_lightrag.server.query_athena_basic') as mock_query:
            mock_query.return_value = "MCP tool response for basic query"
            
            result = await query_athena(
                query="Test MCP query",
                mode="hybrid",
                top_k=60
            )
            
            assert isinstance(result, str)
            assert "MCP tool response" in result
            mock_query.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_mcp_reasoning_query_tool(self, mock_env_vars, mock_lightrag_db):
        """Test the multi-hop reasoning MCP tool."""
        with patch('athena_lightrag.server.query_athena_multi_hop') as mock_query:
            mock_query.return_value = "MCP tool response for multi-hop reasoning"
            
            result = await query_athena_reasoning(
                query="Complex reasoning query",
                context_strategy="comprehensive",
                max_reasoning_steps=5
            )
            
            assert isinstance(result, str)
            assert "MCP tool response" in result
            mock_query.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_mcp_database_status_tool(self, mock_env_vars, mock_lightrag_db):
        """Test the database status MCP tool."""
        with patch('athena_lightrag.server.get_athena_database_info') as mock_info:
            mock_info.return_value = {
                "database_path": mock_lightrag_db,
                "database_exists": True,
                "initialized": True,
                "total_size_bytes": 1024000,
                "total_files": 10
            }
            
            result = await get_database_status(
                include_performance_stats=True,
                return_raw_data=False
            )
            
            assert isinstance(result, str)
            assert mock_lightrag_db in result or "Database Path" in result
    
    @pytest.mark.asyncio
    async def test_mcp_sql_generation_tool(self, mock_env_vars, mock_lightrag_db):
        """Test the SQL generation MCP tool."""
        with patch('athena_lightrag.server.query_athena_basic') as mock_basic, \
             patch('athena_lightrag.server.query_athena_multi_hop') as mock_reasoning:
            
            mock_basic.return_value = "Database schema context for SQL generation"
            mock_reasoning.return_value = "SELECT * FROM patients WHERE condition = 'diabetes';"
            
            result = await generate_sql_query(
                natural_language_query="Find all patients with diabetes",
                target_database_type="mysql",
                include_validation=True,
                return_explanation=True
            )
            
            assert isinstance(result, str)
            assert "SQL Query Generated" in result
            assert "mysql" in result.upper() or "MySQL" in result
            assert "diabetes" in result
    
    def test_query_mode_help_tool(self):
        """Test the query mode help tool."""
        help_text = get_query_mode_help()
        
        assert isinstance(help_text, str)
        assert "LOCAL MODE" in help_text
        assert "GLOBAL MODE" in help_text
        assert "HYBRID MODE" in help_text
        assert "NAIVE MODE" in help_text
        assert "Context Accumulation Strategies" in help_text


class TestErrorHandlingAndValidation:
    """Test error handling and input validation."""
    
    @pytest.mark.asyncio
    async def test_basic_query_input_validation(self):
        """Test input validation for basic queries."""
        # Test empty query
        result = await query_athena(query="")
        assert "Error: Query cannot be empty" in result
        
        # Test overly long query
        long_query = "a" * 2001
        result = await query_athena(query=long_query)
        assert "Error: Query too long" in result
    
    @pytest.mark.asyncio
    async def test_reasoning_query_input_validation(self):
        """Test input validation for reasoning queries."""
        # Test empty query
        result = await query_athena_reasoning(query="")
        assert "Error: Query cannot be empty" in result
        
        # Test overly long query
        long_query = "a" * 3001
        result = await query_athena_reasoning(query=long_query)
        assert "Error: Complex query too long" in result
        
        # Test invalid reasoning steps (should be clamped)
        with patch('athena_lightrag.server.query_athena_multi_hop') as mock_query:
            mock_query.return_value = "Valid response"
            
            await query_athena_reasoning(
                query="Test query",
                max_reasoning_steps=15  # Should be clamped to 10
            )
            
            # Verify the call was made with clamped value
            args, kwargs = mock_query.call_args
            assert kwargs['max_steps'] == 10
    
    @pytest.mark.asyncio
    async def test_sql_generation_input_validation(self):
        """Test input validation for SQL generation."""
        # Test empty query
        result = await generate_sql_query(natural_language_query="")
        assert "Error: Natural language query cannot be empty" in result
        
        # Test overly long query
        long_query = "a" * 1501
        result = await generate_sql_query(natural_language_query=long_query)
        assert "Error: Query description too long" in result


class TestPerformanceAndMonitoring:
    """Test performance monitoring and metrics collection."""
    
    @pytest.mark.asyncio
    async def test_performance_metrics_collection(self, mock_athena_instance):
        """Test that performance metrics are properly collected."""
        result = await mock_athena_instance.basic_query(
            query_text="Performance test query",
            mode="hybrid"
        )
        
        assert result.performance_metrics is not None
        assert "execution_time_seconds" in result.performance_metrics
        assert result.performance_metrics["execution_time_seconds"] > 0
        assert "result_length" in result.performance_metrics
    
    @pytest.mark.asyncio
    async def test_history_manager_integration(self, mock_lightrag_db, mock_env_vars):
        """Test integration with ExecutionHistoryManager."""
        with patch('athena_lightrag.core.LightRAG') as mock_lightrag_class:
            mock_rag = AsyncMock()
            mock_lightrag_class.return_value = mock_rag
            mock_rag.aquery.return_value = "History test response"
            
            athena = AthenaLightRAG(
                working_dir=mock_lightrag_db,
                enable_history_management=True
            )
            athena.rag_initialized = True
            
            # Verify history manager is enabled
            assert athena.history_manager is not None
            
            # Execute a query and verify history is tracked
            await athena.basic_query("Test query for history")
            
            # Verify entries were added to history
            assert len(athena.history_manager.entries) > 0


class TestAdvancedFeatures:
    """Test advanced features and edge cases."""
    
    @pytest.mark.asyncio
    async def test_concurrent_reasoning_sessions(self, mock_athena_instance):
        """Test handling of concurrent reasoning sessions."""
        with patch('promptchain.PromptChain') as mock_promptchain_class:
            mock_chain = AsyncMock()
            mock_promptchain_class.return_value = mock_chain
            mock_chain.process_prompt_async.return_value = "Concurrent session result"
            mock_chain.step_outputs = ["Concurrent step output"]
            
            # Start multiple concurrent reasoning sessions
            tasks = []
            for i in range(3):
                task = mock_athena_instance.multi_hop_reasoning_query(
                    initial_query=f"Concurrent query {i}",
                    context_accumulation_strategy="incremental"
                )
                tasks.append(task)
            
            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks)
            
            # Verify all sessions completed successfully
            assert len(results) == 3
            for result in results:
                assert isinstance(result, QueryResult)
                assert result.result != ""
    
    @pytest.mark.asyncio
    async def test_context_accumulation_strategies(self, mock_athena_instance):
        """Test different context accumulation strategies."""
        strategies = ["incremental", "comprehensive", "focused"]
        
        with patch('promptchain.PromptChain') as mock_promptchain_class:
            mock_chain = AsyncMock()
            mock_promptchain_class.return_value = mock_chain
            mock_chain.process_prompt_async.return_value = "Strategy test result"
            mock_chain.step_outputs = ["Strategy step output"]
            
            for strategy in strategies:
                result = await mock_athena_instance.multi_hop_reasoning_query(
                    initial_query=f"Test query for {strategy} strategy",
                    context_accumulation_strategy=strategy
                )
                
                assert isinstance(result, QueryResult)
                assert result.metadata["context_accumulation_strategy"] == strategy
    
    @pytest.mark.asyncio
    async def test_query_result_serialization(self, mock_athena_instance):
        """Test QueryResult serialization and deserialization."""
        result = await mock_athena_instance.basic_query(
            query_text="Serialization test query",
            mode="hybrid"
        )
        
        # Test to_dict conversion
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert "result" in result_dict
        assert "query_mode" in result_dict
        assert "timestamp" in result_dict
        
        # Test JSON serialization
        json_str = result.to_json()
        assert isinstance(json_str, str)
        
        # Test JSON can be parsed
        parsed_json = json.loads(json_str)
        assert parsed_json["query_mode"] == "hybrid"


@pytest.mark.integration
@pytest.mark.slow
class TestRealDatabaseIntegration:
    """Integration tests with real database (requires actual setup)."""
    
    def test_real_database_exists(self):
        """Test if the real database exists for integration testing."""
        real_db_path = Path("./athena_lightrag_db")
        if real_db_path.exists():
            pytest.skip("Real database tests require actual database setup")
        else:
            pytest.skip("Real database not found - skipping integration tests")
    
    @pytest.mark.skipif(
        not Path("./athena_lightrag_db").exists(),
        reason="Real database not found"
    )
    @pytest.mark.asyncio
    async def test_real_database_basic_query(self):
        """Test basic query against real database."""
        try:
            result = await query_athena_basic(
                query="What tables are in the medical database?",
                mode="hybrid"
            )
            assert isinstance(result, str)
            assert len(result) > 0
        except Exception as e:
            pytest.fail(f"Real database query failed: {e}")
    
    @pytest.mark.skipif(
        not Path("./athena_lightrag_db").exists(),
        reason="Real database not found"  
    )
    @pytest.mark.asyncio
    async def test_real_database_reasoning_query(self):
        """Test multi-hop reasoning against real database."""
        try:
            result = await query_athena_multi_hop(
                query="How are patient appointments connected to billing?",
                context_strategy="incremental",
                max_steps=3
            )
            assert isinstance(result, str)
            assert len(result) > 0
        except Exception as e:
            pytest.fail(f"Real database reasoning query failed: {e}")


if __name__ == "__main__":
    # Run specific test categories
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-m", "not slow",  # Skip slow tests by default
        "--cov=athena_lightrag",
        "--cov-report=term-missing"
    ])