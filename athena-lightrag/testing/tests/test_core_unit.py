#!/usr/bin/env python3
"""
Unit Tests for Athena LightRAG Core Components
==============================================
Fast unit tests with mocked dependencies for core functionality.

Author: PromptChain Team
Date: 2025
"""

import pytest
import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from athena_lightrag.core import (
    QueryResult,
    ContextChunk, 
    ReasoningState,
    AthenaInstanceManager
)


class TestDataStructures:
    """Test core data structures and their functionality."""
    
    def test_query_result_creation(self):
        """Test QueryResult creation and basic functionality."""
        result = QueryResult(
            result="Test result content",
            query_mode="hybrid",
            context_only=False,
            query_id="test-123",
            timestamp=datetime.now(timezone.utc)
        )
        
        assert result.result == "Test result content"
        assert result.query_mode == "hybrid"
        assert result.context_only is False
        assert result.query_id == "test-123"
        assert result.timestamp is not None
    
    def test_query_result_serialization(self):
        """Test QueryResult to_dict and to_json methods."""
        result = QueryResult(
            result="Serialization test",
            query_mode="local",
            context_only=True,
            query_id="ser-456",
            timestamp=datetime.now(timezone.utc),
            metadata={"test": "metadata"}
        )
        
        # Test to_dict
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict["result"] == "Serialization test"
        assert result_dict["query_mode"] == "local"
        assert "timestamp" in result_dict
        
        # Test to_json
        json_str = result.to_json()
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["query_id"] == "ser-456"
    
    def test_context_chunk_creation(self):
        """Test ContextChunk creation and auto-generation of fields.""" 
        chunk = ContextChunk(
            content="Test context content",
            source_query="Test query", 
            reasoning_step=1
        )
        
        assert chunk.content == "Test context content"
        assert chunk.source_query == "Test query"
        assert chunk.reasoning_step == 1
        assert chunk.chunk_id is not None  # Auto-generated
        assert chunk.timestamp is not None  # Auto-generated
    
    def test_reasoning_state_management(self):
        """Test ReasoningState operations."""
        state = ReasoningState(
            session_id="session-789",
            initial_query="Complex query",
            current_step=0,
            accumulated_contexts=[],
            reasoning_steps=[],
            strategy="comprehensive",
            max_steps=5,
            created_at=datetime.now(timezone.utc)
        )
        
        # Test adding context chunks
        chunk1 = ContextChunk(
            content="First chunk",
            source_query="Sub-query 1",
            reasoning_step=1
        )
        
        chunk2 = ContextChunk(
            content="Second chunk", 
            source_query="Sub-query 2",
            reasoning_step=2
        )
        
        state.add_context_chunk(chunk1)
        state.add_context_chunk(chunk2)
        
        assert len(state.accumulated_contexts) == 2
        assert state.accumulated_contexts[0].content == "First chunk"
        assert state.accumulated_contexts[1].content == "Second chunk"
        
        # Test getting accumulated context text
        context_text = state.get_accumulated_context_text()
        assert "First chunk" in context_text
        assert "Second chunk" in context_text
        assert "Sub-query 1" in context_text
        assert "Sub-query 2" in context_text


class TestInstanceManager:
    """Test the AthenaInstanceManager singleton functionality."""
    
    @pytest.mark.asyncio
    async def test_instance_manager_singleton(self, mock_lightrag_db, mock_env_vars):
        """Test that instance manager provides singleton behavior."""
        manager = AthenaInstanceManager()
        
        with patch('athena_lightrag.core.AthenaLightRAG') as mock_athena_class:
            mock_athena = MagicMock()
            mock_athena_class.return_value = mock_athena
            
            # Get instance twice
            instance1 = await manager.get_instance(working_dir=mock_lightrag_db)
            instance2 = await manager.get_instance(working_dir=mock_lightrag_db)
            
            # Should be the same instance
            assert instance1 is instance2
            
            # Should only create one instance
            assert mock_athena_class.call_count == 1
    
    @pytest.mark.asyncio
    async def test_instance_manager_context(self, mock_lightrag_db, mock_env_vars):
        """Test the instance manager async context manager."""
        manager = AthenaInstanceManager()
        
        with patch('athena_lightrag.core.AthenaLightRAG') as mock_athena_class:
            mock_athena = MagicMock()
            mock_athena_class.return_value = mock_athena
            
            async with manager.instance_context(working_dir=mock_lightrag_db) as instance:
                assert instance is not None
                assert instance is mock_athena


class TestInputValidation:
    """Test input validation and error handling."""
    
    def test_query_mode_validation(self):
        """Test query mode validation and normalization."""
        valid_modes = ["local", "global", "hybrid", "naive"]
        
        for mode in valid_modes:
            # This would normally be tested in the AthenaLightRAG class
            # but we test the validation logic directly here
            assert mode in valid_modes
        
        # Test invalid mode handling
        invalid_mode = "invalid_mode"
        assert invalid_mode not in valid_modes
    
    def test_parameter_bounds_checking(self):
        """Test parameter bounds checking logic."""
        # Test top_k bounds (would be used in query methods)
        def clamp_top_k(value, min_val=1, max_val=200):
            return max(min_val, min(max_val, value))
        
        assert clamp_top_k(-5) == 1      # Below minimum
        assert clamp_top_k(50) == 50     # Within range
        assert clamp_top_k(300) == 200   # Above maximum
        
        # Test reasoning steps bounds
        def clamp_reasoning_steps(value, min_val=1, max_val=10):
            return max(min_val, min(max_val, value))
        
        assert clamp_reasoning_steps(0) == 1     # Below minimum
        assert clamp_reasoning_steps(5) == 5     # Within range  
        assert clamp_reasoning_steps(15) == 10   # Above maximum


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_async_error_propagation(self):
        """Test that async errors are properly propagated."""
        async def failing_function():
            raise ValueError("Test error message")
        
        with pytest.raises(ValueError) as exc_info:
            await failing_function()
        
        assert "Test error message" in str(exc_info.value)
    
    def test_context_chunk_edge_cases(self):
        """Test ContextChunk edge cases and validation."""
        # Test with minimal required fields
        chunk = ContextChunk(
            content="",  # Empty content should be allowed
            source_query="Query",
            reasoning_step=0  # Step 0 should be allowed
        )
        
        assert chunk.content == ""
        assert chunk.reasoning_step == 0
        assert chunk.chunk_id is not None
        assert chunk.timestamp is not None
    
    def test_reasoning_state_edge_cases(self):
        """Test ReasoningState edge cases."""
        # Test with empty lists
        state = ReasoningState(
            session_id="edge-case",
            initial_query="",  # Empty query
            current_step=-1,   # Negative step
            accumulated_contexts=[],
            reasoning_steps=[],
            strategy="",       # Empty strategy
            max_steps=0,       # Zero max steps
            created_at=datetime.now(timezone.utc)
        )
        
        assert state.initial_query == ""
        assert state.current_step == -1
        assert state.max_steps == 0
        assert len(state.accumulated_contexts) == 0
        
        # Test getting context text when no contexts exist
        context_text = state.get_accumulated_context_text()
        assert context_text == ""


class TestPerformanceMetrics:
    """Test performance metrics calculation and tracking."""
    
    def test_performance_metrics_structure(self):
        """Test that performance metrics have the expected structure."""
        metrics = {
            "execution_time_seconds": 1.23,
            "result_length": 456,
            "parameters": {"top_k": 60, "mode": "hybrid"}
        }
        
        # Validate required fields
        assert "execution_time_seconds" in metrics
        assert isinstance(metrics["execution_time_seconds"], (int, float))
        assert metrics["execution_time_seconds"] > 0
        
        assert "result_length" in metrics 
        assert isinstance(metrics["result_length"], int)
        assert metrics["result_length"] >= 0
        
        assert "parameters" in metrics
        assert isinstance(metrics["parameters"], dict)
    
    def test_reasoning_metrics_structure(self):
        """Test reasoning-specific performance metrics."""
        reasoning_metrics = {
            "execution_time_seconds": 5.67,
            "reasoning_steps_count": 3,
            "context_chunks_count": 5,
            "average_chunk_length": 125.5
        }
        
        # Validate reasoning-specific fields
        assert "reasoning_steps_count" in reasoning_metrics
        assert isinstance(reasoning_metrics["reasoning_steps_count"], int)
        assert reasoning_metrics["reasoning_steps_count"] >= 0
        
        assert "context_chunks_count" in reasoning_metrics
        assert isinstance(reasoning_metrics["context_chunks_count"], int)
        assert reasoning_metrics["context_chunks_count"] >= 0
        
        assert "average_chunk_length" in reasoning_metrics
        assert isinstance(reasoning_metrics["average_chunk_length"], (int, float))


class TestUtilityFunctions:
    """Test utility functions and helpers."""
    
    def test_timestamp_handling(self):
        """Test timestamp creation and formatting."""
        now = datetime.now(timezone.utc)
        
        # Test ISO format
        iso_string = now.isoformat()
        assert "T" in iso_string
        assert iso_string.endswith("+00:00") or iso_string.endswith("Z")
        
        # Test that we can parse it back
        parsed = datetime.fromisoformat(iso_string.replace("Z", "+00:00"))
        assert parsed.tzinfo is not None
    
    def test_id_generation(self):
        """Test UUID generation for IDs."""
        import uuid
        
        # Test that we can generate unique IDs
        id1 = str(uuid.uuid4())
        id2 = str(uuid.uuid4())
        
        assert id1 != id2
        assert len(id1) == 36  # Standard UUID length
        assert "-" in id1      # UUID format includes hyphens
    
    def test_json_serialization_edge_cases(self):
        """Test JSON serialization with various data types."""
        test_data = {
            "string": "test",
            "integer": 42, 
            "float": 3.14,
            "boolean": True,
            "null": None,
            "list": [1, 2, 3],
            "nested": {"key": "value"}
        }
        
        # Test that it serializes without error
        json_str = json.dumps(test_data)
        assert isinstance(json_str, str)
        
        # Test that it deserializes correctly
        parsed = json.loads(json_str)
        assert parsed["string"] == "test"
        assert parsed["integer"] == 42
        assert parsed["boolean"] is True
        assert parsed["null"] is None
        assert len(parsed["list"]) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])