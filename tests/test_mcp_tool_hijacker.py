"""
Comprehensive test suite for MCP Tool Hijacker functionality.

This module tests all aspects of the MCP Tool Hijacker including:
- Direct tool execution without LLM overhead
- Parameter management (static, dynamic, transformation)
- Schema validation
- Connection management
- Error handling
- Performance tracking
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from typing import Dict, Any, List

# Import the modules to test
from promptchain.utils.mcp_tool_hijacker import (
    MCPToolHijacker,
    ToolNotFoundError,
    ToolExecutionError,
    create_temperature_clamped_hijacker,
    create_production_hijacker
)
from promptchain.utils.tool_parameter_manager import (
    ToolParameterManager,
    ParameterValidationError,
    ParameterTransformationError,
    CommonTransformers,
    CommonValidators
)
from promptchain.utils.mcp_connection_manager import (
    MCPConnectionManager,
    MCPConnectionError,
    MCPToolDiscoveryError
)
from promptchain.utils.mcp_schema_validator import (
    MCPSchemaValidator,
    SchemaValidationError
)


# Fixtures for testing
@pytest.fixture
def mock_mcp_config():
    """Mock MCP server configuration."""
    return [
        {
            "id": "test_server",
            "type": "stdio",
            "command": "/usr/bin/test-mcp-server",
            "args": ["--test"]
        }
    ]


@pytest.fixture
def mock_tool_schema():
    """Mock tool schema for testing."""
    return {
        "type": "function",
        "function": {
            "name": "mcp_test_server_echo",
            "description": "Test echo tool",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Message to echo"
                    },
                    "temperature": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0
                    }
                },
                "required": ["message"]
            }
        }
    }


@pytest.fixture
async def mock_hijacker(mock_mcp_config):
    """Create a mock hijacker for testing."""
    hijacker = MCPToolHijacker(
        mcp_servers_config=mock_mcp_config,
        verbose=True,
        parameter_validation=True
    )
    
    # Mock the connection manager
    hijacker.connection_manager = AsyncMock(spec=MCPConnectionManager)
    hijacker.connection_manager.is_connected = True
    hijacker.connection_manager.available_tools = ["mcp_test_server_echo", "mcp_test_server_calculate"]
    hijacker.connection_manager.execute_tool = AsyncMock(return_value="Success")
    
    # Set as connected
    hijacker._connected = True
    hijacker._init_performance_tracking()
    
    return hijacker


class TestMCPToolHijacker:
    """Test suite for MCPToolHijacker class."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, mock_mcp_config):
        """Test hijacker initialization."""
        hijacker = MCPToolHijacker(
            mcp_servers_config=mock_mcp_config,
            verbose=False,
            connection_timeout=60.0,
            max_retries=5,
            parameter_validation=True
        )
        
        assert hijacker.mcp_servers_config == mock_mcp_config
        assert hijacker.verbose == False
        assert hijacker.parameter_validation == True
        assert not hijacker.is_connected
        assert len(hijacker.pre_execution_hooks) == 0
        assert len(hijacker.post_execution_hooks) == 0
    
    @pytest.mark.asyncio
    async def test_connect_success(self, mock_mcp_config):
        """Test successful connection to MCP servers."""
        hijacker = MCPToolHijacker(mock_mcp_config)
        
        # Mock the connection manager's connect method
        with patch.object(hijacker.connection_manager, 'connect', new_callable=AsyncMock) as mock_connect:
            with patch.object(hijacker.connection_manager, 'available_tools', ["tool1", "tool2"]):
                await hijacker.connect()
                
                mock_connect.assert_called_once()
                assert hijacker.is_connected
                assert len(hijacker._performance_stats) == 2
    
    @pytest.mark.asyncio
    async def test_connect_failure(self, mock_mcp_config):
        """Test connection failure handling."""
        hijacker = MCPToolHijacker(mock_mcp_config)
        
        # Mock connection failure
        with patch.object(
            hijacker.connection_manager, 
            'connect', 
            new_callable=AsyncMock,
            side_effect=MCPConnectionError("Connection failed")
        ):
            with pytest.raises(MCPConnectionError):
                await hijacker.connect()
            
            assert not hijacker.is_connected
    
    @pytest.mark.asyncio
    async def test_direct_tool_execution(self, mock_hijacker):
        """Test direct tool execution without LLM."""
        result = await mock_hijacker.call_tool(
            "mcp_test_server_echo",
            message="Hello",
            temperature=0.5
        )
        
        assert result == "Success"
        mock_hijacker.connection_manager.execute_tool.assert_called_once()
        
        # Check performance stats
        stats = mock_hijacker._performance_stats["mcp_test_server_echo"]
        assert stats["call_count"] == 1
        assert stats["error_count"] == 0
        assert stats["success_rate"] == 1.0
    
    @pytest.mark.asyncio
    async def test_tool_not_found(self, mock_hijacker):
        """Test error when tool doesn't exist."""
        with pytest.raises(ToolNotFoundError) as exc_info:
            await mock_hijacker.call_tool("non_existent_tool", param="value")
        
        assert "non_existent_tool" in str(exc_info.value)
        assert "Available tools:" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_not_connected_error(self, mock_hijacker):
        """Test error when calling tool without connection."""
        mock_hijacker._connected = False
        
        with pytest.raises(ToolExecutionError) as exc_info:
            await mock_hijacker.call_tool("mcp_test_server_echo", message="test")
        
        assert "not connected" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_static_parameters(self, mock_hijacker):
        """Test static parameter management."""
        # Set static parameters
        mock_hijacker.set_static_params(
            "mcp_test_server_echo",
            temperature=0.7,
            model="default"
        )
        
        # Call tool with only required parameter
        await mock_hijacker.call_tool("mcp_test_server_echo", message="test")
        
        # Verify static params were included
        call_args = mock_hijacker.connection_manager.execute_tool.call_args
        assert call_args[0][0] == "mcp_test_server_echo"
        params = call_args[0][1]
        assert params.get("temperature") == 0.7
        assert params.get("model") == "default"
        assert params.get("message") == "test"
    
    @pytest.mark.asyncio
    async def test_parameter_transformation(self, mock_hijacker):
        """Test parameter transformation functionality."""
        # Add transformer to clamp temperature
        mock_hijacker.add_param_transformer(
            "mcp_test_server_echo",
            "temperature",
            CommonTransformers.clamp_float(0.0, 1.0)
        )
        
        # Call with out-of-range temperature
        await mock_hijacker.call_tool(
            "mcp_test_server_echo",
            message="test",
            temperature=1.5  # Should be clamped to 1.0
        )
        
        # Verify transformation was applied
        call_args = mock_hijacker.connection_manager.execute_tool.call_args
        params = call_args[0][1]
        assert params.get("temperature") == 1.0
    
    @pytest.mark.asyncio
    async def test_parameter_validation(self, mock_hijacker):
        """Test parameter validation."""
        # Add validator
        mock_hijacker.add_param_validator(
            "mcp_test_server_echo",
            "message",
            CommonValidators.is_non_empty_string()
        )
        
        # Test with invalid parameter
        with pytest.raises(ParameterValidationError):
            await mock_hijacker.call_tool(
                "mcp_test_server_echo",
                message=""  # Empty string should fail validation
            )
    
    @pytest.mark.asyncio
    async def test_batch_execution(self, mock_hijacker):
        """Test batch tool execution."""
        batch_calls = [
            {
                "tool_name": "mcp_test_server_echo",
                "params": {"message": "test1"}
            },
            {
                "tool_name": "mcp_test_server_echo",
                "params": {"message": "test2"}
            },
            {
                "tool_name": "mcp_test_server_calculate",
                "params": {"expression": "2+2"}
            }
        ]
        
        results = await mock_hijacker.call_tool_batch(batch_calls, max_concurrent=2)
        
        assert len(results) == 3
        assert all(r["success"] for r in results)
        assert results[0]["tool_name"] == "mcp_test_server_echo"
        assert results[2]["tool_name"] == "mcp_test_server_calculate"
    
    @pytest.mark.asyncio
    async def test_execution_hooks(self, mock_hijacker):
        """Test pre and post execution hooks."""
        pre_hook_called = False
        post_hook_called = False
        
        def pre_hook(tool_name, params):
            nonlocal pre_hook_called
            pre_hook_called = True
            assert tool_name == "mcp_test_server_echo"
            assert "message" in params
        
        def post_hook(tool_name, params, result, execution_time):
            nonlocal post_hook_called
            post_hook_called = True
            assert result == "Success"
            assert execution_time > 0
        
        mock_hijacker.add_execution_hook(pre_hook, "pre")
        mock_hijacker.add_execution_hook(post_hook, "post")
        
        await mock_hijacker.call_tool("mcp_test_server_echo", message="test")
        
        assert pre_hook_called
        assert post_hook_called
    
    @pytest.mark.asyncio
    async def test_performance_tracking(self, mock_hijacker):
        """Test performance statistics tracking."""
        # Execute tool multiple times
        for i in range(5):
            await mock_hijacker.call_tool("mcp_test_server_echo", message=f"test{i}")
        
        # Force one failure
        mock_hijacker.connection_manager.execute_tool.side_effect = Exception("Test error")
        try:
            await mock_hijacker.call_tool("mcp_test_server_echo", message="fail")
        except:
            pass
        
        # Check performance stats
        stats = mock_hijacker.get_performance_stats("mcp_test_server_echo")
        assert stats["call_count"] == 6
        assert stats["error_count"] == 1
        assert stats["success_rate"] == 5/6
        assert stats["avg_time"] > 0
    
    @pytest.mark.asyncio
    async def test_context_manager(self, mock_mcp_config):
        """Test async context manager functionality."""
        hijacker = MCPToolHijacker(mock_mcp_config)
        
        with patch.object(hijacker, 'connect', new_callable=AsyncMock) as mock_connect:
            with patch.object(hijacker, 'disconnect', new_callable=AsyncMock) as mock_disconnect:
                async with hijacker as h:
                    assert h is hijacker
                    mock_connect.assert_called_once()
                
                mock_disconnect.assert_called_once()


class TestToolParameterManager:
    """Test suite for ToolParameterManager class."""
    
    def test_static_parameters(self):
        """Test static parameter management."""
        manager = ToolParameterManager()
        
        # Set static parameters
        manager.set_static_params("tool1", param1="value1", param2=42)
        
        # Get static parameters
        static = manager.get_static_params("tool1")
        assert static["param1"] == "value1"
        assert static["param2"] == 42
        
        # Remove parameter
        assert manager.remove_static_param("tool1", "param1")
        static = manager.get_static_params("tool1")
        assert "param1" not in static
        assert "param2" in static
    
    def test_parameter_merging(self):
        """Test parameter merging priority."""
        manager = ToolParameterManager()
        
        # Set different parameter sources
        manager.set_default_params("tool1", param1="default", param2="default")
        manager.set_static_params("tool1", param1="static")
        
        # Merge with dynamic params
        merged = manager.merge_params("tool1", param1="dynamic", param3="dynamic")
        
        # Check priority: dynamic > static > default
        assert merged["param1"] == "dynamic"
        assert merged["param2"] == "default"
        assert merged["param3"] == "dynamic"
    
    def test_parameter_transformation(self):
        """Test parameter transformation."""
        manager = ToolParameterManager()
        
        # Add transformer
        manager.add_transformer(
            "tool1",
            "temperature",
            lambda x: max(0.0, min(1.0, float(x)))
        )
        
        # Transform parameters
        params = {"temperature": 1.5, "other": "unchanged"}
        transformed = manager.transform_params("tool1", params)
        
        assert transformed["temperature"] == 1.0
        assert transformed["other"] == "unchanged"
    
    def test_parameter_validation(self):
        """Test parameter validation."""
        manager = ToolParameterManager()
        
        # Set required parameters
        manager.set_required_params("tool1", ["param1", "param2"])
        
        # Add validator
        manager.add_validator(
            "tool1",
            "param1",
            lambda x: isinstance(x, str) and len(x) > 0
        )
        
        # Test missing required parameter
        with pytest.raises(ParameterValidationError) as exc_info:
            manager.validate_params("tool1", {"param1": "valid"})
        assert "Missing required parameters" in str(exc_info.value)
        
        # Test invalid parameter
        with pytest.raises(ParameterValidationError) as exc_info:
            manager.validate_params("tool1", {"param1": "", "param2": "valid"})
        assert "Validation failed" in str(exc_info.value)
    
    def test_parameter_templates(self):
        """Test parameter template substitution."""
        manager = ToolParameterManager()
        
        # Set parameter template
        manager.set_parameter_template(
            "tool1",
            "message",
            "Hello {name}, the temperature is {temp}"
        )
        
        # Apply templates
        params = {"message": "placeholder"}
        template_vars = {"name": "World", "temp": "25°C"}
        result = manager.apply_templates("tool1", params, template_vars)
        
        assert result["message"] == "Hello World, the temperature is 25°C"
    
    def test_global_transformers(self):
        """Test global parameter transformers."""
        manager = ToolParameterManager()
        
        # Add global transformer
        manager.add_global_transformer(
            "temperature",
            CommonTransformers.clamp_float(0.0, 2.0)
        )
        
        # Transform parameters for any tool
        params = {"temperature": 3.0}
        transformed = manager.transform_params("any_tool", params)
        
        assert transformed["temperature"] == 2.0
    
    def test_complete_processing_pipeline(self):
        """Test complete parameter processing pipeline."""
        manager = ToolParameterManager()
        
        # Configure tool
        manager.set_default_params("tool1", temperature=0.5)
        manager.set_static_params("tool1", model="gpt-4")
        manager.set_required_params("tool1", ["message"])
        manager.add_transformer("tool1", "temperature", CommonTransformers.clamp_float(0.0, 1.0))
        manager.add_validator("tool1", "message", CommonValidators.is_non_empty_string())
        
        # Process parameters
        result = manager.process_params(
            "tool1",
            message="Hello",
            temperature=1.5  # Will be clamped
        )
        
        assert result["message"] == "Hello"
        assert result["temperature"] == 1.0
        assert result["model"] == "gpt-4"


class TestMCPSchemaValidator:
    """Test suite for MCPSchemaValidator class."""
    
    def test_basic_validation(self, mock_tool_schema):
        """Test basic schema validation."""
        validator = MCPSchemaValidator()
        
        # Valid parameters
        params = {"message": "test", "temperature": 0.5}
        validator.validate_parameters("test_tool", mock_tool_schema, params)  # Should not raise
        
        # Missing required parameter
        params = {"temperature": 0.5}
        with pytest.raises(SchemaValidationError) as exc_info:
            validator.validate_parameters("test_tool", mock_tool_schema, params)
        assert "Missing required parameters" in str(exc_info.value)
    
    def test_type_validation(self, mock_tool_schema):
        """Test parameter type validation."""
        validator = MCPSchemaValidator()
        
        # Wrong type for parameter
        params = {"message": 123, "temperature": 0.5}  # message should be string
        with pytest.raises(SchemaValidationError):
            validator.validate_parameters("test_tool", mock_tool_schema, params, strict=True)
    
    def test_range_validation(self, mock_tool_schema):
        """Test numeric range validation."""
        validator = MCPSchemaValidator()
        
        # Temperature out of range
        params = {"message": "test", "temperature": 1.5}  # max is 1.0
        with pytest.raises(SchemaValidationError) as exc_info:
            validator.validate_parameters("test_tool", mock_tool_schema, params)
        assert "must be <=" in str(exc_info.value)
    
    def test_schema_info_extraction(self, mock_tool_schema):
        """Test schema information extraction."""
        validator = MCPSchemaValidator()
        
        info = validator.get_schema_info(mock_tool_schema)
        
        assert info["has_schema"] == True
        assert info["type"] == "object"
        assert "message" in info["required_params"]
        assert "temperature" in info["optional_params"]
        assert info["properties"]["message"]["type"] == "string"
        assert info["properties"]["temperature"]["type"] == "number"
    
    def test_parameter_fix_suggestions(self, mock_tool_schema):
        """Test parameter fix suggestions."""
        validator = MCPSchemaValidator()
        
        # Parameters with multiple issues
        params = {
            "temperature": "not_a_number",
            "extra_param": "unexpected"
        }
        
        suggestions = validator.suggest_parameter_fixes("test_tool", mock_tool_schema, params)
        
        assert any("Add required parameter: message" in s for s in suggestions)
        assert any("Convert temperature to number" in s for s in suggestions)


class TestConvenienceFunctions:
    """Test convenience functions for creating hijackers."""
    
    def test_temperature_clamped_hijacker(self, mock_mcp_config):
        """Test temperature clamped hijacker creation."""
        hijacker = create_temperature_clamped_hijacker(mock_mcp_config, verbose=True)
        
        assert hijacker.verbose == True
        # Temperature should be clamped by global transformer
        # This would be tested more thoroughly with actual execution
    
    def test_production_hijacker(self, mock_mcp_config):
        """Test production hijacker creation."""
        hijacker = create_production_hijacker(mock_mcp_config, verbose=False)
        
        assert hijacker.verbose == False
        # Check production settings
        assert hijacker.connection_manager.connection_timeout == 60.0
        assert hijacker.connection_manager.max_retries == 5
        assert hijacker.parameter_validation == True


class TestIntegrationWithPromptChain:
    """Test integration with PromptChain class."""
    
    @pytest.mark.asyncio
    async def test_promptchain_hijacker_initialization(self, mock_mcp_config):
        """Test that PromptChain properly initializes hijacker."""
        from promptchain.utils.promptchaining import PromptChain
        
        # Create PromptChain with hijacker enabled
        chain = PromptChain(
            models=["openai/gpt-4"],
            instructions=["Process: {input}"],
            mcp_servers=mock_mcp_config,
            enable_mcp_hijacker=True,
            hijacker_config={
                "connection_timeout": 45.0,
                "max_retries": 3,
                "parameter_validation": True
            }
        )
        
        # Check hijacker is initialized
        assert chain.mcp_hijacker is not None
        assert isinstance(chain.mcp_hijacker, MCPToolHijacker)
        
        # Check configuration was applied
        # Note: This would need actual attribute access or mocking
    
    @pytest.mark.asyncio
    async def test_promptchain_context_manager_with_hijacker(self, mock_mcp_config):
        """Test that PromptChain context manager handles hijacker."""
        from promptchain.utils.promptchaining import PromptChain
        
        chain = PromptChain(
            models=["openai/gpt-4"],
            instructions=["Process: {input}"],
            mcp_servers=mock_mcp_config,
            enable_mcp_hijacker=True
        )
        
        # Mock the hijacker methods
        with patch.object(chain.mcp_hijacker, 'connect', new_callable=AsyncMock) as mock_connect:
            with patch.object(chain.mcp_hijacker, 'disconnect', new_callable=AsyncMock) as mock_disconnect:
                async with chain:
                    mock_connect.assert_called_once()
                
                mock_disconnect.assert_called_once()


# Run tests with pytest
if __name__ == "__main__":
    pytest.main([__file__, "-v"])