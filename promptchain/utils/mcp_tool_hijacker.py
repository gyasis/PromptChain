"""
MCP Tool Hijacker for PromptChain

This module provides direct MCP tool execution capabilities without requiring
LLM agent processing. It enables performance optimization for tool-heavy workflows
while maintaining compatibility with the existing PromptChain ecosystem.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Union, Callable
import json

from .mcp_connection_manager import MCPConnectionManager, MCPConnectionError, MCPToolDiscoveryError
from .tool_parameter_manager import (
    ToolParameterManager, 
    ParameterValidationError, 
    ParameterTransformationError,
    CommonTransformers,
    CommonValidators
)
from .step_chaining_manager import StepChainingManager
from .json_output_parser import JSONOutputParser, CommonExtractions


class ToolNotFoundError(Exception):
    """Raised when attempting to call a non-existent tool."""
    pass


class ToolExecutionError(Exception):
    """Raised when tool execution fails."""
    pass


class MCPToolHijacker:
    """
    Direct MCP tool execution without LLM agent processing.
    
    This class provides:
    - Direct tool execution bypassing LLM workflows
    - Static and dynamic parameter management
    - Parameter validation and transformation
    - Connection pooling and session management
    - Performance monitoring and logging
    - Integration with existing PromptChain infrastructure
    """
    
    def __init__(self, 
                 mcp_servers_config: List[Dict[str, Any]], 
                 verbose: bool = False,
                 connection_timeout: float = 30.0,
                 max_retries: int = 3,
                 parameter_validation: bool = True):
        """
        Initialize MCP Tool Hijacker.
        
        Args:
            mcp_servers_config: List of MCP server configurations
            verbose: Enable debug output
            connection_timeout: Timeout for server connections
            max_retries: Maximum connection retry attempts
            parameter_validation: Enable parameter validation
        """
        self.mcp_servers_config = mcp_servers_config or []
        self.verbose = verbose
        self.parameter_validation = parameter_validation
        
        # Initialize components
        self.connection_manager = MCPConnectionManager(
            mcp_servers_config=mcp_servers_config,
            verbose=verbose,
            connection_timeout=connection_timeout,
            max_retries=max_retries
        )
        
        self.param_manager = ToolParameterManager(verbose=verbose)
        
        # Step chaining support for dynamic parameter passing
        self.step_chaining_manager = StepChainingManager(verbose=verbose)
        self.json_parser = JSONOutputParser(verbose=verbose)
        
        # State management
        self._connected = False
        self._performance_stats: Dict[str, Dict[str, Any]] = {}
        self._stats_lock = asyncio.Lock()  # BUG-005 fix: Thread-safe stats updates

        # Logging
        self.logger = logging.getLogger(__name__)
        if verbose:
            self.logger.setLevel(logging.DEBUG)
        
        # Tool execution hooks (BUG-006 fix: support blocking hooks)
        # Format: List[Tuple[Callable, bool]] where bool is blocking flag
        self.pre_execution_hooks: List[Tuple[Callable, bool]] = []
        self.post_execution_hooks: List[Tuple[Callable, bool]] = []
        
        self.logger.info(f"MCPToolHijacker initialized with {len(self.mcp_servers_config)} servers")
    
    @property
    def is_connected(self) -> bool:
        """Check if hijacker is connected to MCP servers."""
        return self._connected and self.connection_manager.is_connected
    
    @property
    def available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return self.connection_manager.available_tools
    
    @property
    def connected_servers(self) -> List[str]:
        """Get list of connected server IDs."""
        return self.connection_manager.connected_servers
    
    async def connect(self) -> None:
        """
        Establish connections to MCP servers and initialize tool discovery.
        
        Raises:
            MCPConnectionError: If connection fails
            MCPToolDiscoveryError: If tool discovery fails
        """
        if self._connected:
            self.logger.debug("Already connected to MCP servers")
            return
        
        try:
            self.logger.info("Connecting MCP Tool Hijacker...")
            
            # Connect via connection manager
            await self.connection_manager.connect()
            
            # Initialize performance tracking for available tools
            self._init_performance_tracking()
            
            self._connected = True
            
            self.logger.info(
                f"MCP Tool Hijacker connected successfully. "
                f"Servers: {len(self.connected_servers)}, "
                f"Tools: {len(self.available_tools)}"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to connect MCP Tool Hijacker: {e}")
            self._connected = False
            raise
    
    def _init_performance_tracking(self) -> None:
        """Initialize performance tracking for discovered tools."""
        for tool_name in self.available_tools:
            self._performance_stats[tool_name] = {
                "call_count": 0,
                "total_time": 0.0,
                "avg_time": 0.0,
                "last_call_time": 0.0,
                "error_count": 0,
                "success_rate": 1.0
            }
    
    async def disconnect(self) -> None:
        """Disconnect from MCP servers and clean up resources."""
        if not self._connected:
            return
        
        self.logger.info("Disconnecting MCP Tool Hijacker...")
        
        try:
            await self.connection_manager.disconnect()
        except Exception as e:
            self.logger.error(f"Error during disconnection: {e}")
        finally:
            self._connected = False
            self.logger.info("MCP Tool Hijacker disconnected")
    
    def set_static_params(self, tool_name: str, **params) -> None:
        """
        Set static parameters for a tool.
        
        Args:
            tool_name: Name of the tool
            **params: Parameter key-value pairs
        """
        self.param_manager.set_static_params(tool_name, **params)
        
        if self.verbose:
            self.logger.debug(f"Set static params for {tool_name}: {list(params.keys())}")
    
    def add_param_transformer(self, tool_name: str, param_name: str, transformer: Callable) -> None:
        """
        Add parameter transformation function.
        
        Args:
            tool_name: Name of the tool
            param_name: Name of the parameter
            transformer: Function that transforms the parameter value
        """
        self.param_manager.add_transformer(tool_name, param_name, transformer)
    
    def add_global_transformer(self, param_name: str, transformer: Callable) -> None:
        """
        Add global parameter transformation function.
        
        Args:
            param_name: Name of the parameter
            transformer: Function that transforms the parameter value
        """
        self.param_manager.add_global_transformer(param_name, transformer)
    
    def add_param_validator(self, tool_name: str, param_name: str, validator: Callable) -> None:
        """
        Add parameter validation function.
        
        Args:
            tool_name: Name of the tool
            param_name: Name of the parameter
            validator: Function that validates the parameter value
        """
        self.param_manager.add_validator(tool_name, param_name, validator)
    
    def add_global_validator(self, param_name: str, validator: Callable) -> None:
        """
        Add global parameter validation function.
        
        Args:
            param_name: Name of the parameter
            validator: Function that validates the parameter value
        """
        self.param_manager.add_global_validator(param_name, validator)
    
    def set_required_params(self, tool_name: str, param_names: List[str]) -> None:
        """
        Set required parameters for a tool.
        
        Args:
            tool_name: Name of the tool
            param_names: List of required parameter names
        """
        self.param_manager.set_required_params(tool_name, param_names)
    
    def add_execution_hook(self, hook: Callable, stage: str = "pre", blocking: bool = False) -> None:
        """
        Add execution hook.

        BUG-006 fix: Added blocking parameter to control hook failure behavior.

        Args:
            hook: Callable to execute (receives tool_name, params)
            stage: "pre" for before execution, "post" for after execution
            blocking: If True, tool execution aborts if hook fails.
                     If False, hook failures only log warnings (default).
        """
        if stage == "pre":
            self.pre_execution_hooks.append((hook, blocking))
        elif stage == "post":
            self.post_execution_hooks.append((hook, blocking))
        else:
            raise ValueError("Stage must be 'pre' or 'post'")
    
    def get_available_tools(self) -> List[str]:
        """
        Get list of available tools.
        
        Returns:
            List of tool names
        """
        return self.available_tools.copy()
    
    def get_tool_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get schema for a specific tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool schema dictionary or None if not found
        """
        return self.connection_manager.get_tool_schema(tool_name)
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive information about a tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool information including schema, config, and stats
        """
        schema = self.connection_manager.get_tool_schema(tool_name)
        config = self.param_manager.get_tool_config(tool_name)
        stats = self._performance_stats.get(tool_name, {})
        
        if not schema:
            return None
        
        return {
            "schema": schema,
            "parameter_config": config,
            "performance_stats": stats,
            "available": tool_name in self.available_tools
        }
    
    async def call_tool(self, tool_name: str, template_vars: Optional[Dict[str, Any]] = None,
                       **kwargs) -> Any:
        """
        Execute MCP tool directly with parameter processing.
        
        Args:
            tool_name: Name of the tool to execute
            template_vars: Variables for template substitution
            **kwargs: Tool parameters
            
        Returns:
            Tool execution result
            
        Raises:
            ToolNotFoundError: If tool doesn't exist
            ParameterValidationError: If parameters are invalid
            ToolExecutionError: If execution fails
        """
        if not self.is_connected:
            raise ToolExecutionError("MCP Tool Hijacker not connected. Call connect() first.")
        
        if tool_name not in self.available_tools:
            available = ", ".join(self.available_tools) if self.available_tools else "none"
            raise ToolNotFoundError(
                f"Tool '{tool_name}' not found. Available tools: {available}"
            )
        
        start_time = time.time()
        
        try:
            # Execute pre-execution hooks (BUG-006 fix: blocking hooks abort execution)
            for hook, blocking in self.pre_execution_hooks:
                try:
                    hook(tool_name, kwargs)
                except Exception as e:
                    if blocking:
                        # Blocking hook failure - abort tool execution
                        error_msg = f"Blocking pre-execution hook failed for {tool_name}: {e}"
                        self.logger.error(error_msg)
                        raise ToolExecutionError(error_msg)
                    else:
                        # Non-blocking hook - just log warning
                        self.logger.warning(f"Pre-execution hook failed: {e}")
            
            # Process parameters
            if self.parameter_validation:
                processed_params = self.param_manager.process_params(
                    tool_name, template_vars, **kwargs
                )
            else:
                # Simple merge without validation/transformation
                processed_params = self.param_manager.merge_params(tool_name, **kwargs)
            
            if self.verbose:
                self.logger.debug(
                    f"Executing {tool_name} with {len(processed_params)} parameters"
                )
            
            # Execute tool via connection manager
            result = await self.connection_manager.execute_tool(tool_name, processed_params)

            # Update performance stats
            execution_time = time.time() - start_time
            await self._update_performance_stats(tool_name, execution_time, success=True)
            
            # Execute post-execution hooks (BUG-006 fix: blocking hooks abort execution)
            for hook, blocking in self.post_execution_hooks:
                try:
                    hook(tool_name, processed_params, result, execution_time)
                except Exception as e:
                    if blocking:
                        # Blocking hook failure - raise error (result already obtained)
                        error_msg = f"Blocking post-execution hook failed for {tool_name}: {e}"
                        self.logger.error(error_msg)
                        raise ToolExecutionError(error_msg)
                    else:
                        # Non-blocking hook - just log warning
                        self.logger.warning(f"Post-execution hook failed: {e}")
            
            if self.verbose:
                self.logger.debug(f"Tool {tool_name} executed successfully in {execution_time:.3f}s")
            
            return result
            
        except (ParameterValidationError, ParameterTransformationError) as e:
            # Parameter processing errors
            execution_time = time.time() - start_time
            await self._update_performance_stats(tool_name, execution_time, success=False)
            self.logger.error(f"Parameter error for {tool_name}: {e}")
            raise
            
        except Exception as e:
            # Tool execution errors
            execution_time = time.time() - start_time
            await self._update_performance_stats(tool_name, execution_time, success=False)
            error_msg = f"Tool execution failed for {tool_name}: {e}"
            self.logger.error(error_msg)
            raise ToolExecutionError(error_msg)
    
    async def _update_performance_stats(self, tool_name: str, execution_time: float, success: bool) -> None:
        """Update performance statistics for a tool (thread-safe).

        BUG-005 fix: Uses asyncio.Lock to prevent race conditions during
        concurrent batch_call_tools execution.
        """
        async with self._stats_lock:
            if tool_name not in self._performance_stats:
                self._performance_stats[tool_name] = {
                    "call_count": 0,
                    "total_time": 0.0,
                    "avg_time": 0.0,
                    "last_call_time": 0.0,
                    "error_count": 0,
                    "success_rate": 1.0
                }

            stats = self._performance_stats[tool_name]
            stats["call_count"] += 1
            stats["total_time"] += execution_time
            stats["last_call_time"] = execution_time
            stats["avg_time"] = stats["total_time"] / stats["call_count"]

            if not success:
                stats["error_count"] += 1

            stats["success_rate"] = (stats["call_count"] - stats["error_count"]) / stats["call_count"]
    
    async def call_tool_batch(self, batch_calls: List[Dict[str, Any]], 
                             max_concurrent: int = 5) -> List[Dict[str, Any]]:
        """
        Execute multiple tool calls concurrently.
        
        Args:
            batch_calls: List of tool call dictionaries with 'tool_name' and 'params'
            max_concurrent: Maximum concurrent executions
            
        Returns:
            List of results with success/error status
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def execute_single_call(call_info: Dict[str, Any]) -> Dict[str, Any]:
            async with semaphore:
                tool_name = call_info["tool_name"]
                params = call_info.get("params", {})
                template_vars = call_info.get("template_vars")
                
                try:
                    result = await self.call_tool(tool_name, template_vars, **params)
                    return {
                        "tool_name": tool_name,
                        "success": True,
                        "result": result,
                        "error": None
                    }
                except Exception as e:
                    return {
                        "tool_name": tool_name,
                        "success": False,
                        "result": None,
                        "error": str(e)
                    }
        
        # Execute all calls concurrently
        tasks = [execute_single_call(call) for call in batch_calls]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        
        return results
    
    def get_performance_stats(self, tool_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get performance statistics.
        
        Args:
            tool_name: Specific tool name, or None for all tools
            
        Returns:
            Performance statistics
        """
        if tool_name:
            return self._performance_stats.get(tool_name, {}).copy()
        
        return {
            "overall": {
                "total_tools": len(self._performance_stats),
                "total_calls": sum(stats["call_count"] for stats in self._performance_stats.values()),
                "total_errors": sum(stats["error_count"] for stats in self._performance_stats.values()),
                "average_success_rate": sum(stats["success_rate"] for stats in self._performance_stats.values()) / len(self._performance_stats) if self._performance_stats else 0
            },
            "by_tool": self._performance_stats.copy()
        }
    
    def clear_performance_stats(self) -> None:
        """Clear all performance statistics."""
        self._performance_stats.clear()
        self._init_performance_tracking()
        
        if self.verbose:
            self.logger.debug("Performance statistics cleared")
    
    # === Step Chaining Methods ===
    
    def store_step_output(self, step_index: int, output: Any, step_name: Optional[str] = None):
        """
        Store output from a step for later use in parameter templates.
        
        Args:
            step_index: Index of the step (1-based)
            output: Step output data
            step_name: Optional name for the step
        """
        self.step_chaining_manager.store_step_output(step_index, output, step_name)
    
    def create_template_vars_for_current_step(self, current_step: int) -> Dict[str, Any]:
        """
        Create template variables for the current step from previous step outputs.
        
        Args:
            current_step: Current step index
            
        Returns:
            Dictionary of template variables
        """
        return self.step_chaining_manager.create_template_vars_for_step(current_step)
    
    async def call_tool_with_chaining(self, tool_name: str, current_step: int, **params) -> Any:
        """
        Execute tool with automatic step chaining support.
        
        Args:
            tool_name: Name of the tool to execute
            current_step: Current step index for template variable creation
            **params: Tool parameters (may include templates like {previous.results[0].id})
            
        Returns:
            Tool execution result
        """
        # Create template variables from previous steps
        template_vars = self.create_template_vars_for_current_step(current_step)
        
        # Execute tool with template variables
        return await self.call_tool(tool_name, template_vars=template_vars, **params)
    
    def parse_output_for_chaining(self, output: Any, parse_config: Optional[Dict[str, Any]] = None) -> Any:
        """
        Parse tool output for easier step chaining.
        
        Args:
            output: Raw tool output  
            parse_config: Optional parsing configuration
            
        Returns:
            Processed output with extracted values
        """
        return self.step_chaining_manager.parse_step_output_for_chaining(output, parse_config)
    
    def get_step_reference_info(self) -> Dict[str, Any]:
        """Get available step references for debugging."""
        return {
            "available_references": self.step_chaining_manager.get_available_references(),
            "current_step": self.step_chaining_manager.current_step
        }
    
    def clear_step_outputs(self):
        """Clear all stored step outputs."""
        self.step_chaining_manager.clear_outputs()
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status information.
        
        Returns:
            Status dictionary with connection, tool, and performance information
        """
        return {
            "connected": self.is_connected,
            "connection_manager": self.connection_manager.get_connection_status(),
            "available_tools": len(self.available_tools),
            "tool_list": self.available_tools,
            "performance_summary": self.get_performance_stats(),
            "parameter_manager": {
                "tools_with_static_params": len(self.param_manager.static_params),
                "tools_with_transformers": len(self.param_manager.transformers),
                "tools_with_validators": len(self.param_manager.validators)
            }
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()


# Convenience functions for common configurations
def create_temperature_clamped_hijacker(mcp_servers_config: List[Dict[str, Any]], 
                                       verbose: bool = False) -> MCPToolHijacker:
    """
    Create hijacker with temperature parameter clamping for LLM tools.
    
    Args:
        mcp_servers_config: MCP server configurations
        verbose: Enable debug output
        
    Returns:
        Configured MCPToolHijacker instance
    """
    hijacker = MCPToolHijacker(mcp_servers_config, verbose=verbose)
    
    # Add global temperature clamping
    hijacker.add_global_transformer("temperature", CommonTransformers.clamp_float(0.0, 2.0))
    hijacker.add_global_validator("temperature", CommonValidators.is_float_in_range(0.0, 2.0))
    
    return hijacker


def create_production_hijacker(mcp_servers_config: List[Dict[str, Any]], 
                              verbose: bool = False) -> MCPToolHijacker:
    """
    Create production-ready hijacker with common configurations.
    
    Args:
        mcp_servers_config: MCP server configurations
        verbose: Enable debug output
        
    Returns:
        Production-configured MCPToolHijacker instance
    """
    hijacker = MCPToolHijacker(
        mcp_servers_config=mcp_servers_config,
        verbose=verbose,
        connection_timeout=60.0,  # Longer timeout for production
        max_retries=5,            # More retries for reliability
        parameter_validation=True
    )
    
    # Add common transformers and validators
    hijacker.add_global_transformer("temperature", CommonTransformers.clamp_float(0.0, 2.0))
    hijacker.add_global_transformer("prompt", CommonTransformers.truncate_string(10000))
    
    hijacker.add_global_validator("temperature", CommonValidators.is_float_in_range(0.0, 2.0))
    hijacker.add_global_validator("prompt", CommonValidators.is_non_empty_string())
    
    return hijacker