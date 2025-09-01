"""
MCP Connection Manager for PromptChain Tool Hijacker

This module provides connection management functionality for MCP servers,
handling connection pooling, discovery, and session management for direct tool execution.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from contextlib import AsyncExitStack
import json

# Attempt to import MCP components, handle gracefully if not installed
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    ClientSession = None
    StdioServerParameters = None
    stdio_client = None


class MCPConnectionError(Exception):
    """Raised when MCP server connection fails."""
    pass


class MCPToolDiscoveryError(Exception):
    """Raised when tool discovery fails."""
    pass


class AsyncProcessContextManager:
    """Manages the lifecycle of an asyncio subprocess, ensuring termination."""
    
    def __init__(self, process: asyncio.subprocess.Process):
        self.process = process

    async def __aenter__(self):
        return self.process

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.process.returncode is None:
            try:
                # Try graceful termination first
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                # Force kill if graceful termination fails
                self.process.kill()
                await self.process.wait()
            except (ProcessLookupError, OSError):
                # Process already terminated
                pass


class MCPConnectionManager:
    """
    Manages MCP server connections, tool discovery, and session lifecycle.
    
    This class provides:
    - Connection pooling and management
    - Automatic tool discovery
    - Session lifecycle management
    - Error handling and retry logic
    - Resource cleanup
    """
    
    def __init__(self, 
                 mcp_servers_config: List[Dict[str, Any]], 
                 verbose: bool = False,
                 connection_timeout: float = 30.0,
                 max_retries: int = 3):
        """
        Initialize MCP Connection Manager.
        
        Args:
            mcp_servers_config: List of MCP server configurations
            verbose: Enable debug output
            connection_timeout: Timeout for server connections
            max_retries: Maximum connection retry attempts
        """
        self.mcp_servers_config = mcp_servers_config or []
        self.verbose = verbose
        self.connection_timeout = connection_timeout
        self.max_retries = max_retries
        
        # Connection state
        self.sessions: Dict[str, ClientSession] = {}
        self.tools_map: Dict[str, Dict[str, Any]] = {}  # tool_name -> {schema, server_id}
        self.server_tools: Dict[str, List[str]] = {}  # server_id -> [tool_names]
        
        # Resource management
        self.exit_stack = AsyncExitStack() if MCP_AVAILABLE and self.mcp_servers_config else None
        self._connected = False
        
        # Logging
        self.logger = logging.getLogger(__name__)
        if verbose:
            self.logger.setLevel(logging.DEBUG)
        
        # Validate MCP availability
        if self.mcp_servers_config and not MCP_AVAILABLE:
            self.logger.warning(
                "MCPConnectionManager initialized with servers, but 'mcp' library not installed. "
                "MCP features will be disabled."
            )
    
    @property
    def is_connected(self) -> bool:
        """Check if manager has active connections."""
        return self._connected and bool(self.sessions)
    
    @property
    def connected_servers(self) -> List[str]:
        """Get list of successfully connected server IDs."""
        return list(self.sessions.keys())
    
    @property  
    def available_tools(self) -> List[str]:
        """Get list of all available tool names."""
        return list(self.tools_map.keys())
    
    async def connect(self) -> None:
        """
        Establish connections to all configured MCP servers and discover tools.
        
        Raises:
            MCPConnectionError: If unable to connect to any server
            MCPToolDiscoveryError: If tool discovery fails
        """
        if not MCP_AVAILABLE:
            self.logger.warning("MCP library not available, skipping connection")
            return
        
        if not self.mcp_servers_config:
            self.logger.info("No MCP servers configured")
            return
        
        if self._connected:
            self.logger.debug("Already connected to MCP servers")
            return
        
        self.logger.info(f"Connecting to {len(self.mcp_servers_config)} MCP servers...")
        
        # Track connection results
        connection_results = []
        
        # Connect to each server
        for server_config in self.mcp_servers_config:
            server_id = server_config.get("id", "unknown")
            
            for attempt in range(self.max_retries):
                try:
                    session = await self._connect_single_server(server_config)
                    self.sessions[server_id] = session
                    
                    self.logger.info(f"Successfully connected to MCP server: {server_id}")
                    connection_results.append((server_id, True, None))
                    break
                    
                except Exception as e:
                    error_msg = f"Connection attempt {attempt + 1}/{self.max_retries} failed for {server_id}: {e}"
                    self.logger.warning(error_msg)
                    
                    if attempt == self.max_retries - 1:
                        connection_results.append((server_id, False, str(e)))
        
        # Check if we have any successful connections
        successful_connections = [r for r in connection_results if r[1]]
        if not successful_connections:
            failed_servers = [f"{r[0]} ({r[2]})" for r in connection_results if not r[1]]
            raise MCPConnectionError(
                f"Failed to connect to any MCP servers. Failures: {failed_servers}"
            )
        
        self._connected = True
        
        # Discover tools from connected servers
        try:
            await self._discover_all_tools()
            self.logger.info(f"Discovered {len(self.tools_map)} tools from {len(self.sessions)} servers")
        except Exception as e:
            self.logger.error(f"Tool discovery failed: {e}")
            raise MCPToolDiscoveryError(f"Failed to discover tools: {e}")
    
    async def _connect_single_server(self, server_config: Dict[str, Any]) -> ClientSession:
        """
        Connect to a single MCP server.
        
        Args:
            server_config: Server configuration dictionary
            
        Returns:
            Connected ClientSession
            
        Raises:
            MCPConnectionError: If connection fails
        """
        server_id = server_config.get("id", "unknown")
        command = server_config.get("command")
        args = server_config.get("args", [])
        env = server_config.get("env", {})
        
        if not command:
            raise MCPConnectionError(f"No command specified for server {server_id}")
        
        try:
            # Create server parameters
            server_params = StdioServerParameters(
                command=command,
                args=args,
                env=env
            )
            
            # Connect with timeout
            session = await asyncio.wait_for(
                stdio_client(server_params),
                timeout=self.connection_timeout
            )
            
            # Register session for cleanup
            if self.exit_stack:
                await self.exit_stack.enter_async_context(session)
            
            self.logger.debug(f"Connected to {server_id} with command: {command} {args}")
            return session
            
        except asyncio.TimeoutError:
            raise MCPConnectionError(f"Connection to {server_id} timed out after {self.connection_timeout}s")
        except Exception as e:
            raise MCPConnectionError(f"Failed to connect to {server_id}: {e}")
    
    async def _discover_all_tools(self) -> None:
        """
        Discover tools from all connected servers.
        
        Raises:
            MCPToolDiscoveryError: If discovery fails
        """
        self.tools_map.clear()
        self.server_tools.clear()
        
        for server_id, session in self.sessions.items():
            try:
                tools = await self._discover_server_tools(server_id, session)
                self.server_tools[server_id] = tools
                self.logger.debug(f"Discovered {len(tools)} tools from {server_id}")
                
            except Exception as e:
                self.logger.error(f"Failed to discover tools from {server_id}: {e}")
                # Continue with other servers rather than failing completely
                self.server_tools[server_id] = []
    
    async def _discover_server_tools(self, server_id: str, session: ClientSession) -> List[str]:
        """
        Discover tools from a specific server.
        
        Args:
            server_id: Server identifier
            session: Connected session
            
        Returns:
            List of discovered tool names
        """
        try:
            # List available tools
            tools_response = await session.list_tools()
            discovered_tools = []
            
            for tool in tools_response.tools:
                # Create prefixed tool name to avoid conflicts
                prefixed_name = f"mcp_{server_id}_{tool.name}"
                
                # Store tool information
                self.tools_map[prefixed_name] = {
                    "original_name": tool.name,
                    "server_id": server_id,
                    "schema": {
                        "type": "function",
                        "function": {
                            "name": prefixed_name,
                            "description": tool.description or f"MCP tool from {server_id}",
                            "parameters": tool.inputSchema or {"type": "object", "properties": {}}
                        }
                    }
                }
                
                discovered_tools.append(prefixed_name)
                
                if self.verbose:
                    self.logger.debug(f"Discovered tool: {prefixed_name} from {server_id}")
            
            return discovered_tools
            
        except Exception as e:
            self.logger.error(f"Tool discovery failed for {server_id}: {e}")
            raise MCPToolDiscoveryError(f"Failed to discover tools from {server_id}: {e}")
    
    def get_session(self, server_id: str) -> Optional[ClientSession]:
        """
        Get MCP session for a specific server.
        
        Args:
            server_id: Server identifier
            
        Returns:
            ClientSession if available, None otherwise
        """
        return self.sessions.get(server_id)
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific tool.
        
        Args:
            tool_name: Prefixed tool name
            
        Returns:
            Tool information dictionary or None if not found
        """
        return self.tools_map.get(tool_name)
    
    def get_tool_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get schema for a specific tool.
        
        Args:
            tool_name: Prefixed tool name
            
        Returns:
            Tool schema dictionary or None if not found
        """
        tool_info = self.tools_map.get(tool_name)
        return tool_info.get("schema") if tool_info else None
    
    def get_server_tools(self, server_id: str) -> List[str]:
        """
        Get list of tools from a specific server.
        
        Args:
            server_id: Server identifier
            
        Returns:
            List of tool names from the server
        """
        return self.server_tools.get(server_id, [])
    
    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """
        Execute a tool on the appropriate server.
        
        Args:
            tool_name: Prefixed tool name
            parameters: Tool parameters
            
        Returns:
            Tool execution result
            
        Raises:
            ValueError: If tool not found
            RuntimeError: If execution fails
        """
        tool_info = self.tools_map.get(tool_name)
        if not tool_info:
            available = ", ".join(self.tools_map.keys()) if self.tools_map else "none"
            raise ValueError(
                f"Tool '{tool_name}' not found. Available tools: {available}"
            )
        
        server_id = tool_info["server_id"]
        original_name = tool_info["original_name"]
        
        session = self.sessions.get(server_id)
        if not session:
            raise RuntimeError(f"No active session for server {server_id}")
        
        try:
            # Execute tool on the server
            if self.verbose:
                self.logger.debug(f"Executing {original_name} on {server_id} with params: {parameters}")
            
            result = await session.call_tool(original_name, parameters)
            
            if self.verbose:
                self.logger.debug(f"Tool {original_name} executed successfully")
            
            # Extract content from result
            if hasattr(result, 'content') and result.content:
                # Handle different content types
                content_parts = []
                for content_item in result.content:
                    if hasattr(content_item, 'text'):
                        content_parts.append(content_item.text)
                    else:
                        content_parts.append(str(content_item))
                return "\n".join(content_parts)
            
            # Fallback to string representation
            return str(result) if result else ""
            
        except Exception as e:
            error_msg = f"Tool execution failed for {tool_name}: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    async def disconnect(self) -> None:
        """
        Disconnect from all MCP servers and clean up resources.
        """
        if not self._connected:
            return
        
        self.logger.info("Disconnecting from MCP servers...")
        
        try:
            if self.exit_stack:
                await self.exit_stack.aclose()
        except Exception as e:
            self.logger.error(f"Error during MCP cleanup: {e}")
        finally:
            self.sessions.clear()
            self.tools_map.clear()
            self.server_tools.clear()
            self._connected = False
        
        self.logger.info("Disconnected from all MCP servers")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
    
    def get_connection_status(self) -> Dict[str, Any]:
        """
        Get detailed connection status information.
        
        Returns:
            Dictionary containing connection status details
        """
        return {
            "connected": self._connected,
            "mcp_available": MCP_AVAILABLE,
            "configured_servers": len(self.mcp_servers_config),
            "active_sessions": len(self.sessions),
            "connected_servers": list(self.sessions.keys()),
            "total_tools": len(self.tools_map),
            "tools_by_server": {
                server_id: len(tools) 
                for server_id, tools in self.server_tools.items()
            }
        }