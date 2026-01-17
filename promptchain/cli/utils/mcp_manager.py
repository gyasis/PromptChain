"""MCP server connection manager for CLI (T061).

This module provides centralized management of MCP server connections,
wrapping MCPHelper functionality with session-aware state tracking.
"""

from typing import Optional, List
from datetime import datetime
import logging

from promptchain.cli.models.session import Session
from promptchain.cli.models.mcp_config import MCPServerConfig
from promptchain.utils.mcp_helpers import MCPHelper


class MCPManager:
    """Manages MCP server connections for a CLI session.

    Wraps MCPHelper functionality with session-aware state tracking,
    ensuring MCP server configurations are properly synchronized with
    the session database.

    Attributes:
        session: The active session managing MCP server states
    """

    def __init__(self, session: Session):
        """Initialize MCP manager for a session.

        Args:
            session: Active session with MCP server configurations
        """
        self.session = session

    async def connect_server(
        self, server_id: str, auto_register_with_agents: Optional[List[str]] = None
    ) -> bool:
        """Connect to an MCP server by ID using real MCPHelper (T061).

        Args:
            server_id: Server identifier from session.mcp_servers list
            auto_register_with_agents: Optional list of agent names to auto-register
                                       tools with after connection (T063)

        Returns:
            bool: True if connection successful, False if failed

        Raises:
            ValueError: If server_id not found in session
        """
        # Find server config by ID
        server_config = self._find_server_by_id(server_id)
        if not server_config:
            raise ValueError(f"MCP server '{server_id}' not found in session")

        try:
            # T061: Use real MCPHelper for connection and tool discovery
            # Convert MCPServerConfig to MCPHelper format
            mcp_server_configs = [{
                "id": server_config.id,
                "type": server_config.type,
                "command": server_config.command,
                "args": server_config.args or [],
                "url": server_config.url
            }]

            # Create MCPHelper instance
            logger = logging.getLogger(__name__)
            mcp_helper = MCPHelper(
                mcp_servers=mcp_server_configs,
                verbose=False,
                logger_instance=logger,
                local_tool_schemas_ref=[],  # No local tools in CLI context
                local_tool_functions_ref={}  # No local functions in CLI context
            )

            try:
                # Connect to MCP server and discover tools
                await mcp_helper.connect_mcp_async()

                # Extract discovered tools from MCPHelper
                # Tools are in mcp_tool_schemas with names like "mcp_{server_id}_{tool_name}"
                discovered_tool_names = [
                    tool_schema.get("function", {}).get("name")
                    for tool_schema in mcp_helper.mcp_tool_schemas
                    if tool_schema.get("function", {}).get("name")
                ]

                # Mark server as connected with real discovered tools
                server_config.mark_connected(tools=discovered_tool_names)
            finally:
                # CRITICAL: Properly close MCP connection to avoid async cleanup errors
                await mcp_helper.close_mcp_async()

            # T063: Auto-register tools with agents if requested
            if auto_register_with_agents:
                for agent_name in auto_register_with_agents:
                    try:
                        await self.register_tools_with_agent(server_id, agent_name)
                    except ValueError as e:
                        logger.warning(f"Auto-register failed for {agent_name}: {e}")

            return True

        except Exception as e:
            # Mark server as error state
            server_config.mark_error(str(e))

            # T068: Display warning message with tool count
            tool_count = len(server_config.discovered_tools)  # Will be 0 for failed connections
            logger.warning(
                f"MCP server '{server_id}' unavailable - {tool_count} tools disabled"
            )

            return False

    async def disconnect_server(self, server_id: str) -> bool:
        """Disconnect from an MCP server.

        Args:
            server_id: Server identifier to disconnect

        Returns:
            bool: True if disconnection successful, False if failed

        Raises:
            ValueError: If server_id not found in session
        """
        # Find server config by ID
        server_config = self._find_server_by_id(server_id)
        if not server_config:
            raise ValueError(f"MCP server '{server_id}' not found in session")

        try:
            # Unregister tools from all agents before disconnecting
            await self._unregister_server_tools_from_all_agents(server_id)

            # Clean disconnection
            server_config.mark_disconnected()
            return True

        except Exception as e:
            # Log error but still mark as disconnected
            server_config.mark_error(f"Disconnect error: {e}")
            server_config.mark_disconnected()
            return False

    async def register_tools_with_agent(
        self, server_id: str, agent_name: str
    ) -> bool:
        """Register MCP server tools with a specific agent (T063).

        Args:
            server_id: Server identifier whose tools to register
            agent_name: Agent name to register tools with

        Returns:
            bool: True if registration successful, False if failed

        Raises:
            ValueError: If server_id not found or not connected
            ValueError: If agent_name not found in session
        """
        # Find server config by ID
        server_config = self._find_server_by_id(server_id)
        if not server_config:
            raise ValueError(f"MCP server '{server_id}' not found in session")

        # Verify server is connected
        if server_config.state != "connected":
            raise ValueError(f"Server '{server_id}' is not connected")

        # Find agent in session
        if agent_name not in self.session.agents:
            raise ValueError(f"Agent '{agent_name}' not found in session")

        agent = self.session.agents[agent_name]

        try:
            # Add discovered tools to agent's tools list
            for tool_name in server_config.discovered_tools:
                if tool_name not in agent.tools:
                    agent.tools.append(tool_name)

            return True

        except Exception as e:
            # Log error but don't fail session
            logging.getLogger(__name__).error(
                f"Failed to register tools from {server_id} with {agent_name}: {e}"
            )
            return False

    async def unregister_tools_from_agent(
        self, server_id: str, agent_name: str
    ) -> bool:
        """Unregister MCP server tools from a specific agent.

        Args:
            server_id: Server identifier whose tools to unregister
            agent_name: Agent name to unregister tools from

        Returns:
            bool: True if unregistration successful, False if failed

        Raises:
            ValueError: If server_id not found
            ValueError: If agent_name not found in session
        """
        # Find server config by ID
        server_config = self._find_server_by_id(server_id)
        if not server_config:
            raise ValueError(f"MCP server '{server_id}' not found in session")

        # Find agent in session
        if agent_name not in self.session.agents:
            raise ValueError(f"Agent '{agent_name}' not found in session")

        agent = self.session.agents[agent_name]

        try:
            # Remove server's tools from agent's tools list
            agent.tools = [
                tool for tool in agent.tools
                if tool not in server_config.discovered_tools
            ]

            return True

        except Exception as e:
            logging.getLogger(__name__).error(
                f"Failed to unregister tools from {server_id} with {agent_name}: {e}"
            )
            return False

    async def _unregister_server_tools_from_all_agents(self, server_id: str) -> None:
        """Unregister tools from a server from all agents.

        Args:
            server_id: Server identifier whose tools to unregister
        """
        # Find server config
        server_config = self._find_server_by_id(server_id)
        if not server_config:
            return

        # Unregister from all agents
        for agent_name in self.session.agents.keys():
            try:
                await self.unregister_tools_from_agent(server_id, agent_name)
            except Exception as e:
                logging.getLogger(__name__).warning(
                    f"Failed to unregister tools from {agent_name}: {e}"
                )

    def _find_server_by_id(self, server_id: str) -> Optional[MCPServerConfig]:
        """Find MCP server config by ID.

        Args:
            server_id: Server identifier to find

        Returns:
            MCPServerConfig: Server configuration if found, None otherwise
        """
        for server in self.session.mcp_servers:
            if server.id == server_id:
                return server
        return None

    async def connect_all_auto_servers(self) -> List[str]:
        """Connect to all servers with auto_connect=True.

        Returns:
            List[str]: List of successfully connected server IDs
        """
        connected_ids = []

        for server in self.session.mcp_servers:
            if server.auto_connect:
                success = await self.connect_server(server.id)
                if success:
                    connected_ids.append(server.id)

        return connected_ids

    async def disconnect_all_servers(self) -> None:
        """Disconnect from all connected servers.

        Ensures clean shutdown of all MCP connections.
        """
        for server in self.session.mcp_servers:
            if server.state == "connected":
                await self.disconnect_server(server.id)

    def get_connected_servers(self) -> List[MCPServerConfig]:
        """Get all currently connected MCP servers.

        Returns:
            List[MCPServerConfig]: List of connected server configurations
        """
        return [s for s in self.session.mcp_servers if s.state == "connected"]

    def get_all_discovered_tools(self) -> List[str]:
        """Get all tools discovered from connected servers.

        Returns:
            List[str]: List of all discovered tool names across all servers
        """
        all_tools = []
        for server in self.session.mcp_servers:
            if server.state == "connected":
                all_tools.extend(server.discovered_tools)
        return all_tools
