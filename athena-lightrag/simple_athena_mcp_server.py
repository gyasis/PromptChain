#!/usr/bin/env python3
"""
Simplified Athena MCP Server for Testing
======================================
Basic MCP server that works without complex dependencies.
"""

import asyncio
import logging
import json
import sys
from pathlib import Path

# Configure logging to file to avoid stdio conflicts
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='/tmp/simple_athena_mcp.log',
    filemode='w'
)
logger = logging.getLogger(__name__)

try:
    # Try FastMCP first
    from mcp.server.fastmcp.server import FastMCP
    
    mcp = FastMCP("Simple Athena Test Server")
    
    @mcp.tool()
    async def test_query(query: str) -> dict:
        """Simple test query function."""
        logger.info(f"Test query received: {query}")
        return {
            "success": True,
            "result": f"Test response for: {query}",
            "server": "simplified-athena"
        }
    
    @mcp.tool()
    async def server_status() -> dict:
        """Get server status."""
        return {
            "status": "running",
            "server": "simplified-athena",
            "tools": ["test_query", "server_status"]
        }
    
    if __name__ == "__main__":
        logger.info("Starting simplified Athena MCP server...")
        print("Simple Athena MCP Server starting...", file=sys.stderr)
        asyncio.run(mcp.run())

except ImportError as e:
    logger.error(f"FastMCP not available: {e}")
    
    # Fallback to basic stdio MCP
    class SimpleMCPServer:
        def __init__(self):
            self.tools = {
                "test_query": self.test_query,
                "server_status": self.server_status
            }
        
        async def test_query(self, query: str):
            return {
                "success": True,
                "result": f"Fallback response for: {query}",
                "server": "simplified-athena-fallback"
            }
        
        async def server_status(self):
            return {
                "status": "running",
                "server": "simplified-athena-fallback",
                "tools": list(self.tools.keys())
            }
        
        async def run(self):
            logger.info("Starting fallback MCP server...")
            print("Fallback Athena MCP Server starting...", file=sys.stderr)
            
            # Basic MCP protocol handler
            while True:
                try:
                    line = input()
                    if not line:
                        break
                        
                    request = json.loads(line)
                    
                    if request.get("method") == "initialize":
                        response = {
                            "jsonrpc": "2.0",
                            "id": request.get("id"),
                            "result": {
                                "serverInfo": {
                                    "name": "simplified-athena",
                                    "version": "1.0.0"
                                },
                                "capabilities": {
                                    "tools": {}
                                }
                            }
                        }
                        print(json.dumps(response))
                        
                except EOFError:
                    break
                except Exception as e:
                    logger.error(f"Protocol error: {e}")
                    break
    
    if __name__ == "__main__":
        server = SimpleMCPServer()
        asyncio.run(server.run())
