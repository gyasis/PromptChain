#!/usr/bin/env python3
"""
MCP Development Test Script
==========================
Quick tool testing for Athena LightRAG MCP Server - FastMCP 2.0 Compatible
"""

import asyncio
from fastmcp import Client

# Import the module-level mcp instance
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from athena_mcp_server import mcp

async def test_mcp_tools():
    """Test FastMCP 2.0 tools directly."""
    print("🧪 Testing Athena LightRAG MCP Tools (FastMCP 2.0)")
    print("=" * 55)
    
    try:
        print("✅ MCP Server module loaded successfully")
        
        # Use FastMCP Client for in-memory testing
        async with Client(mcp) as client:
            print("✅ FastMCP Client connected")
            
            # Test basic tool
            print("\n🔍 Testing lightrag_local_query...")
            result = await client.call_tool("lightrag_local_query", {
                "query": "athena.athenaone.PATIENT table relationships",
                "top_k": 5
            })
            
            if result and hasattr(result, 'content'):
                print("✅ Tool executed successfully")
                content = result.content[0].text if result.content else str(result)
                print(f"Preview: {content[:200]}...")
            else:
                print(f"✅ Result received: {str(result)[:200]}...")
            
            # List all available tools
            tools = await client.list_tools()
            print(f"\n📋 Available tools ({len(tools)}):")
            for tool in tools:
                print(f"  • {tool.name}")
                
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_mcp_tools())