#!/usr/bin/env python3
"""
Test Real MCP Client with Sci-Hub Integration
"""

import asyncio
import json
import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from research_agent.integrations.mcp_client import MCPClient

async def test_real_mcp_client():
    """Test the real MCP client with Sci-Hub integration"""
    print("=== Testing Real MCP Client with Sci-Hub ===\n")
    
    # Initialize MCP client
    mcp_client = MCPClient()
    
    try:
        # Connect to servers
        print("1. Connecting to MCP servers...")
        connected = await mcp_client.connect()
        print(f"Connection result: {connected}")
        
        if not connected:
            print("Failed to connect to any servers")
            return
        
        print(f"Connected servers: {list(mcp_client.servers.keys())}")
        
        # Test keyword search
        print("\n2. Testing keyword search...")
        keyword_result = await mcp_client.call_tool(
            "search_scihub_by_keyword", 
            {"keywords": "machine learning", "limit": 3}
        )
        
        if keyword_result:
            print("Keyword search result:")
            print(json.dumps(keyword_result, indent=2))
            
            # Check if we got real data vs mock data
            papers = keyword_result.get('papers', [])
            if papers:
                first_paper = papers[0]
                if 'mock' in first_paper.get('id', '').lower():
                    print("⚠️  Still getting mock data!")
                else:
                    print("✅ Getting real data from CrossRef!")
        
        # Test DOI search with a real DOI
        print("\n3. Testing DOI search...")
        doi_result = await mcp_client.call_tool(
            "search_scihub_by_doi",
            {"doi": "10.1038/nature12373"}
        )
        
        if doi_result:
            print("DOI search result:")
            print(json.dumps(doi_result, indent=2))
        
        # Test title search
        print("\n4. Testing title search...")
        title_result = await mcp_client.call_tool(
            "search_scihub_by_title",
            {"title": "Machine learning applications in healthcare"}
        )
        
        if title_result:
            print("Title search result:")
            print(json.dumps(title_result, indent=2))
        
        # Get available tools
        print("\n5. Available tools:")
        tools = mcp_client.get_available_tools()
        for tool in tools:
            print(f"  - {tool}")
        
        print("\n=== Test Results Summary ===")
        if keyword_result and keyword_result.get('source') == 'sci_hub_real_crossref':
            print("✅ Real CrossRef keyword search working!")
        else:
            print("❌ Still using mock keyword search")
            
        if doi_result and doi_result.get('source') == 'sci_hub_real_doi':
            print("✅ Real Sci-Hub DOI search working!")
        else:
            print("❌ Still using mock DOI search")
            
        if title_result and title_result.get('source') == 'sci_hub_real_title':
            print("✅ Real Sci-Hub title search working!")
        else:
            print("❌ Still using mock title search")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Disconnect
        await mcp_client.disconnect()
        print("\nMCP client disconnected")

if __name__ == "__main__":
    asyncio.run(test_real_mcp_client())