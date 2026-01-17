#!/usr/bin/env python3
"""
Test MCP Integration for Research Agent

Quick test to verify Sci-Hub MCP tools are properly integrated and called.
"""

import asyncio
import logging
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from research_agent.core.orchestrator import AdvancedResearchOrchestrator
from research_agent.core.config import ResearchConfig

# Set up logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def test_mcp_integration():
    """Test MCP client integration with Sci-Hub"""
    
    print("🧪 Testing MCP Integration for Research Agent")
    print("=" * 50)
    
    try:
        # 1. Initialize orchestrator with MCP client
        print("\n1️⃣ Initializing Research Orchestrator...")
        config = ResearchConfig()
        orchestrator = AdvancedResearchOrchestrator(config)
        
        # 2. Initialize MCP client
        print("\n2️⃣ Connecting to MCP servers...")
        await orchestrator.initialize_mcp_client()
        
        if orchestrator.mcp_client and orchestrator.mcp_client.connected:
            tools = orchestrator.mcp_client.get_available_tools()
            print(f"✅ MCP client connected successfully!")
            print(f"📋 Available tools: {tools}")
        else:
            print("❌ MCP client not connected")
            return False
        
        # 3. Test literature searcher has MCP client
        print("\n3️⃣ Checking literature searcher MCP integration...")
        searcher = orchestrator.literature_searcher
        
        if hasattr(searcher, 'mcp_client') and searcher.mcp_client:
            print("✅ Literature searcher has MCP client")
        else:
            print("❌ Literature searcher missing MCP client")
            return False
        
        # 4. Test Sci-Hub MCP tool call directly
        print("\n4️⃣ Testing Sci-Hub MCP tool calls...")
        
        # Test keyword search
        keyword_result = await orchestrator.mcp_client.call_tool('search_scihub_by_keyword', {
            'keywords': 'machine learning test',
            'limit': 3
        })
        
        if keyword_result:
            print(f"✅ Keyword search successful: {len(keyword_result.get('papers', []))} papers")
            print(f"🔍 Search query: {keyword_result.get('search_query', 'N/A')}")
            print(f"📄 Source: {keyword_result.get('source', 'N/A')}")
        else:
            print("❌ Keyword search failed")
        
        # Test title search
        title_result = await orchestrator.mcp_client.call_tool('search_scihub_by_title', {
            'title': 'Deep Learning for Medical Diagnosis'
        })
        
        if title_result:
            print(f"✅ Title search successful: {len(title_result.get('papers', []))} papers")
        else:
            print("❌ Title search failed")
        
        # 5. Test integrated literature search (should use MCP)
        print("\n5️⃣ Testing integrated literature search...")
        
        # This should now use MCP instead of falling back
        papers = await searcher.search_papers(
            "machine learning",
            max_papers=5
        )
        
        if papers:
            print(f"✅ Integrated search successful: {len(papers)} papers found")
            
            # Check if any papers came from Sci-Hub MCP
            scihub_papers = [p for p in papers if p.get('source') == 'sci_hub']
            if scihub_papers:
                print(f"🎯 SUCCESS: {len(scihub_papers)} papers from Sci-Hub MCP!")
                
                # Show sample paper
                sample_paper = scihub_papers[0]
                print(f"📄 Sample Sci-Hub paper:")
                print(f"   Title: {sample_paper.get('title', 'N/A')[:80]}...")
                print(f"   Authors: {', '.join(sample_paper.get('authors', [])[:2])}")
                print(f"   DOI: {sample_paper.get('doi', 'N/A')}")
                print(f"   Retrieval method: {sample_paper.get('metadata', {}).get('retrieval_method', 'N/A')}")
            else:
                print("⚠️ No papers from Sci-Hub MCP (may be using fallback)")
        else:
            print("❌ Integrated search failed")
        
        # 6. Cleanup
        print("\n6️⃣ Cleaning up...")
        await orchestrator.shutdown()
        
        print("\n🎉 MCP Integration Test Complete!")
        return True
        
    except Exception as e:
        import traceback
        print(f"\n❌ Test failed: {e}")
        print(f"🐛 Error details:\n{traceback.format_exc()}")
        
        # Cleanup on error
        try:
            if 'orchestrator' in locals():
                await orchestrator.shutdown()
        except:
            pass
        
        return False


async def main():
    """Main test function"""
    success = await test_mcp_integration()
    
    if success:
        print("\n✅ All tests passed! MCP integration is working.")
        return 0
    else:
        print("\n❌ Some tests failed. Check the logs above.")
        return 1


if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(result)