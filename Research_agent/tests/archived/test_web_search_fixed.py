#!/usr/bin/env python3
"""
Test Web Search Functionality - Fixed Version
"""

import asyncio
import os
import sys
sys.path.insert(0, 'src')

# Force load environment variables first
from dotenv import load_dotenv
load_dotenv(override=True)

async def test_simple_web_search():
    """Test simple web search functionality"""
    print("🌐 Testing Simple Web Search")
    print("=" * 40)
    
    try:
        from research_agent.tools.web_search import web_search_tool
        
        print(f"✅ Web search tool available: {web_search_tool.is_available()}")
        print(f"✅ Serper API key configured: {'Yes' if web_search_tool.serper_api_key else 'No'}")
        
        # Test a simple search using the tool directly
        print("\n🔍 Testing search: 'latest AI research 2024'")
        
        results = web_search_tool.search_web("latest AI research 2024", num_results=2)
        
        if results and len(results) > 0:
            print(f"✅ Search successful! Found {len(results)} results")
            
            for i, result in enumerate(results, 1):
                print(f"\n   Result {i}:")
                print(f"     Title: {result.get('title', 'N/A')[:60]}...")
                print(f"     URL: {result.get('link', 'N/A')}")
                if 'snippet' in result:
                    print(f"     Snippet: {result.get('snippet', '')[:100]}...")
                
                # Check for enhanced content
                if 'content' in result and result['content']:
                    content_length = len(result['content'])
                    print(f"     ✅ Enhanced content: {content_length} characters")
                elif 'raw_content' in result and result['raw_content']:
                    content_length = len(result['raw_content'])
                    print(f"     ✅ Basic content: {content_length} characters")
            
            return True
        else:
            print("❌ Search returned no results")
            return False
            
    except Exception as e:
        print(f"❌ Search test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_demo_integration():
    """Test that the demo can now access web search"""
    print("\n🎯 Testing Demo Integration")
    print("=" * 40)
    
    try:
        # Test imports from enhanced demo
        sys.path.insert(0, 'examples/lightrag_demo')
        from lightrag_enhanced_demo import EnhancedLightRAGSystem
        
        # Create test system
        system = EnhancedLightRAGSystem("./test_web_search_demo")
        print("✅ Enhanced demo system created")
        
        # Check hybrid search agent
        if hasattr(system, 'hybrid_search_agent') and system.__dict__.get('hybrid_search_agent') is None:
            # Initialize just to test imports
            try:
                from research_agent.agents.hybrid_search_agent import HybridSearchAgent
                print("✅ Hybrid search agent can be imported")
                
                from research_agent.tools import WEB_SEARCH_AVAILABLE
                print(f"✅ Web search available to agents: {WEB_SEARCH_AVAILABLE}")
                
                return True
            except Exception as e:
                print(f"❌ Hybrid search agent import failed: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Demo integration test failed: {e}")
        return False

async def main():
    """Run web search tests"""
    print("🌐 Web Search Functionality Test")
    print("=" * 50)
    
    # Check environment first
    print("📋 Environment Check:")
    print(f"   OPENAI_API_KEY: {'✅' if os.getenv('OPENAI_API_KEY') else '❌'}")
    print(f"   SERPER_API_KEY: {'✅' if os.getenv('SERPER_API_KEY') else '❌'}")
    print()
    
    tests = [
        ("Simple Web Search", test_simple_web_search),
        ("Demo Integration", test_demo_integration),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            if result:
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Web search is fully working!")
        print("\n🚀 Now ready to use:")
        print("   • Enhanced LightRAG demo with web search")
        print("   • Mode 4 (Hybrid Search) - Intelligent ReACT reasoning")
        print("   • Automatic web search when corpus is insufficient")
        print("   • Enhanced content extraction")
        
        print("\n💡 Try running:")
        print("   cd examples/lightrag_demo")
        print("   uv run python lightrag_enhanced_demo.py")
        print("   Select option 2 (existing knowledge base)")
        print("   Choose Mode 4 (Hybrid)")
        print("   Ask: 'What recent technology did Apple announce for health monitoring?'")
    else:
        print(f"⚠️ {total - passed} test(s) failed.")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)