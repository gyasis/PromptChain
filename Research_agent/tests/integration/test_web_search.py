#!/usr/bin/env python3
"""
Test Web Search Functionality
Verifies that the Serper API key is working and web search is functional
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project paths
sys.path.insert(0, 'src')

async def test_web_search_functionality():
    """Test the web search tool with Serper API"""
    print("🌐 Testing Web Search Functionality")
    print("=" * 50)
    
    try:
        from research_agent.tools.web_search import web_search_tool, web_search
        
        # Check if web search is available
        if not web_search_tool.is_available():
            print("❌ Web search tool not available")
            return False
        
        print("✅ Web search tool is available")
        print(f"   Using: {web_search_tool.__class__.__name__}")
        print(f"   Advanced scraping: {web_search_tool.use_advanced_scraping}")
        
        # Test a simple search
        print("\n🔍 Testing search: 'latest AI research 2024'")
        
        try:
            results = await web_search("latest AI research 2024", num_results=2)
            
            if results and len(results) > 0:
                print(f"✅ Search successful! Found {len(results)} results")
                
                for i, result in enumerate(results, 1):
                    print(f"\n   Result {i}:")
                    print(f"     Title: {result.get('title', 'N/A')[:60]}...")
                    print(f"     URL: {result.get('link', 'N/A')}")
                    print(f"     Snippet: {result.get('snippet', 'N/A')[:100]}...")
                    
                    if 'content' in result:
                        content_length = len(result['content'])
                        print(f"     Content: {content_length} characters extracted")
                
                return True
            else:
                print("❌ Search returned no results")
                return False
                
        except Exception as e:
            print(f"❌ Search failed: {e}")
            return False
            
    except ImportError as e:
        print(f"❌ Failed to import web search tools: {e}")
        return False

async def test_environment_configuration():
    """Test environment variable configuration"""
    print("\n⚙️ Testing Environment Configuration")
    print("=" * 50)
    
    # Check required environment variables
    env_vars = {
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
        'SERPER_API_KEY': os.getenv('SERPER_API_KEY'),
        'LIGHTRAG_ENABLE_RERANK': os.getenv('LIGHTRAG_ENABLE_RERANK', 'true'),
    }
    
    all_good = True
    
    for var_name, var_value in env_vars.items():
        if var_value:
            if var_name.endswith('_KEY'):
                # Show partial key for security
                masked_value = f"{var_value[:10]}...{var_value[-4:]}"
                print(f"✅ {var_name}: {masked_value}")
            else:
                print(f"✅ {var_name}: {var_value}")
        else:
            print(f"❌ {var_name}: Not configured")
            if var_name in ['OPENAI_API_KEY', 'SERPER_API_KEY']:
                all_good = False
    
    return all_good

async def test_hybrid_search_integration():
    """Test that hybrid search agent has web search capability"""
    print("\n🧠 Testing Hybrid Search Integration")
    print("=" * 50)
    
    try:
        from research_agent.agents.hybrid_search_agent import HybridSearchAgent
        from research_agent.tools import WEB_SEARCH_AVAILABLE
        
        print(f"✅ HybridSearchAgent imported successfully")
        print(f"✅ WEB_SEARCH_AVAILABLE: {WEB_SEARCH_AVAILABLE}")
        
        if WEB_SEARCH_AVAILABLE:
            print("✅ Hybrid search agent has web search capability")
            print("   Ready for intelligent corpus + web search decisions")
            return True
        else:
            print("⚠️ Web search not available to hybrid search agent")
            return False
        
    except ImportError as e:
        print(f"❌ Failed to import hybrid search agent: {e}")
        return False

async def main():
    """Run all web search tests"""
    print("🌐 Web Search Configuration Test")
    print("=" * 60)
    
    tests = [
        ("Environment Configuration", test_environment_configuration),
        ("Web Search Functionality", test_web_search_functionality),
        ("Hybrid Search Integration", test_hybrid_search_integration),
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
    
    print("\n" + "=" * 60)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Web search is fully configured and working!")
        print("\n🚀 Ready to use:")
        print("   • Mode 4 (Hybrid Search) in LightRAG demo")
        print("   • Intelligent corpus analysis + web search decisions")
        print("   • Enhanced content extraction with crawl4ai")
        print("   • ReACT-style reasoning for search necessity")
        print("\n💡 Try asking questions like:")
        print("   • 'What recent technology did Apple announce for health monitoring?'")
        print("   • 'How do current AR developments relate to neurological research?'")
    else:
        print(f"⚠️ {total - passed} test(s) failed. Check the configuration above.")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)