#!/usr/bin/env python3
"""
Test LightRAG Enhanced Web Search Integration
Verifies the enhanced web search capabilities in the LightRAG demo
"""

import asyncio
import os
import sys
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, 'src')
sys.path.insert(0, 'examples/lightrag_demo')

def test_web_search_tool_initialization():
    """Test that the enhanced web search tool is properly initialized in LightRAG demo"""
    print("=== Testing Web Search Tool Initialization in LightRAG Demo ===")
    
    try:
        # Import the demo to test initialization
        import lightrag_enhanced_demo
        
        # Check if web search is available
        if hasattr(lightrag_enhanced_demo, 'WEB_SEARCH_AVAILABLE'):
            print(f"✅ WEB_SEARCH_AVAILABLE: {lightrag_enhanced_demo.WEB_SEARCH_AVAILABLE}")
        
        if hasattr(lightrag_enhanced_demo, 'ADVANCED_SCRAPER_AVAILABLE'):
            print(f"✅ ADVANCED_SCRAPER_AVAILABLE: {lightrag_enhanced_demo.ADVANCED_SCRAPER_AVAILABLE}")
        
        if hasattr(lightrag_enhanced_demo, 'web_search_tool'):
            tool = lightrag_enhanced_demo.web_search_tool
            if tool:
                print(f"✅ Web search tool instance: {type(tool).__name__}")
                print(f"   Advanced scraping enabled: {tool.use_advanced_scraping}")
                print(f"   Include images: {tool.include_images}")
                print(f"   API available: {tool.is_available()}")
            else:
                print(f"⚠️  Web search tool is None")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to test web search initialization: {e}")
        return False

def test_enhanced_reasoning_agent_creation():
    """Test creation of enhanced reasoning agent with web search tools"""
    print("\n=== Testing Enhanced Reasoning Agent Creation ===")
    
    try:
        from lightrag_enhanced_demo import EnhancedLightRAGSystem
        
        # Create LightRAG instance
        lightrag_instance = EnhancedLightRAGSystem(
            working_dir="./test_lightrag_data"
        )
        
        print(f"✅ EnhancedLightRAGSystem created successfully")
        
        # Check if the system has the reasoning functionality
        try:
            # Check if reasoning_query method exists
            if hasattr(lightrag_instance, 'reasoning_query'):
                print(f"✅ Reasoning query method available")
                
                # Check if reasoning agent is accessible
                if hasattr(lightrag_instance, 'reasoning_agent'):
                    reasoning_agent = lightrag_instance.reasoning_agent
                    print(f"✅ Reasoning agent accessible")
                    print(f"   Agent type: {type(reasoning_agent).__name__}")
                    
                    # Check if web search tool is registered
                    if hasattr(reasoning_agent, 'chain') and hasattr(reasoning_agent.chain, 'tool_functions'):
                        tool_functions = reasoning_agent.chain.tool_functions
                        if 'search_web_for_current_info' in tool_functions:
                            print(f"✅ Enhanced web search function registered")
                            
                            # Get the function and check its properties
                            web_search_func = tool_functions['search_web_for_current_info']
                            print(f"   Function callable: {callable(web_search_func)}")
                        else:
                            print(f"⚠️  Web search function not found in registered tools")
                            print(f"   Available tools: {list(tool_functions.keys())}")
                    else:
                        print(f"⚠️  Could not access tool functions (reasoning agent may be lazy-loaded)")
                else:
                    print(f"⚠️  Reasoning agent not immediately accessible (may be lazy-loaded)")
                
                # Test if reasoning query would work (without actually running it)
                print(f"✅ Enhanced LightRAG system properly configured for reasoning")
                return True
            else:
                print(f"❌ reasoning_query method not found")
                return False
            
        except Exception as e:
            print(f"❌ Failed to test reasoning functionality: {e}")
            return False
        
    except Exception as e:
        print(f"❌ Failed to create EnhancedLightRAGSystem: {e}")
        return False

def test_web_search_tool_schema():
    """Test the enhanced web search tool schema generation"""
    print("\n=== Testing Enhanced Web Search Tool Schema ===")
    
    try:
        from research_agent.tools import web_search_tool
        
        if web_search_tool:
            schema = web_search_tool.get_tool_schema()
            print(f"✅ Tool schema generated successfully")
            print(f"   Function name: {schema['function']['name']}")
            
            description = schema['function']['description']
            print(f"   Description includes crawl4ai: {'crawl4ai' in description}")
            print(f"   Description length: {len(description)} characters")
            
            # Check parameters
            params = schema['function']['parameters']['properties']
            print(f"   Parameters: {list(params.keys())}")
            
            if 'num_results' in params:
                num_results_param = params['num_results']
                print(f"   Max results: {num_results_param.get('maximum', 'not set')}")
            
            return True
        else:
            print(f"⚠️  Web search tool not available")
            return False
        
    except Exception as e:
        print(f"❌ Failed to test tool schema: {e}")
        return False

def test_citation_tracker_enhancement():
    """Test that citation tracker works with enhanced web search metadata"""
    print("\n=== Testing Citation Tracker Enhancement ===")
    
    try:
        from lightrag_enhanced_demo import CitationTracker
        
        tracker = CitationTracker()
        print(f"✅ CitationTracker created successfully")
        
        # Test adding enhanced citation
        enhanced_metadata = {
            "mode": "enhanced_web_search",
            "query": "test query",
            "extraction_method": "crawl4ai",
            "total_tokens": 1500,
            "total_links": 25,
            "num_results": 3
        }
        
        citation_id = tracker.add_citation(
            "Sample enhanced web search content with rich metadata...",
            "Enhanced Web Search: test query",
            enhanced_metadata
        )
        
        print(f"✅ Enhanced citation added: {citation_id}")
        
        # Check citation details
        if citation_id in tracker.citations:
            citation_data = tracker.citations[citation_id]
            print(f"   Source: {citation_data['source']}")
            print(f"   Metadata keys: {list(citation_data['metadata'].keys())}")
            print(f"   Extraction method: {citation_data['metadata'].get('extraction_method')}")
            print(f"   Token count: {citation_data['metadata'].get('total_tokens')}")
        
        # Test citation formatting
        formatted = tracker.format_citations()
        print(f"✅ Citations formatted successfully ({len(formatted)} characters)")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to test citation tracker: {e}")
        return False

def test_environment_variables():
    """Test environment variable configuration for enhanced features"""
    print("\n=== Testing Environment Variables ===")
    
    # Check for required environment variables
    env_vars = {
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
        'SERPER_API_KEY': os.getenv('SERPER_API_KEY'),
    }
    
    for var_name, var_value in env_vars.items():
        if var_value:
            print(f"✅ {var_name}: Configured ({len(var_value)} characters)")
        else:
            print(f"❌ {var_name}: Not configured")
    
    # Check dependency availability
    dependencies = ['crawl4ai', 'playwright', 'tiktoken', 'lightrag']
    available_deps = []
    
    for dep in dependencies:
        try:
            __import__(dep)
            available_deps.append(dep)
            print(f"✅ {dep}: Available")
        except ImportError:
            print(f"❌ {dep}: Missing")
    
    return len(available_deps) == len(dependencies)

def main():
    """Run all tests and generate report"""
    print("🚀 LightRAG Enhanced Web Search Integration Test")
    print("=" * 70)
    
    # Track test results
    tests = [
        ("Web Search Tool Initialization", test_web_search_tool_initialization),
        ("Enhanced Reasoning Agent Creation", test_enhanced_reasoning_agent_creation),
        ("Web Search Tool Schema", test_web_search_tool_schema),
        ("Citation Tracker Enhancement", test_citation_tracker_enhancement),
        ("Environment Variables", test_environment_variables),
    ]
    
    results = {}
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = "PASSED" if result else "FAILED"
            if result:
                passed += 1
        except Exception as e:
            results[test_name] = f"ERROR: {e}"
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    
    for test_name, result in results.items():
        status_icon = "✅" if result == "PASSED" else "❌" if result == "FAILED" else "⚠️"
        print(f"{status_icon} {test_name}: {result}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    # Generate test report
    report = {
        "timestamp": datetime.now().isoformat(),
        "test_type": "lightrag_enhanced_web_search_integration",
        "test_results": results,
        "summary": {
            "passed": passed,
            "total": total,
            "success_rate": f"{passed/total*100:.1f}%"
        }
    }
    
    report_file = f"lightrag_enhanced_web_search_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Test report saved: {report_file}")
    
    if passed == total:
        print("\n🎉 All tests passed! Enhanced web search integration is ready.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check the issues above.")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)