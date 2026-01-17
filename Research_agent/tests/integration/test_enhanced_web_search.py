#!/usr/bin/env python3
"""
Test Enhanced Web Search Tool with crawl4ai Integration
Verifies both basic and advanced content extraction capabilities
"""

import asyncio
import os
import sys
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, 'src')

def test_web_search_availability():
    """Test basic web search tool availability"""
    print("=== Testing Web Search Availability ===")
    
    try:
        from research_agent.tools import WebSearchTool, ADVANCED_SCRAPER_AVAILABLE
        
        # Test basic initialization
        tool = WebSearchTool(use_advanced_scraping=False)
        print(f"✅ Basic WebSearchTool initialized")
        print(f"   API Available: {tool.is_available()}")
        
        # Test advanced initialization
        if ADVANCED_SCRAPER_AVAILABLE:
            advanced_tool = WebSearchTool(use_advanced_scraping=True)
            print(f"✅ Advanced WebSearchTool initialized with crawl4ai")
            print(f"   Advanced scraping enabled: {advanced_tool.use_advanced_scraping}")
        else:
            print(f"⚠️  Advanced scraper (crawl4ai) not available")
        
        print(f"   Advanced scraper available: {ADVANCED_SCRAPER_AVAILABLE}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize WebSearchTool: {e}")
        return False

def test_advanced_page_scraper():
    """Test the AdvancedPageScraper directly"""
    print("\n=== Testing AdvancedPageScraper ===")
    
    try:
        from research_agent.tools.singlepage_advanced import AdvancedPageScraper
        
        async def test_scraper():
            # Test with a simple, reliable URL
            test_url = "https://httpbin.org/html"
            
            print(f"Testing with URL: {test_url}")
            
            async with AdvancedPageScraper(include_images=False) as scraper:
                result = await scraper.scrape_url(test_url)
                
                print(f"✅ Advanced scraping completed")
                print(f"   Content length: {len(result.get('content', ''))}")
                print(f"   Token count: {result.get('token_count', 0)}")
                print(f"   Links found: {len(result.get('links', []))}")
                
                # Show a snippet of content
                content = result.get('content', '')
                if content:
                    snippet = content[:200] + "..." if len(content) > 200 else content
                    print(f"   Content snippet: {snippet}")
                
                return True
        
        # Run the async test
        return asyncio.run(test_scraper())
        
    except ImportError as e:
        print(f"⚠️  AdvancedPageScraper not available: {e}")
        return False
    except Exception as e:
        print(f"❌ AdvancedPageScraper test failed: {e}")
        return False

def test_web_search_tool_schema():
    """Test web search tool schema generation"""
    print("\n=== Testing Tool Schema ===")
    
    try:
        from research_agent.tools import WebSearchTool
        
        # Test basic tool schema
        basic_tool = WebSearchTool(use_advanced_scraping=False)
        basic_schema = basic_tool.get_tool_schema()
        
        print(f"✅ Basic tool schema generated")
        print(f"   Function name: {basic_schema['function']['name']}")
        print(f"   Description length: {len(basic_schema['function']['description'])}")
        
        # Test advanced tool schema if available
        try:
            advanced_tool = WebSearchTool(use_advanced_scraping=True)
            advanced_schema = advanced_tool.get_tool_schema()
            
            print(f"✅ Advanced tool schema generated")
            print(f"   Enhanced description: {'crawl4ai' in advanced_schema['function']['description']}")
            
        except Exception as e:
            print(f"⚠️  Advanced tool schema failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Tool schema test failed: {e}")
        return False

def test_env_setup_instructions():
    """Test environment setup instructions"""
    print("\n=== Testing Environment Setup ===")
    
    try:
        from research_agent.tools import ENV_SETUP_INSTRUCTIONS
        
        print(f"✅ Environment setup instructions available")
        print(f"   Instructions length: {len(ENV_SETUP_INSTRUCTIONS)}")
        print(f"   Contains crawl4ai info: {'crawl4ai' in ENV_SETUP_INSTRUCTIONS}")
        
        # Check if Serper API key is configured
        serper_key = os.getenv('SERPER_API_KEY')
        if serper_key:
            print(f"✅ SERPER_API_KEY configured ({len(serper_key)} characters)")
        else:
            print(f"⚠️  SERPER_API_KEY not configured")
        
        return True
        
    except Exception as e:
        print(f"❌ Environment setup test failed: {e}")
        return False

def test_import_integration():
    """Test proper import integration"""
    print("\n=== Testing Import Integration ===")
    
    try:
        # Test main imports
        from research_agent.tools import WebSearchTool, web_search_tool, web_search
        print(f"✅ Main imports successful")
        
        # Test global instance
        print(f"   Global tool initialized: {web_search_tool is not None}")
        print(f"   Global tool advanced: {web_search_tool.use_advanced_scraping}")
        
        # Test function wrapper
        print(f"   Function wrapper available: {callable(web_search)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Import integration test failed: {e}")
        return False

def main():
    """Run all tests and generate report"""
    print("🔍 Enhanced Web Search Tool Integration Test")
    print("=" * 60)
    
    # Track test results
    tests = [
        ("Web Search Availability", test_web_search_availability),
        ("Advanced Page Scraper", test_advanced_page_scraper),
        ("Tool Schema Generation", test_web_search_tool_schema),
        ("Environment Setup", test_env_setup_instructions),
        ("Import Integration", test_import_integration),
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
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    for test_name, result in results.items():
        status_icon = "✅" if result == "PASSED" else "❌" if result == "FAILED" else "⚠️"
        print(f"{status_icon} {test_name}: {result}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    # Environment status
    print("\n📋 ENVIRONMENT STATUS")
    print("-" * 30)
    
    # Check dependencies
    dependencies = {
        "crawl4ai": False,
        "playwright": False,
        "tiktoken": False,
        "SERPER_API_KEY": bool(os.getenv('SERPER_API_KEY'))
    }
    
    for dep in ["crawl4ai", "playwright", "tiktoken"]:
        try:
            __import__(dep)
            dependencies[dep] = True
        except ImportError:
            pass
    
    for dep, available in dependencies.items():
        status = "✅" if available else "❌"
        print(f"{status} {dep}: {'Available' if available else 'Missing'}")
    
    # Generate test report
    report = {
        "timestamp": datetime.now().isoformat(),
        "test_results": results,
        "environment": dependencies,
        "summary": {
            "passed": passed,
            "total": total,
            "success_rate": f"{passed/total*100:.1f}%"
        }
    }
    
    report_file = f"enhanced_web_search_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Test report saved: {report_file}")
    
    # Exit with appropriate code
    return 0 if passed == total else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)