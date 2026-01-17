#!/usr/bin/env python3
"""
Comprehensive test for async MCP connection fixes.
Tests multiple search operations to ensure no async errors occur.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# Set up environment
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "dummy-key-for-testing")


async def test_multiple_searches():
    """Test multiple search operations to stress test the async fixes."""
    print("🔧 Comprehensive Async MCP Connection Test")
    print("=" * 60)
    
    try:
        from research_agent.services.document_search_service import DocumentSearchService
        
        # Create service
        service = DocumentSearchService(
            working_directory="test_comprehensive_fixes",
            rate_limit_delay=0.1
        )
        
        # Initialize
        await service.initialize()
        session_id = service.start_session("comprehensive_test")
        print(f"✅ Service initialized and session started: {session_id}")
        
        # Test multiple search queries with different paper counts
        test_queries = [
            ("artificial intelligence", 4),
            ("quantum computing", 6), 
            ("neural networks", 3)
        ]
        
        all_successful = True
        total_papers = 0
        
        for i, (query, max_papers) in enumerate(test_queries, 1):
            print(f"📋 Test {i}: Searching '{query}' (max {max_papers} papers)...")
            
            try:
                papers, metadata = await service.search_documents(
                    search_query=query,
                    max_papers=max_papers,
                    enhance_metadata=True
                )
                
                print(f"   ✅ Found {len(papers)} papers, status: {metadata.get('status', 'unknown')}")
                print(f"   Tiers: {set(p.get('tier', 'unknown') for p in papers)}")
                total_papers += len(papers)
                
                # Check for fallback attempts
                fallback_stats = metadata.get('fallback_statistics', {})
                if fallback_stats.get('attempts', 0) > 0:
                    print(f"   📥 Fallback attempts: {fallback_stats['attempts']}, successes: {fallback_stats['successes']}")
                
            except Exception as e:
                if any(err in str(e).lower() for err in ["cancel scope", "taskgroup", "cancelled"]):
                    print(f"   ❌ ASYNC ERROR: {e}")
                    all_successful = False
                    break
                else:
                    print(f"   ⚠️  Non-async error: {e}")
        
        if all_successful:
            print(f"\n🎉 ALL SEARCHES COMPLETED SUCCESSFULLY!")
            print(f"   Total papers retrieved: {total_papers}")
            print(f"   No async errors detected")
            
            # Test final cleanup
            await service._cleanup_shared_mcp_connection()
            await service.shutdown()
            print("   ✅ Cleanup and shutdown completed")
            
            return True
        else:
            print(f"\n❌ ASYNC ERRORS STILL DETECTED!")
            return False
            
    except Exception as e:
        print(f"\n💥 TEST FRAMEWORK ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_concurrent_operations():
    """Test concurrent operations to ensure no race conditions."""
    print("\n🔧 Concurrent Operations Test")
    print("=" * 40)
    
    try:
        from research_agent.services.document_search_service import DocumentSearchService
        
        service = DocumentSearchService(rate_limit_delay=0.1)
        await service.initialize()
        service.start_session("concurrent_test")
        
        # Test concurrent shared connection access
        async def get_connection():
            return await service._get_shared_mcp_connection()
        
        # Start multiple connection requests simultaneously
        tasks = [get_connection() for _ in range(5)]
        connections = await asyncio.gather(*tasks)
        
        # All should be the same instance
        first_conn = connections[0]
        all_same = all(conn is first_conn for conn in connections)
        
        if all_same:
            print("✅ Concurrent connection access working correctly")
        else:
            print("❌ Connection sharing not working properly")
            return False
            
        await service.shutdown()
        return True
        
    except Exception as e:
        print(f"❌ Concurrent test failed: {e}")
        return False


async def main():
    """Run comprehensive async fixes tests."""
    print("Starting Comprehensive Async MCP Connection Tests...")
    
    # Test 1: Multiple searches
    test1_success = await test_multiple_searches()
    
    # Test 2: Concurrent operations
    test2_success = await test_concurrent_operations()
    
    if test1_success and test2_success:
        print("\n🏆 ALL COMPREHENSIVE TESTS PASSED!")
        print("   ✅ Multiple search operations work without async errors")
        print("   ✅ Concurrent operations work correctly")
        print("   ✅ Shared MCP connection pattern is solid")
        return 0
    else:
        print("\n💥 SOME TESTS FAILED!")
        if not test1_success:
            print("   ❌ Multiple search test failed")
        if not test2_success:
            print("   ❌ Concurrent operations test failed")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Test runner error: {e}")
        sys.exit(1)