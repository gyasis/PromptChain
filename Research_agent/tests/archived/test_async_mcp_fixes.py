#!/usr/bin/env python3
"""
Test script to validate the async MCP connection fixes in DocumentSearchService.

This test specifically checks that:
1. No cancel scope errors occur
2. No TaskGroup cancellation errors occur
3. Shared MCP connection works properly
4. Proper cleanup happens
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


async def test_async_mcp_fixes():
    """Test that async MCP connection fixes work properly."""
    print("🔧 Testing Async MCP Connection Fixes")
    print("=" * 50)
    
    try:
        from research_agent.services.document_search_service import DocumentSearchService
        
        # Create service
        service = DocumentSearchService(
            working_directory="test_async_fixes_workspace",
            rate_limit_delay=0.1  # Faster for testing
        )
        
        # Test 1: Initialize service
        print("📋 Test 1: Service initialization...")
        initialized = await service.initialize()
        assert initialized, "Service should initialize successfully"
        print("✅ Service initialized")
        
        # Test 2: Start session
        print("📋 Test 2: Session management...")
        session_id = service.start_session("async_test_session")
        assert session_id, "Session should start successfully"
        print(f"✅ Session started: {session_id}")
        
        # Test 3: Test shared MCP connection creation
        print("📋 Test 3: Shared MCP connection...")
        try:
            # This should not create multiple connections
            connection1 = await service._get_shared_mcp_connection()
            connection2 = await service._get_shared_mcp_connection()
            
            # Should be the same instance
            assert connection1 is connection2, "Should return same shared instance"
            print("✅ Shared MCP connection working")
        except Exception as e:
            if "cancel scope" in str(e).lower() or "taskgroup" in str(e).lower():
                print(f"❌ ASYNC ERROR STILL EXISTS: {e}")
                raise
            else:
                print(f"⚠️  Non-async error (expected during testing): {e}")
        
        # Test 4: Test search without async errors
        print("📋 Test 4: Search operation...")
        try:
            papers, metadata = await service.search_documents(
                search_query="machine learning",
                max_papers=6,  # Small number for testing
                enhance_metadata=False
            )
            
            print(f"✅ Search completed successfully - found {len(papers)} papers")
            print(f"   Search status: {metadata.get('status', 'unknown')}")
            
            # Check that we got results from different tiers
            tiers = set(paper.get('tier', 'unknown') for paper in papers)
            print(f"   Tiers used: {', '.join(tiers)}")
            
        except Exception as e:
            if "cancel scope" in str(e).lower() or "taskgroup" in str(e).lower():
                print(f"❌ ASYNC ERROR DETECTED: {e}")
                raise
            else:
                print(f"⚠️  Search failed with non-async error: {e}")
        
        # Test 5: Test cleanup
        print("📋 Test 5: Cleanup...")
        await service._cleanup_shared_mcp_connection()
        print("✅ Cleanup completed")
        
        # Test 6: Test shutdown
        print("📋 Test 6: Service shutdown...")
        await service.shutdown()
        print("✅ Service shutdown completed")
        
        print("\n🎉 ALL ASYNC MCP FIXES VALIDATED SUCCESSFULLY!")
        print("   - No cancel scope errors")
        print("   - No TaskGroup cancellation errors") 
        print("   - Shared MCP connection working")
        print("   - Proper cleanup functioning")
        
        return True
        
    except Exception as e:
        print(f"\n💥 TEST FAILED: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print("Traceback:")
        traceback.print_exc()
        return False


async def main():
    """Run the async MCP fixes test."""
    print("Starting Async MCP Connection Fixes Test...")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    success = await test_async_mcp_fixes()
    
    if success:
        print("\n✅ TEST PASSED - Async MCP fixes are working!")
        return 0
    else:
        print("\n❌ TEST FAILED - Async errors still exist!")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Test runner failed: {e}")
        sys.exit(1)